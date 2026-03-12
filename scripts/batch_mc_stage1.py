#!/usr/bin/env python3
"""
Batch Stage 1 pipeline for Modernist Cuisine vol3/vol4/vol1.

Pipeline per book:
0. Convert EPUB to PDF when needed.
1. Extract markdown with MinerU, auto-splitting PDFs when limits are exceeded.
2. Render PDF pages and run DashScope qwen3-vl-plus for tables/figures.
3. Merge MinerU markdown with vision replacements.
4. Split merged markdown into chapter-local chunks with qwen3.5:2b via Ollama.
5. Annotate chunks with summary/topics using qwen3.5:9b via Ollama.

The script is resumable at both batch and per-step level.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz
import requests


DEFAULT_BOOKS: List[Dict[str, str]] = [
    {
        "id": "mc_vol3",
        "path": "/Users/jeff/Documents/厨书数据库/工具科学书/Volume 3 - Animals and Plants.pdf",
        "type": "pdf",
    },
    {
        "id": "mc_vol4",
        "path": "/Users/jeff/Documents/厨书数据库/工具科学书/Volume 4 - Ingredients and Preparations.pdf",
        "type": "pdf",
    },
    {
        "id": "mc_vol1",
        "path": "/Users/jeff/Documents/厨书数据库/工具科学书/volume-1-History+and+Fundamentals.epub",
        "type": "epub",
    },
]

VALID_TOPICS = [
    "heat_transfer",
    "chemical_reaction",
    "physical_change",
    "water_activity",
    "protein_science",
    "lipid_science",
    "carbohydrate",
    "enzyme",
    "flavor_sensory",
    "fermentation",
    "food_safety",
    "emulsion_colloid",
    "color_pigment",
    "equipment_physics",
]

STATUS_ORDER = [
    "pending",
    "step0_done",
    "step1_done",
    "step2_done",
    "step3_done",
    "step4_done",
    "completed",
    "failed",
]

MINERU_BASE_URL = "https://mineru.net/api/v4"
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "qwen3-vl-plus"
SPLIT_MODEL = "qwen3.5:2b"
ANNOTATE_MODEL = "qwen3.5:9b"
PNG_DPI = 150
MINERU_MAX_PAGES = 200
MINERU_MAX_MB = 100.0
MINERU_DAILY_PAGE_LIMIT = 2000

VISION_PROMPT = """请逐一提取这一页的每个独立内容元素，用JSON数组输出：

[
  {
    "type": "table",
    "content": "完整Markdown表格"
  },
  {
    "type": "figure",
    "content": "图片描述1-2句"
  },
  {
    "type": "text_block",
    "content": "独立文字段落（如有）"
  }
]

规则：
- 每个表格单独一个元素。
- 每张图片、示意图或图表单独一个元素。
- 有几个就输出几个，不要合并。
- 不要抄写普通正文段落，只保留视觉相关内容。
- 只输出JSON数组，不要其他说明。
"""

SPLIT_PROMPT_TEMPLATE = """You are splitting a book chapter into semantic chunks.
Return JSON only with this schema:
{{"chunks": ["chunk text 1", "chunk text 2"]}}

Rules:
- Keep original language and wording from the source.
- Split on semantic boundaries.
- Keep each chunk around 300-500 Chinese characters, or roughly 250-450 English words.
- Do not merge content across chapter boundaries.
- Keep tables, lists, and figure explanations with their nearby context when possible.
- Do not summarize or rewrite.

Chapter title: {chapter_title}

Chapter text:
{chapter_text}
"""

ANNOTATE_PROMPT_TEMPLATE = """You are annotating a food-science chunk.
Return JSON only with this schema:
{{
  "summary": "Chinese summary within 50 characters",
  "topics": ["one_or_more_allowed_topics"]
}}

Allowed topics:
{topics}

Rules:
- Summary must be concise Chinese and <= 50 characters.
- Topics must be selected only from the allowed list.
- Choose 1-3 topics.
- If uncertain, choose the closest scientifically grounded topic instead of inventing one.

Chunk:
{chunk_text}
"""


class PipelineError(RuntimeError):
    """Raised for expected pipeline failures."""


@dataclass
class BookSpec:
    book_id: str
    path: Path
    file_type: str

    @property
    def slug(self) -> str:
        return self.book_id.replace("mc_", "")


@dataclass
class PdfPart:
    index: int
    path: Path
    page_start: int
    page_end: int

    @property
    def pages(self) -> int:
        return self.page_end - self.page_start + 1

    @property
    def part_id(self) -> str:
        return f"part{self.index}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def today_key() -> str:
    return time.strftime("%Y-%m-%d")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return default
    return json.loads(text)


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_list(path: Path) -> List[Any]:
    data = load_json(path, [])
    return data if isinstance(data, list) else []


def status_rank(status: str) -> int:
    try:
        return STATUS_ORDER.index(status)
    except ValueError:
        return -1


def normalize_books(raw_books: Sequence[Dict[str, Any]]) -> List[BookSpec]:
    books: List[BookSpec] = []
    for item in raw_books:
        books.append(
            BookSpec(
                book_id=str(item["id"]),
                path=Path(str(item["path"])).expanduser(),
                file_type=str(item["type"]).lower(),
            )
        )
    return books


def load_books(args: argparse.Namespace) -> List[BookSpec]:
    if args.books_json:
        raw = json.loads(Path(args.books_json).read_text(encoding="utf-8"))
    else:
        raw = DEFAULT_BOOKS
    return normalize_books(raw)


def load_batch_progress(path: Path) -> Dict[str, Any]:
    progress = load_json(path, {})
    if "_meta" not in progress:
        progress["_meta"] = {"mineru_daily_pages": {}}
    if "mineru_daily_pages" not in progress["_meta"]:
        progress["_meta"]["mineru_daily_pages"] = {}
    return progress


def save_batch_progress(path: Path, progress: Dict[str, Any]) -> None:
    save_json(path, progress)


def ensure_book_progress(progress: Dict[str, Any], book_id: str) -> Dict[str, Any]:
    if book_id not in progress:
        progress[book_id] = {
            "status": "pending",
            "total_chunks": 0,
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "error": None,
        }
    return progress[book_id]


def set_book_status(
    progress: Dict[str, Any],
    path: Path,
    book_id: str,
    *,
    status: Optional[str] = None,
    total_chunks: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    entry = ensure_book_progress(progress, book_id)
    if status is not None:
        entry["status"] = status
    if total_chunks is not None:
        entry["total_chunks"] = total_chunks
    if error is not None or status == "failed":
        entry["error"] = error
    elif status and status != "failed":
        entry["error"] = None
    entry["updated_at"] = now_iso()
    save_batch_progress(path, progress)


def record_daily_pages(progress: Dict[str, Any], path: Path, pages: int) -> None:
    meta = progress["_meta"]["mineru_daily_pages"]
    day = today_key()
    meta[day] = int(meta.get(day, 0)) + int(pages)
    save_batch_progress(path, progress)


def prompt_yes_no(message: str) -> bool:
    print(message, file=sys.stderr)
    if not sys.stdin.isatty():
        return False
    answer = input("Type Y to continue: ").strip().lower()
    return answer == "y"


def guard_daily_quota(progress: Dict[str, Any], path: Path, pages_needed: int) -> None:
    used = int(progress["_meta"]["mineru_daily_pages"].get(today_key(), 0))
    if used + pages_needed <= MINERU_DAILY_PAGE_LIMIT:
        return
    ok = prompt_yes_no(
        f"[warn] MinerU daily quota may be exceeded today: used={used}, "
        f"next_part={pages_needed}, limit={MINERU_DAILY_PAGE_LIMIT}."
    )
    if not ok:
        raise PipelineError("MinerU daily page quota confirmation declined")
    save_batch_progress(path, progress)


def check_command_exists(name: str) -> None:
    if not shutil_which(name):
        raise PipelineError(f"Required command not found: {name}")


def shutil_which(name: str) -> Optional[str]:
    from shutil import which

    return which(name)


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(stripped[start : end + 1])


def extract_json_value(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    starts = [(stripped.find("{"), "{"), (stripped.find("["), "[")]
    starts = [(idx, token) for idx, token in starts if idx != -1]
    if not starts:
        raise ValueError("No JSON object or array found")
    start, token = min(starts, key=lambda item: item[0])
    end_token = "}" if token == "{" else "]"
    end = stripped.rfind(end_token)
    if end == -1 or end <= start:
        raise ValueError("No complete JSON payload found")
    return json.loads(stripped[start : end + 1])


def normalize_vision_parsed(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        elements: List[Dict[str, str]] = []
        tables: List[Dict[str, str]] = []
        figures: List[Dict[str, str]] = []
        text_blocks: List[str] = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            element_type = str(raw.get("type") or "").strip().lower()
            content = str(raw.get("content") or "").strip()
            if not content:
                continue
            if element_type == "table":
                tables.append({"title": "", "markdown": content, "notes": ""})
                elements.append({"type": "table", "content": content})
            elif element_type == "figure":
                figures.append({"type": "figure", "description": content})
                elements.append({"type": "figure", "content": content})
            elif element_type == "text_block":
                text_blocks.append(content)
                elements.append({"type": "text_block", "content": content})
        return {
            "elements": elements,
            "tables": tables,
            "figures": figures,
            "text_blocks": text_blocks,
            "page_note": "",
        }
    if isinstance(payload, dict):
        tables: List[Dict[str, str]] = []
        figures: List[Dict[str, str]] = []
        elements: List[Dict[str, str]] = []
        text_blocks = payload.get("text_blocks") or []
        if not isinstance(text_blocks, list):
            text_blocks = []
        normalized_text_blocks = [str(item).strip() for item in text_blocks if str(item).strip()]
        for table in payload.get("tables") or []:
            if not isinstance(table, dict):
                continue
            markdown = str(table.get("markdown") or "").strip()
            title = str(table.get("title") or "").strip()
            notes = str(table.get("notes") or "").strip()
            if not markdown:
                continue
            table_item = {"title": title, "markdown": markdown, "notes": notes}
            tables.append(table_item)
            elements.append({"type": "table", "content": markdown})
        for figure in payload.get("figures") or []:
            if not isinstance(figure, dict):
                continue
            description = str(figure.get("description") or "").strip()
            figure_type = str(figure.get("type") or "figure").strip() or "figure"
            if not description:
                continue
            figure_item = {"type": figure_type, "description": description}
            figures.append(figure_item)
            elements.append({"type": "figure", "content": description, "figure_type": figure_type})
        page_note = str(payload.get("page_note") or "").strip()
        if page_note:
            normalized_text_blocks.append(page_note)
            elements.append({"type": "text_block", "content": page_note})
        return {
            "elements": elements,
            "tables": tables,
            "figures": figures,
            "text_blocks": normalized_text_blocks,
            "page_note": page_note,
        }
    raise ValueError("Unsupported vision payload type")


def post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: int = 120) -> Dict[str, Any]:
    response = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def run_subprocess(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def resolve_work_pdf(book: BookSpec, book_dir: Path) -> Path:
    if book.file_type == "pdf":
        return book.path
    if book.file_type != "epub":
        raise PipelineError(f"Unsupported file type for {book.book_id}: {book.file_type}")
    check_command_exists("ebook-convert")
    out_pdf = book_dir / f"{book.slug}.pdf"
    if out_pdf.exists() and out_pdf.stat().st_size > 0:
        return out_pdf
    print(f"[step0] converting EPUB to PDF for {book.book_id}")
    run_subprocess(["ebook-convert", str(book.path), str(out_pdf)])
    return out_pdf


def get_pdf_page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    try:
        return doc.page_count
    finally:
        doc.close()


def write_pdf_slice(pdf_path: Path, out_path: Path, start_idx: int, end_idx: int) -> None:
    src = fitz.open(str(pdf_path))
    dst = fitz.open()
    ensure_dir(out_path.parent)
    try:
        dst.insert_pdf(src, from_page=start_idx, to_page=end_idx)
        dst.save(str(out_path))
    finally:
        dst.close()
        src.close()


def split_range_by_size(
    pdf_path: Path,
    base_name: str,
    parts_dir: Path,
    start_idx: int,
    end_idx: int,
    ranges_out: List[Tuple[int, int]],
) -> None:
    temp_path = parts_dir / f"{base_name}_{start_idx+1}_{end_idx+1}.pdf"
    write_pdf_slice(pdf_path, temp_path, start_idx, end_idx)
    size_mb = temp_path.stat().st_size / (1024 * 1024)
    page_count = end_idx - start_idx + 1
    if size_mb <= MINERU_MAX_MB or page_count <= 1:
        ranges_out.append((start_idx, end_idx))
        temp_path.unlink(missing_ok=True)
        return
    temp_path.unlink(missing_ok=True)
    mid = start_idx + ((end_idx - start_idx) // 2)
    split_range_by_size(pdf_path, base_name, parts_dir, start_idx, mid, ranges_out)
    split_range_by_size(pdf_path, base_name, parts_dir, mid + 1, end_idx, ranges_out)


def build_pdf_parts(pdf_path: Path, parts_dir: Path, book_id: str) -> List[PdfPart]:
    total_pages = get_pdf_page_count(pdf_path)
    base_ranges: List[Tuple[int, int]] = []
    for start in range(0, total_pages, MINERU_MAX_PAGES):
        end = min(start + MINERU_MAX_PAGES - 1, total_pages - 1)
        split_range_by_size(pdf_path, book_id, parts_dir, start, end, base_ranges)
    parts: List[PdfPart] = []
    for idx, (start_idx, end_idx) in enumerate(base_ranges, start=1):
        part_path = parts_dir / f"{book_id}_part{idx}.pdf"
        if not part_path.exists():
            write_pdf_slice(pdf_path, part_path, start_idx, end_idx)
        parts.append(PdfPart(index=idx, path=part_path, page_start=start_idx + 1, page_end=end_idx + 1))
    return parts


def mineru_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def request_mineru_upload_url(token: str, filename: str) -> Tuple[str, str]:
    payload = {
        "enable_formula": True,
        "enable_table": True,
        "language": "en",
        "files": [{"name": filename, "is_ocr": False, "data_id": filename}],
    }
    response = post_json(
        f"{MINERU_BASE_URL}/file-urls/batch",
        payload,
        headers=mineru_headers(token),
        timeout=60,
    )
    if response.get("code") != 0:
        raise PipelineError(f"MinerU upload-url request failed: {response}")
    data = response["data"]
    return data["file_urls"][0], data["batch_id"]


def upload_to_presigned_url(upload_url: str, pdf_path: Path) -> None:
    with pdf_path.open("rb") as fh:
        response = requests.put(upload_url, data=fh, timeout=900)
    response.raise_for_status()


def poll_mineru_batch(token: str, batch_id: str, timeout_sec: int = 3600) -> str:
    started = time.time()
    url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    while time.time() - started < timeout_sec:
        response = requests.get(url, headers=mineru_headers(token), timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise PipelineError(f"MinerU poll failed: {payload}")
        result = payload["data"]["extract_result"][0]
        state = result["state"]
        progress = result.get("extract_progress", {})
        print(
            f"    [mineru] batch={batch_id} state={state} "
            f"pages={progress.get('extracted_pages', '?')}/{progress.get('total_pages', '?')}"
        )
        if state == "done":
            return result["full_zip_url"]
        if state == "failed":
            raise PipelineError(f"MinerU extraction failed: {result.get('err_msg')}")
        time.sleep(5)
    raise PipelineError(f"MinerU polling timed out for batch {batch_id}")


def infer_resume_status(book: BookSpec, book_dir: Path) -> str:
    stage1_chunks = book_dir / "stage1" / "chunks_smart.json"
    if stage1_chunks.exists() and isinstance(load_json(stage1_chunks, []), list) and len(load_json(stage1_chunks, [])) > 0:
        return "completed"
    chunks_raw = book_dir / "chunks_raw.json"
    if chunks_raw.exists() and isinstance(load_json(chunks_raw, []), list) and len(load_json(chunks_raw, [])) > 0:
        return "step4_done"
    if (book_dir / "raw_merged.md").exists():
        return "step3_done"
    if (book_dir / "raw_vision.json").exists():
        return "step2_done"
    if (book_dir / "raw_mineru.md").exists():
        return "step1_done"
    if book.file_type == "epub" and (book_dir / f"{book.slug}.pdf").exists():
        return "step0_done"
    return "pending"


def download_mineru_result(zip_url: str, out_dir: Path, stem: str) -> Path:
    response = requests.get(zip_url, timeout=300)
    response.raise_for_status()
    ensure_dir(out_dir)
    md_path: Optional[Path] = None
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        for name in zf.namelist():
            pure_name = Path(name).name
            if not pure_name:
                continue
            if name.endswith(".md"):
                md_path = out_dir / f"{stem}.md"
                md_path.write_bytes(zf.read(name))
            elif "images/" in name or name.endswith((".png", ".jpg", ".jpeg")):
                img_path = out_dir / "images" / pure_name
                ensure_dir(img_path.parent)
                img_path.write_bytes(zf.read(name))
            elif name.endswith(".json"):
                json_path = out_dir / f"{stem}_content_list.json"
                json_path.write_bytes(zf.read(name))
    if md_path is None:
        raise PipelineError("MinerU result zip did not contain a markdown file")
    return md_path


def run_mineru_for_part(
    token: str,
    part: PdfPart,
    part_dir: Path,
    progress: Dict[str, Any],
    batch_progress_path: Path,
) -> Path:
    guard_daily_quota(progress, batch_progress_path, part.pages)
    upload_url, batch_id = request_mineru_upload_url(token, part.path.name)
    upload_to_presigned_url(upload_url, part.path)
    zip_url = poll_mineru_batch(token, batch_id)
    md_path = download_mineru_result(zip_url, part_dir, part.part_id)
    record_daily_pages(progress, batch_progress_path, part.pages)
    return md_path


def combine_markdown_files(md_paths: Sequence[Path], out_path: Path) -> None:
    pieces: List[str] = []
    for idx, path in enumerate(md_paths, start=1):
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        pieces.append(f"<!-- mineru part {idx}: {path.name} -->\n{content}")
    out_path.write_text("\n\n".join(pieces).strip() + "\n", encoding="utf-8")


def run_step1_mineru(
    book: BookSpec,
    pdf_path: Path,
    book_dir: Path,
    progress: Dict[str, Any],
    batch_progress_path: Path,
) -> Path:
    token = os.environ.get("MINERU_API_KEY", "").strip()
    if not token:
        raise PipelineError("MINERU_API_KEY is required for step1")
    parts_dir = ensure_dir(book_dir / "mineru_parts")
    parts_progress_path = book_dir / "mineru_parts_progress.json"
    raw_mineru_path = book_dir / "raw_mineru.md"
    parts = build_pdf_parts(pdf_path, parts_dir, book.book_id)
    parts_progress = load_json(parts_progress_path, {"parts": []})
    existing_by_id = {item["part_id"]: item for item in parts_progress.get("parts", []) if "part_id" in item}
    entries: List[Dict[str, Any]] = []
    produced_mds: List[Path] = []
    print(f"[step1] {book.book_id}: {len(parts)} MinerU part(s)")
    for part in parts:
        part_dir = ensure_dir(parts_dir / part.part_id)
        md_path = part_dir / f"{part.part_id}.md"
        existing = existing_by_id.get(part.part_id, {})
        entry = {
            "part_id": part.part_id,
            "path": str(part.path),
            "page_start": part.page_start,
            "page_end": part.page_end,
            "pages": part.pages,
            "status": existing.get("status", "pending"),
            "md_path": existing.get("md_path"),
            "updated_at": now_iso(),
        }
        if existing.get("status") == "done" and md_path.exists():
            entry["md_path"] = str(md_path)
            produced_mds.append(md_path)
            entry["updated_at"] = now_iso()
            entries.append(entry)
            continue
        print(f"  [step1] {book.book_id} {part.part_id} pages={part.page_start}-{part.page_end}")
        try:
            result_md = run_mineru_for_part(token, part, part_dir, progress, batch_progress_path)
            entry["status"] = "done"
            entry["md_path"] = str(result_md)
            produced_mds.append(result_md)
        except requests.HTTPError as exc:
            body = exc.response.text[:500] if exc.response is not None else str(exc)
            if "2000" in body or "quota" in body.lower():
                if not prompt_yes_no("[warn] MinerU quota response detected. Continue after confirmation?"):
                    raise PipelineError(f"MinerU quota blocked part {part.part_id}") from exc
            raise PipelineError(f"MinerU request failed for {part.part_id}: {body}") from exc
        entries.append(entry)
        parts_progress = {"parts": entries}
        save_json(parts_progress_path, parts_progress)
    parts_progress = {"parts": entries}
    save_json(parts_progress_path, parts_progress)
    combine_markdown_files(produced_mds, raw_mineru_path)
    return raw_mineru_path


def render_pdf_pages(pdf_path: Path, pages_dir: Path, dpi: int = PNG_DPI) -> List[Path]:
    ensure_dir(pages_dir)
    doc = fitz.open(str(pdf_path))
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    rendered: List[Path] = []
    try:
        for page_index in range(doc.page_count):
            out_path = pages_dir / f"page_{page_index + 1:04d}.png"
            if not out_path.exists():
                pix = doc.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
                pix.save(str(out_path))
            rendered.append(out_path)
    finally:
        doc.close()
    return rendered


def dashscope_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def recognize_page_visual(api_key: str, png_path: Path, page_num: int) -> Dict[str, Any]:
    b64 = base64.b64encode(png_path.read_bytes()).decode("utf-8")
    payload = {
        "model": VISION_MODEL,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 2048,
    }
    response = requests.post(DASHSCOPE_URL, json=payload, headers=dashscope_headers(api_key), timeout=180)
    response.raise_for_status()
    body = response.json()
    message = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    parsed = normalize_vision_parsed(extract_json_value(message))
    return {
        "page_num": page_num,
        "image_path": str(png_path),
        "model": VISION_MODEL,
        "parsed": parsed,
        "raw_response": message,
    }


def run_step2_vision(pdf_path: Path, book_dir: Path) -> Path:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise PipelineError("DASHSCOPE_API_KEY is required for step2")
    pages_dir = book_dir / "pages_150dpi"
    vision_path = book_dir / "raw_vision.json"
    page_paths = render_pdf_pages(pdf_path, pages_dir, dpi=PNG_DPI)
    target_pages = build_visual_target_pages(book_dir)
    page_meta_map = build_page_metadata_map(book_dir)
    if not target_pages:
        save_json(
            vision_path,
            {
                "expected_pages": len(page_paths),
                "target_pages": [],
                "updated_at": now_iso(),
                "pages": [],
            },
        )
        return vision_path
    existing = load_json(vision_path, {"pages": []})
    page_results = {int(item["page_num"]): item for item in existing.get("pages", []) if "page_num" in item}

    def persist() -> None:
        save_json(
            vision_path,
            {
                "expected_pages": len(page_paths),
                "target_pages": target_pages,
                "updated_at": now_iso(),
                "pages": [page_results[idx] for idx in sorted(page_results)],
            },
        )

    print(f"[step2] rendering/vision for {len(target_pages)} target page(s) out of {len(page_paths)} total")
    for page_num in target_pages:
        png_path = page_paths[page_num - 1]
        existing_result = page_results.get(page_num)
        page_meta = page_meta_map.get(page_num, {"pdf_page_num": page_num, "book_page_label": None, "book_page_candidates": []})
        if existing_result and existing_result.get("parsed") is not None:
            existing_result.update(page_meta)
            page_results[page_num] = existing_result
            continue
        print(f"  [step2] page {page_num}/{len(page_paths)}")
        try:
            result = recognize_page_visual(api_key, png_path, page_num)
        except Exception as exc:
            result = {
                "page_num": page_num,
                "image_path": str(png_path),
                "error": str(exc),
            }
        result.update(page_meta)
        page_results[page_num] = result
        persist()
        time.sleep(0.3)
    persist()
    return vision_path


def extract_markdown_tables(text: str) -> List[str]:
    matches = re.findall(r"(?:^\|.*\n?)+", text, flags=re.MULTILINE)
    return [match.strip() for match in matches if len(match.strip().splitlines()) >= 2]


def visual_blocks_from_entry(entry: Dict[str, Any]) -> List[str]:
    if entry.get("error"):
        return []
    parsed = entry.get("parsed") or {}
    page_num = entry.get("page_num")
    book_page_label = entry.get("book_page_label")
    blocks: List[str] = []
    page_suffix = f" pdf-page {page_num}"
    if book_page_label:
        page_suffix += f", book-page {book_page_label}"
    for table in parsed.get("tables") or []:
        markdown = str((table or {}).get("markdown") or "").strip()
        title = str((table or {}).get("title") or "").strip()
        notes = str((table or {}).get("notes") or "").strip()
        if markdown:
            header = f"<!-- qwen3-vl-plus{page_suffix} table"
            if title:
                header += f": {title}"
            header += " -->"
            block = f"{header}\n{markdown}"
            if notes:
                block += f"\n\n> Note: {notes}"
            blocks.append(block)
    for figure in parsed.get("figures") or []:
        description = str((figure or {}).get("description") or "").strip()
        figure_type = str((figure or {}).get("type") or "figure").strip()
        if description:
            blocks.append(f"> [{figure_type}{page_suffix}] {description}")
    page_note = str(parsed.get("page_note") or "").strip()
    if page_note:
        blocks.append(f"> [{page_suffix.strip()}] {page_note}")
    return blocks


def collect_image_paths(node: Any, out: List[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"image_path", "img_path"} and isinstance(value, str):
                out.append(Path(value).name)
            else:
                collect_image_paths(value, out)
    elif isinstance(node, list):
        for item in node:
            collect_image_paths(item, out)


def collect_text_spans(node: Any, out: List[str]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "text" and isinstance(node.get("content"), str):
            text = node["content"].strip()
            if text:
                out.append(text)
        for value in node.values():
            collect_text_spans(value, out)
    elif isinstance(node, list):
        for item in node:
            collect_text_spans(item, out)


def looks_like_page_label(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned or len(cleaned) > 24:
        return False
    if re.fullmatch(r"[ivxlcdmIVXLCDM]+", cleaned):
        return True
    if re.fullmatch(r"\d{1,4}", cleaned):
        return True
    if re.fullmatch(r"[A-Za-z]|\d+\s*[A-Za-z]?", cleaned):
        return True
    return False


def extract_page_label_candidates(page: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in ["discarded_blocks", "preproc_blocks"]:
        for block in page.get(key) or []:
            texts: List[str] = []
            collect_text_spans(block, texts)
            joined = " ".join(texts).strip()
            if not joined:
                continue
            if looks_like_page_label(joined):
                candidates.append(joined)
    seen: set[str] = set()
    deduped: List[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def build_image_page_map(book_dir: Path) -> Dict[str, int]:
    parts_progress_path = book_dir / "mineru_parts_progress.json"
    parts_progress = load_json(parts_progress_path, {"parts": []})
    image_page_map: Dict[str, int] = {}
    for part in parts_progress.get("parts", []):
        part_id = str(part.get("part_id") or "")
        page_start = int(part.get("page_start") or 1)
        md_path = Path(str(part.get("md_path") or ""))
        if not part_id or not md_path.exists():
            continue
        content_list_path = md_path.parent / f"{part_id}_content_list.json"
        if not content_list_path.exists():
            continue
        content_list = load_json(content_list_path, {})
        if isinstance(content_list, dict):
            pdf_info = content_list.get("pdf_info") or []
        elif isinstance(content_list, list):
            pdf_info = content_list
        else:
            pdf_info = []
        for page in pdf_info:
            if not isinstance(page, dict):
                continue
            page_idx = int(page.get("page_idx") or 0)
            actual_page = page_start + page_idx
            image_paths: List[str] = []
            collect_image_paths(page, image_paths)
            for image_name in image_paths:
                image_page_map.setdefault(image_name, actual_page)
    return image_page_map


def build_page_image_sequence(book_dir: Path) -> Dict[int, List[str]]:
    parts_progress_path = book_dir / "mineru_parts_progress.json"
    parts_progress = load_json(parts_progress_path, {"parts": []})
    page_images: Dict[int, List[str]] = {}
    for part in parts_progress.get("parts", []):
        part_id = str(part.get("part_id") or "")
        page_start = int(part.get("page_start") or 1)
        md_path = Path(str(part.get("md_path") or ""))
        if not part_id or not md_path.exists():
            continue
        content_list_path = md_path.parent / f"{part_id}_content_list.json"
        if not content_list_path.exists():
            continue
        content_list = load_json(content_list_path, {})
        if isinstance(content_list, dict):
            pdf_info = content_list.get("pdf_info") or []
        elif isinstance(content_list, list):
            pdf_info = content_list
        else:
            pdf_info = []
        for page in pdf_info:
            if not isinstance(page, dict):
                continue
            page_idx = int(page.get("page_idx") or 0)
            actual_page = page_start + page_idx
            image_paths: List[str] = []
            collect_image_paths(page, image_paths)
            if image_paths:
                page_images[actual_page] = [Path(name).name for name in image_paths]
    return page_images


def build_visual_target_pages(book_dir: Path) -> List[int]:
    return sorted(set(build_image_page_map(book_dir).values()))


def build_page_metadata_map(book_dir: Path) -> Dict[int, Dict[str, Any]]:
    parts_progress_path = book_dir / "mineru_parts_progress.json"
    parts_progress = load_json(parts_progress_path, {"parts": []})
    meta_map: Dict[int, Dict[str, Any]] = {}
    for part in parts_progress.get("parts", []):
        part_id = str(part.get("part_id") or "")
        page_start = int(part.get("page_start") or 1)
        md_path = Path(str(part.get("md_path") or ""))
        if not part_id or not md_path.exists():
            continue
        content_list_path = md_path.parent / f"{part_id}_content_list.json"
        if not content_list_path.exists():
            continue
        content_list = load_json(content_list_path, {})
        if isinstance(content_list, dict):
            pdf_info = content_list.get("pdf_info") or []
        elif isinstance(content_list, list):
            pdf_info = content_list
        else:
            pdf_info = []
        for page in pdf_info:
            if not isinstance(page, dict):
                continue
            page_idx = int(page.get("page_idx") or 0)
            actual_page = page_start + page_idx
            candidates = extract_page_label_candidates(page)
            meta_map[actual_page] = {
                "pdf_page_num": actual_page,
                "part_id": part_id,
                "part_page_idx": page_idx,
                "book_page_candidates": candidates,
                "book_page_label": candidates[0] if candidates else None,
            }
    return meta_map


def load_merge_helper():
    try:
        from merge_mineru_qwen import merge_mineru_qwen

        return merge_mineru_qwen
    except ModuleNotFoundError:
        script_path = Path(__file__).with_name("merge_mineru_qwen.py")
        spec = importlib.util.spec_from_file_location("merge_mineru_qwen", script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.merge_mineru_qwen


def run_step3_merge(book_dir: Path) -> Path:
    raw_mineru_path = book_dir / "raw_mineru.md"
    raw_vision_path = book_dir / "raw_vision.json"
    merged_path = book_dir / "raw_merged.md"
    parts_progress_path = book_dir / "mineru_parts_progress.json"
    parts_progress = load_json(parts_progress_path, {"parts": []})
    content_list_paths: List[str] = []
    for part in parts_progress.get("parts", []):
        part_id = str(part.get("part_id") or "")
        md_path = Path(str(part.get("md_path") or ""))
        if not part_id or not md_path.exists():
            continue
        content_list_path = md_path.parent / f"{part_id}_content_list.json"
        if content_list_path.exists():
            content_list_paths.append(str(content_list_path))
    if not content_list_paths:
        content_list_paths = [str(book_dir / "mineru_parts")]
    merge_mineru_qwen = load_merge_helper()
    merge_mineru_qwen(
        mineru_path=raw_mineru_path,
        vision_path=raw_vision_path,
        content_list_paths=content_list_paths,
        output_path=merged_path,
        report_path=book_dir / "raw_merge_report.json",
    )
    return merged_path


@dataclass
class Chapter:
    chapter_num: int
    chapter_title: str
    text: str


def split_markdown_into_chapters(markdown_text: str) -> List[Chapter]:
    lines = markdown_text.splitlines()
    chapters: List[Chapter] = []
    current_title = "Introduction"
    current_lines: List[str] = []
    current_num = 0
    for line in lines:
        if re.match(r"^#{1,2}\s+", line):
            if current_lines:
                chapters.append(
                    Chapter(
                        chapter_num=max(current_num, len(chapters) + 1),
                        chapter_title=current_title,
                        text="\n".join(current_lines).strip(),
                    )
                )
            current_title = re.sub(r"^#{1,2}\s+", "", line).strip()
            digits = re.findall(r"\d+", current_title)
            current_num = int(digits[0]) if digits else len(chapters) + 1
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        chapters.append(
            Chapter(
                chapter_num=max(current_num, len(chapters) + 1),
                chapter_title=current_title,
                text="\n".join(current_lines).strip(),
            )
        )
    if not chapters:
        chapters.append(Chapter(chapter_num=1, chapter_title="Full Text", text=markdown_text.strip()))
    return [chapter for chapter in chapters if chapter.text]


def ollama_generate(model: str, prompt: str, timeout: int = 240) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"think": False},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    return str(body.get("response") or "").strip()


def split_chapter_with_model(chapter: Chapter) -> List[str]:
    prompt = SPLIT_PROMPT_TEMPLATE.format(
        chapter_title=chapter.chapter_title,
        chapter_text=chapter.text,
    )
    raw = ollama_generate(SPLIT_MODEL, prompt, timeout=600)
    parsed = extract_json_object(raw)
    chunks = parsed.get("chunks")
    if isinstance(chunks, list):
        return [str(item).strip() for item in chunks if str(item).strip()]
    raise PipelineError(f"Unexpected split response for chapter {chapter.chapter_title}")


def run_step4_chunk(book: BookSpec, book_dir: Path) -> Tuple[Path, int]:
    merged_path = book_dir / "raw_merged.md"
    chunks_path = book_dir / "chunks_raw.json"
    failed_path = book_dir / "failed_chapters.json"
    chapters = split_markdown_into_chapters(merged_path.read_text(encoding="utf-8", errors="ignore"))
    existing_chunks = load_json_list(chunks_path)
    failed_chapters: List[Dict[str, Any]] = []
    done_keys = {(int(item["chapter_num"]), str(item["chapter_title"])) for item in existing_chunks if "chapter_num" in item}
    next_chunk_idx = (max((int(item["chunk_idx"]) for item in existing_chunks), default=-1) + 1)
    print(f"[step4] chapter chunking for {len(chapters)} chapter(s)")
    for chapter in chapters:
        chapter_key = (chapter.chapter_num, chapter.chapter_title)
        if chapter_key in done_keys:
            continue
        print(f"  [step4] chapter {chapter.chapter_num}: {chapter.chapter_title}")
        try:
            chapter_chunks = split_chapter_with_model(chapter)
            for chunk_text in chapter_chunks:
                existing_chunks.append(
                    {
                        "chunk_idx": next_chunk_idx,
                        "full_text": chunk_text,
                        "chapter_num": chapter.chapter_num,
                        "chapter_title": chapter.chapter_title,
                        "source_book": book.book_id,
                    }
                )
                next_chunk_idx += 1
            save_json(chunks_path, existing_chunks)
            done_keys.add(chapter_key)
        except Exception as exc:
            failed_chapters.append(
                {
                    "chapter_num": chapter.chapter_num,
                    "chapter_title": chapter.chapter_title,
                    "error": str(exc),
                    "updated_at": now_iso(),
                }
            )
            save_json(failed_path, failed_chapters)
    save_json(failed_path, failed_chapters)
    if not chunks_path.exists():
        save_json(chunks_path, existing_chunks)
    return chunks_path, len(existing_chunks)


def annotate_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    prompt = ANNOTATE_PROMPT_TEMPLATE.format(
        topics=", ".join(VALID_TOPICS),
        chunk_text=chunk["full_text"],
    )
    raw = ollama_generate(ANNOTATE_MODEL, prompt, timeout=240)
    parsed = extract_json_object(raw)
    summary = str(parsed.get("summary") or "").strip()
    topics = [str(item).strip() for item in parsed.get("topics") or [] if str(item).strip() in VALID_TOPICS]
    if not summary:
        raise PipelineError("Missing summary in annotation response")
    if not topics:
        raise PipelineError("Missing valid topics in annotation response")
    return {
        "summary": summary[:50],
        "topics": topics[:3],
    }


def run_step5_annotate(book_dir: Path) -> Tuple[Path, int]:
    chunks_path = book_dir / "chunks_raw.json"
    stage1_dir = ensure_dir(book_dir / "stage1")
    out_path = stage1_dir / "chunks_smart.json"
    failures_path = stage1_dir / "annotation_failures.json"
    chunks = load_json_list(chunks_path)
    annotated = load_json_list(out_path)
    failures = load_json_list(failures_path)
    annotated_ids = {int(item["chunk_idx"]) for item in annotated if "chunk_idx" in item}
    failure_ids = {int(item["chunk_idx"]) for item in failures if "chunk_idx" in item}
    processed_since_save = 0
    print(f"[step5] annotating {len(chunks)} chunk(s)")
    for chunk in chunks:
        chunk_idx = int(chunk["chunk_idx"])
        if chunk_idx in annotated_ids or chunk_idx in failure_ids:
            continue
        last_error: Optional[str] = None
        for _ in range(3):
            try:
                annotation = annotate_chunk(chunk)
                merged = dict(chunk)
                merged.update(annotation)
                annotated.append(merged)
                annotated_ids.add(chunk_idx)
                processed_since_save += 1
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1)
        if last_error is not None:
            failures.append(
                {
                    "chunk_idx": chunk_idx,
                    "chapter_num": chunk.get("chapter_num"),
                    "chapter_title": chunk.get("chapter_title"),
                    "error": last_error,
                    "updated_at": now_iso(),
                }
            )
            failure_ids.add(chunk_idx)
        if processed_since_save >= 50:
            save_json(out_path, annotated)
            save_json(failures_path, failures)
            processed_since_save = 0
    save_json(out_path, annotated)
    save_json(failures_path, failures)
    return out_path, len(annotated)


def process_book(
    book: BookSpec,
    output_root: Path,
    batch_progress: Dict[str, Any],
    batch_progress_path: Path,
) -> None:
    if not book.path.exists():
        raise PipelineError(f"Input file not found: {book.path}")
    book_dir = ensure_dir(output_root / book.slug)
    entry = ensure_book_progress(batch_progress, book.book_id)
    current_status = entry["status"]
    inferred_status = infer_resume_status(book, book_dir)
    if current_status == "failed":
        current_status = inferred_status
        entry["status"] = current_status
        entry["updated_at"] = now_iso()
        save_batch_progress(batch_progress_path, batch_progress)
    elif status_rank(inferred_status) > status_rank(current_status):
        current_status = inferred_status
        entry["status"] = current_status
        entry["updated_at"] = now_iso()
        save_batch_progress(batch_progress_path, batch_progress)
    work_pdf = book.path if book.file_type == "pdf" else book_dir / f"{book.slug}.pdf"
    print(f"\n=== {book.book_id} ({current_status}) ===")

    if status_rank(current_status) < status_rank("step0_done"):
        work_pdf = resolve_work_pdf(book, book_dir)
        set_book_status(batch_progress, batch_progress_path, book.book_id, status="step0_done")
    else:
        if book.file_type == "epub" and not work_pdf.exists():
            work_pdf = resolve_work_pdf(book, book_dir)
        else:
            work_pdf = book.path if book.file_type == "pdf" else work_pdf

    if status_rank(entry["status"]) < status_rank("step1_done"):
        run_step1_mineru(book, work_pdf, book_dir, batch_progress, batch_progress_path)
        set_book_status(batch_progress, batch_progress_path, book.book_id, status="step1_done")

    if status_rank(entry["status"]) < status_rank("step2_done"):
        run_step2_vision(work_pdf, book_dir)
        set_book_status(batch_progress, batch_progress_path, book.book_id, status="step2_done")

    if status_rank(entry["status"]) < status_rank("step3_done"):
        run_step3_merge(book_dir)
        set_book_status(batch_progress, batch_progress_path, book.book_id, status="step3_done")

    if status_rank(entry["status"]) < status_rank("step4_done"):
        _, total_chunks = run_step4_chunk(book, book_dir)
        if total_chunks <= 0:
            raise PipelineError("step4 produced 0 chunks")
        set_book_status(
            batch_progress,
            batch_progress_path,
            book.book_id,
            status="step4_done",
            total_chunks=total_chunks,
        )

    if status_rank(entry["status"]) < status_rank("completed"):
        _, total_annotated = run_step5_annotate(book_dir)
        if total_annotated <= 0:
            raise PipelineError("step5 produced 0 annotated chunks")
        set_book_status(
            batch_progress,
            batch_progress_path,
            book.book_id,
            status="completed",
            total_chunks=total_annotated,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch MC Stage 1 pipeline")
    parser.add_argument(
        "--output-root",
        default="/Users/jeff/l0-knowledge-engine/output/mc",
        help="Output directory root",
    )
    parser.add_argument(
        "--books-json",
        help="Optional JSON file containing the ordered book list",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = ensure_dir(Path(args.output_root).expanduser())
    batch_progress_path = output_root / "batch_progress.json"
    books = load_books(args)
    batch_progress = load_batch_progress(batch_progress_path)

    for book in books:
        ensure_book_progress(batch_progress, book.book_id)
    save_batch_progress(batch_progress_path, batch_progress)

    for book in books:
        try:
            process_book(book, output_root, batch_progress, batch_progress_path)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            set_book_status(
                batch_progress,
                batch_progress_path,
                book.book_id,
                status="failed",
                error=str(exc),
            )
            print(f"[error] {book.book_id}: {exc}", file=sys.stderr)
            return 1

    print("\nAll requested MC books finished Stage 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
