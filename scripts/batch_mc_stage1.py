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

VISION_PROMPT = """You are extracting non-body visual content from a book page.
Return JSON only with this schema:
{
  "tables": [{"title": "", "markdown": "", "notes": ""}],
  "figures": [{"type": "table|diagram|photo|chart|other", "description": ""}],
  "page_note": ""
}

Rules:
- Focus on tables, charts, diagrams, labeled figures, and image-only explanatory content.
- Do not transcribe ordinary running body paragraphs unless needed to explain a visual.
- Preserve units and numbers.
- If there is no useful visual content, return empty arrays and an empty page_note.
"""

SPLIT_PROMPT_TEMPLATE = """You are splitting a book chapter into semantic chunks.
Return JSON only with this schema:
{"chunks": ["chunk text 1", "chunk text 2"]}

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
{
  "summary": "Chinese summary within 50 characters",
  "topics": ["one_or_more_allowed_topics"]
}

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
    url = f"{MINERU_BASE_URL}/extract/batch/result/{batch_id}"
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
    parsed = extract_json_object(message)
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
    existing = load_json(vision_path, {"pages": []})
    page_results = {int(item["page_num"]): item for item in existing.get("pages", []) if "page_num" in item}
    pages_out: List[Dict[str, Any]] = []
    print(f"[step2] rendering/vision for {len(page_paths)} page(s)")
    for page_num, png_path in enumerate(page_paths, start=1):
        if page_num in page_results and page_results[page_num].get("parsed") is not None:
            pages_out.append(page_results[page_num])
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
        pages_out.append(result)
        save_json(vision_path, {"pages": sorted(pages_out, key=lambda item: item["page_num"])})
        time.sleep(0.3)
    save_json(vision_path, {"pages": sorted(pages_out, key=lambda item: item["page_num"])})
    return vision_path


def extract_markdown_tables(text: str) -> List[str]:
    matches = re.findall(r"(?:^\|.*\n?)+", text, flags=re.MULTILINE)
    return [match.strip() for match in matches if len(match.strip().splitlines()) >= 2]


def visual_blocks_from_entry(entry: Dict[str, Any]) -> List[str]:
    if entry.get("error"):
        return []
    parsed = entry.get("parsed") or {}
    page_num = entry.get("page_num")
    blocks: List[str] = []
    for table in parsed.get("tables") or []:
        markdown = str((table or {}).get("markdown") or "").strip()
        title = str((table or {}).get("title") or "").strip()
        notes = str((table or {}).get("notes") or "").strip()
        if markdown:
            header = f"<!-- qwen3-vl-plus page {page_num} table"
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
            blocks.append(f"> [{figure_type} page {page_num}] {description}")
    page_note = str(parsed.get("page_note") or "").strip()
    if page_note:
        blocks.append(f"> [page {page_num}] {page_note}")
    return blocks


def run_step3_merge(book_dir: Path) -> Path:
    raw_mineru_path = book_dir / "raw_mineru.md"
    raw_vision_path = book_dir / "raw_vision.json"
    merged_path = book_dir / "raw_merged.md"
    mineru_text = raw_mineru_path.read_text(encoding="utf-8", errors="ignore")
    vision_data = load_json(raw_vision_path, {"pages": []})
    replacement_queue: List[str] = []
    for entry in sorted(vision_data.get("pages", []), key=lambda item: item.get("page_num", 0)):
        replacement_queue.extend(visual_blocks_from_entry(entry))
    placeholders = list(re.finditer(r"!\[.*?\]\([^)]+\)", mineru_text))
    replaced_parts: List[str] = []
    cursor = 0
    queue_index = 0
    for match in placeholders:
        replaced_parts.append(mineru_text[cursor : match.start()])
        if queue_index < len(replacement_queue):
            replaced_parts.append("\n" + replacement_queue[queue_index] + "\n")
            queue_index += 1
        else:
            replaced_parts.append(match.group(0))
        cursor = match.end()
    replaced_parts.append(mineru_text[cursor:])
    merged = "".join(replaced_parts)
    extras = replacement_queue[queue_index:]
    if extras:
        merged += "\n\n---\n\n## Additional qwen3-vl-plus extracted visuals\n\n"
        merged += "\n\n".join(extras)
    merged = re.sub(r"\n{4,}", "\n\n\n", merged).strip() + "\n"
    merged_path.write_text(merged, encoding="utf-8")
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
    existing_chunks = load_json(chunks_path, [])
    failed_chapters = load_json(failed_path, [])
    done_keys = {(int(item["chapter_num"]), str(item["chapter_title"])) for item in existing_chunks if "chapter_num" in item}
    failed_keys = {(int(item["chapter_num"]), str(item["chapter_title"])) for item in failed_chapters if "chapter_num" in item}
    next_chunk_idx = (max((int(item["chunk_idx"]) for item in existing_chunks), default=-1) + 1)
    print(f"[step4] chapter chunking for {len(chapters)} chapter(s)")
    for chapter in chapters:
        chapter_key = (chapter.chapter_num, chapter.chapter_title)
        if chapter_key in done_keys or chapter_key in failed_keys:
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
    chunks = load_json(chunks_path, [])
    annotated = load_json(out_path, [])
    failures = load_json(failures_path, [])
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
        set_book_status(
            batch_progress,
            batch_progress_path,
            book.book_id,
            status="step4_done",
            total_chunks=total_chunks,
        )

    if status_rank(entry["status"]) < status_rank("completed"):
        _, total_annotated = run_step5_annotate(book_dir)
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
