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
import ast
import base64
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz
import requests

try:
    from chonkie import SemanticChunker
except ModuleNotFoundError:
    SemanticChunker = None


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

MC_SECTION_TOC: Dict[str, List[Dict[str, Any]]] = {
    "mc_vol2": [
        {
            "chapter_num": 7,
            "chapter_title": "TRADITIONAL COOKING",
            "sections": [
                "GRILLING",
                "BROILING",
                "ROASTING",
                "PANFRYING A LA PLANCHA",
                "SAUTEING",
                "STIR-FRYING",
                "COVERED SAUTEING",
                "BOILING",
                "STEAMING",
                "CANNING",
                "POT-ROASTING AND STEWING",
                "BAKING",
                "COOKING IN OIL",
                "SMOKING",
            ],
        },
        {
            "chapter_num": 8,
            "chapter_title": "COOKING IN MODERN OVENS",
            "sections": [
                "COOKING WITH MOIST AIR",
                "COOKING WITH MICROWAVES",
            ],
        },
        {
            "chapter_num": 9,
            "chapter_title": "COOKING SOUS VIDE",
            "sections": [
                "WHY SOUS VIDE",
                "PACKAGING FOOD FOR SOUS VIDE",
                "SOUS VIDE EQUIPMENT",
                "STRATEGIES FOR COOKING SOUS VIDE",
                "STRATEGIES FOR CHILLING AND REHEATING",
                "BLANCHING AND SEARING FOR SOUS VIDE",
            ],
        },
        {
            "chapter_num": 10,
            "chapter_title": "THE MODERNIST KITCHEN",
            "sections": [
                "EXTRACTING FLAVORS",
                "INFUSING ESSENCES",
                "JUICING",
                "FILTERING",
                "CONCENTRATE",
                "CUTTING EM DOWN TO SIZE",
                "DRYING",
                "CRYOGENIC FREEZING AND CARBONATING",
            ],
        },
    ],
    "mc_vol3": [
        {
            "chapter_num": 11,
            "chapter_title": "MEAT AND SEAFOOD",
            "sections": [
                "HOW MUSCLE WORKS",
                "CONVERTING MUSCLE INTO MEAT",
                "CUTTING",
                "COOKING MEAT AND SEAFOOD",
                "COOKING SKIN AND INNARDS",
                "SALTING AND DRYING",
                "MARINATING",
                "SMOKING",
                "RESTRUCTURING",
            ],
        },
        {
            "chapter_num": 12,
            "chapter_title": "PLANT FOODS",
            "sections": [
                "PLANTS AS FOOD",
                "COOKING SOUS VIDE",
                "PRESSURE-COOKING",
                "MICROWAVING",
                "FRYING",
                "PRESERVING",
                "MODIFYING TEXTURES",
            ],
        },
    ],
    "mc_vol4": [
        {
            "chapter_num": 13,
            "chapter_title": "THICKENERS",
            "sections": [
                "HOW THICKENING WORKS",
                "STRATEGIES FOR THICKENING",
                "STARCHES",
                "HYDROCOLLOIDS",
            ],
        },
        {
            "chapter_num": 14,
            "chapter_title": "GELS",
            "sections": [
                "HOW GELLING WORKS",
                "EGG GELS",
                "DAIRY AND TOFU GELS",
                "GELLING WITH HYDROCOLLOIDS",
                "FLUID GELS",
                "SPHERIFICATION",
            ],
        },
        {
            "chapter_num": 15,
            "chapter_title": "EMULSIONS",
            "sections": [
                "HOW EMULSIFICATION WORKS",
                "METHODS OF EMULSIFYING",
                "MODERNIST EMULSIONS",
            ],
        },
        {
            "chapter_num": 16,
            "chapter_title": "FOAMS",
            "sections": [
                "HOW FOAMS WORK",
                "FORMING FOAMS",
            ],
        },
        {
            "chapter_num": 17,
            "chapter_title": "WINE",
            "sections": [
                "WHAT MAKES A GREAT WINE",
                "TASTING WINE",
            ],
        },
        {
            "chapter_num": 18,
            "chapter_title": "COFFEE",
            "sections": [
                "FROM CHERRY TO BEAN",
                "BREWING",
                "ESPRESSO",
                "THE ART OF MILK AND COFFEE",
                "ACHIEVING CONSISTENCY",
            ],
        },
    ],
}

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
CHONKIE_EMBEDDING_MODEL = "minishlab/potion-base-8M"
CHONKIE_CHUNK_SIZE = 350
CHONKIE_MIN_CHUNK_SIZE = 150
CHONKIE_THRESHOLD = 0.3

MC_BODY_START_HEADINGS: Dict[str, str] = {
    "mc_vol2": "TRADITIONAL COOKING",
    "mc_vol3": "MEAT AND SEAFOOD",
    "mc_vol4": "THICKENERS",
}

MC_PRIMARY_CHAPTERS: Dict[str, List[Tuple[int, str]]] = {
    "mc_vol2": [
        (7, "TRADITIONAL COOKING"),
        (8, "COOKING IN MODERN OVENS"),
        (9, "COOKING SOUS VIDE"),
        (10, "THE MODERNIST KITCHEN"),
    ],
    "mc_vol3": [
        (11, "MEAT AND SEAFOOD"),
        (12, "PLANT FOODS"),
    ],
    "mc_vol4": [
        (13, "THICKENERS"),
        (14, "GELS"),
        (15, "EMULSIONS"),
        (16, "FOAMS"),
        (17, "WINE"),
        (18, "COFFEE"),
    ],
}

BOOKS_USE_QWEN_SPLIT = set(MC_SECTION_TOC.keys())

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
- 如果一张图片里包含多个独立子图、步骤图、拼图面板，请按子图分别输出多个元素。
- 如果页面同时有图和对应图注，优先按视觉单元拆分，不要整页合并成一条。
- 有几个就输出几个，不要合并。
- 不要抄写普通正文段落，只保留视觉相关内容。
- 只输出JSON数组，不要其他说明。
"""

SPLIT_PROMPT_TEMPLATE = """你要把一本书的单一章节片段切成多个原文 chunk。
只输出 JSON，格式必须是：
{{"chunks": ["chunk text 1", "chunk text 2"]}}

规则：
- 只能摘录输入原文，不能翻译，不能改写，不能总结，不能解释。
- 每个 chunk 必须是来自输入文本的连续原文片段。
- 按语义边界切分。
- 每个 chunk 目标长度约 300-400 字；如果是英文，约 180-260 words。
- 不要跨越章节边界。
- 表格、列表、图注尽量和附近正文放在同一 chunk。
- 不要输出空字符串。
- 不要输出除了 JSON 之外的任何文字。

Chapter boundary:
- Start heading: {chapter_start}
- End before: {chapter_end}

Chapter segment: {chapter_segment}

Chapter title: {chapter_title}

Chapter text:
{chapter_text}
"""

ANNOTATE_PROMPT_TEMPLATE = """你要为一个食品科学 chunk 生成检索用摘要和高精度 topics。
只输出 JSON，格式必须是：
{{
  "summary": "50字以内中文摘要",
  "topics": ["allowed_topic_1", "allowed_topic_2"]
}}

Allowed topics:
{topics}

Topic definitions:
- heat_transfer: 传导、对流、辐射、升温降温、热流、火力、蒸汽传热、加热速率
- chemical_reaction: 美拉德、焦糖化、氧化、酸碱反应、褐变、聚合、分解等明确化学反应
- physical_change: 沸腾、蒸发、冷凝、冻结、融化、结晶、相变、压力导致的物理变化
- water_activity: 水分活度、干燥、吸湿、湿度、脱水、保水、边界层含水影响
- protein_science: 蛋白质变性、凝胶、肌肉蛋白、胶原、蛋白结构变化
- lipid_science: 油脂、脂肪氧化、脂肪结晶、脂肪熔化、起酥、油脂行为
- carbohydrate: 淀粉、糖、纤维、糊化、老化、糖的结构与性质
- enzyme: 酶促褐变、酶失活、酶催化
- flavor_sensory: 香气、风味、口感、感官体验，但前提是 chunk 的核心确实在讲感官结果
- fermentation: 发酵、微生物代谢产物、酒精/乳酸/醋酸发酵
- food_safety: 病原体、毒素、杀菌、巴氏杀菌、灭菌、污染、保藏安全、罐藏安全
- emulsion_colloid: 乳化、泡沫、悬浮、胶体稳定性
- color_pigment: 色素、显色、褪色、花青素、叶绿素、肌红蛋白颜色变化
- equipment_physics: 设备结构、风道、压力阀、真空、微波、烤箱/烤架/锅具/泵的工作原理

Strict rules:
- Summary must be concise Chinese and <= 50 characters; 20-40 characters is preferred.
- Summary is for Stage2 cosine matching, so capture only the core scientific idea.
- No examples, no rhetorical phrasing, no fluffy wording.
- Topics must be selected only from the allowed list.
- Choose 1-2 topics only. Do not choose 3 unless absolutely necessary.
- Prefer the main mechanism, not peripheral mentions.
- Do not choose food_safety unless the chunk is substantively about contamination, pathogens, pasteurization, sterilization, spoilage prevention, or safety limits.
- Do not choose chemical_reaction unless the chunk explicitly discusses a reaction mechanism or reaction outcome.
- Do not choose flavor_sensory if the chunk is mainly about heat, moisture, equipment, pressure, or process control.
- If a chunk is mainly about equipment working principles, prefer equipment_physics and/or heat_transfer.
- If a chunk is mainly about drying, humidity, or moisture migration, prefer water_activity and/or physical_change.
- If uncertain, return the single most central topic.

Chapter title: {chapter_title}
Section range: {chapter_start} -> {chapter_end}

Chunk:
{chunk_text}
"""


class PipelineError(RuntimeError):
    """Raised for expected pipeline failures."""


_SEMANTIC_CHUNKER: Optional["SemanticChunker"] = None


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
    chunks_raw = book_dir / "chunks_raw.json"
    chunks_raw_data = load_json(chunks_raw, []) if chunks_raw.exists() else []
    step4_quality = load_json(book_dir / "step4_quality.json", {}) if (book_dir / "step4_quality.json").exists() else {}
    stage1_chunks = book_dir / "stage1" / "chunks_smart.json"
    stage1_failures = book_dir / "stage1" / "annotation_failures.json"
    stage1_chunks_data = load_json(stage1_chunks, []) if stage1_chunks.exists() else []
    stage1_failures_data = load_json(stage1_failures, []) if stage1_failures.exists() else []

    if (
        isinstance(stage1_chunks_data, list)
        and isinstance(chunks_raw_data, list)
        and len(stage1_chunks_data) > 0
        and len(stage1_chunks_data) == len(chunks_raw_data)
        and isinstance(stage1_failures_data, list)
        and len(stage1_failures_data) == 0
    ):
        return "completed"
    if (
        isinstance(chunks_raw_data, list)
        and len(chunks_raw_data) > 0
        and (
            not step4_quality
            or (
                int(step4_quality.get("total_chunks") or 0) == len(chunks_raw_data)
                and int(step4_quality.get("lt50_chars") or 0) == 0
            )
        )
    ):
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


def run_step2_vision(pdf_path: Path, book_dir: Path, force: bool = False) -> Path:
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
        if (not force) and existing_result and existing_result.get("parsed") is not None:
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
    chapter_start: str
    chapter_end: str
    text: str


def normalize_heading_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    cleaned = re.sub(r"^:+", "", normalized.strip())
    cleaned = cleaned.replace("’", "'")
    cleaned = cleaned.replace("'", "")
    cleaned = cleaned.replace('"', "")
    cleaned = cleaned.replace(".", " ")
    cleaned = cleaned.replace("!", " ")
    cleaned = cleaned.replace("?", " ")
    cleaned = cleaned.replace(",", " ")
    cleaned = cleaned.replace(":", " ")
    cleaned = cleaned.replace(";", " ")
    cleaned = cleaned.replace("(", " ")
    cleaned = cleaned.replace(")", " ")
    cleaned = cleaned.replace("/", " ")
    cleaned = cleaned.replace("-", " ")
    cleaned = cleaned.replace("A LA", "ALA")
    cleaned = cleaned.replace("EM DOWN", "EM DOWN")
    cleaned = re.sub(r"\s+\d+$", "", cleaned)
    cleaned = re.sub(r"\bSO US\b", "SOUS", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bIA\b", "LA", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bSAUTEING\b", "SAUTEING", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
    return cleaned.replace(" ", "")


def extract_toc_chapter_map(lines: List[str]) -> Dict[str, Tuple[int, str]]:
    chapter_map: Dict[str, Tuple[int, str]] = {}
    pattern = re.compile(r"^#\s*:?\s*CHAPTER\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        chapter_num = int(match.group(1))
        chapter_title = re.sub(r"\s+\d+$", "", match.group(2).strip()).strip()
        key = normalize_heading_key(chapter_title)
        if key and key not in chapter_map:
            chapter_map[key] = (chapter_num, chapter_title)
    return chapter_map


def find_heading_positions(lines: List[str]) -> List[Tuple[int, str, str]]:
    positions: List[Tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        title = re.sub(r"^#+\s+", "", line).strip()
        if not title:
            continue
        positions.append((idx, title, normalize_heading_key(title)))
    return positions


def looks_like_toc_noise(text: str, section_titles: Optional[Sequence[str]] = None) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    normalized_text = normalize_heading_key(stripped)
    if section_titles:
        section_hits = 0
        for section_title in section_titles:
            key = normalize_heading_key(str(section_title))
            if key and key in normalized_text:
                section_hits += 1
        if section_hits >= 3:
            return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    prose_lines = [line for line in lines if len(re.findall(r"\b\w+\b", line)) >= 18]
    prose_paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", stripped)
        if len(re.findall(r"\b\w+\b", paragraph)) >= 35
    ]
    if prose_paragraphs:
        return False
    short_heading_lines = [
        line
        for line in lines
        if len(re.findall(r"\b\w+\b", line)) <= 12 and not any(ch in line for ch in ".!?")
    ]
    uppercase_lines = [line for line in lines if line == line.upper()]
    return len(prose_lines) <= 1 and (
        len(short_heading_lines) >= max(4, len(lines) // 2) or len(uppercase_lines) >= max(3, len(lines) // 3)
    )


def prose_score(text: str) -> int:
    score = 0
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if len(re.findall(r"\b\w+\b", paragraph)) >= 35 and re.search(r"[.!?。！？]", paragraph):
            score += 1
    return score


def detect_mc_body_start_line(book: BookSpec, markdown_text: str) -> int:
    anchor_title = MC_BODY_START_HEADINGS.get(book.book_id)
    if not anchor_title:
        return 0
    lines = markdown_text.splitlines()
    first_hundred = "\n".join(lines[:100]).upper()
    if "VOLUME 1" not in first_hundred and "VOLUME 2" not in first_hundred:
        return 0
    anchor_key = normalize_heading_key(anchor_title)
    headings = find_heading_positions(lines)
    candidates = [(idx, raw_title) for idx, raw_title, key in headings if key == anchor_key]
    if not candidates:
        return 0
    chosen_idx = candidates[0][0]
    best_score = -1
    for idx, _raw_title in candidates:
        if idx < 150:
            continue
        immediate_sample = "\n".join(lines[idx : min(len(lines), idx + 20)]).strip()
        score = prose_score(immediate_sample)
        if score <= 0:
            continue
        if score > best_score:
            chosen_idx = idx
            best_score = score
    if best_score >= 0:
        return chosen_idx
    for idx, _raw_title in candidates:
        if idx < 150:
            continue
        sample = "\n".join(lines[idx : min(len(lines), idx + 80)]).strip()
        score = prose_score(sample)
        if score > best_score:
            chosen_idx = idx
            best_score = score
    return chosen_idx


def clean_merged_text_for_chunking(book: BookSpec, markdown_text: str) -> str:
    start_line = detect_mc_body_start_line(book, markdown_text)
    if start_line <= 0:
        return markdown_text
    lines = markdown_text.splitlines()
    cleaned = "\n".join(lines[start_line:]).strip()
    return cleaned or markdown_text


def build_toc_section_chapters(book: BookSpec, markdown_text: str) -> Optional[List[Chapter]]:
    toc = MC_SECTION_TOC.get(book.book_id)
    if not toc:
        return None
    lines = markdown_text.splitlines()
    headings = find_heading_positions(lines)
    chapters: List[Chapter] = []
    cursor = 0

    for chapter_index, chapter_cfg in enumerate(toc):
        chapter_title = str(chapter_cfg["chapter_title"])
        chapter_num = int(chapter_cfg["chapter_num"])
        chapter_key = normalize_heading_key(chapter_title)

        chapter_heading = next(
            ((idx, raw_title) for idx, raw_title, key in headings if idx >= cursor and key == chapter_key),
            None,
        )
        if not chapter_heading:
            continue
        chapter_start_idx, _ = chapter_heading

        next_chapter_start_idx = len(lines)
        if chapter_index + 1 < len(toc):
            next_chapter_key = normalize_heading_key(str(toc[chapter_index + 1]["chapter_title"]))
            next_heading = next(
                ((idx, raw_title) for idx, raw_title, key in headings if idx > chapter_start_idx and key == next_chapter_key),
                None,
            )
            if next_heading:
                next_chapter_start_idx = next_heading[0]

        section_positions: List[Tuple[int, str]] = []
        search_cursor = chapter_start_idx
        for section_title in chapter_cfg["sections"]:
            section_key = normalize_heading_key(str(section_title))
            match = next(
                (
                    (idx, raw_title)
                    for idx, raw_title, key in headings
                    if idx >= search_cursor and idx < next_chapter_start_idx and key == section_key
                ),
                None,
            )
            if not match:
                continue
            section_positions.append(match)
            search_cursor = match[0] + 1

        if not section_positions:
            chapter_text = "\n".join(lines[chapter_start_idx:next_chapter_start_idx]).strip()
            if chapter_text:
                next_title = toc[chapter_index + 1]["chapter_title"] if chapter_index + 1 < len(toc) else "End of book"
                chapters.append(
                    Chapter(
                        chapter_num=chapter_num,
                        chapter_title=chapter_title,
                        chapter_start=chapter_title,
                        chapter_end=str(next_title),
                        text=chapter_text,
                    )
                )
            cursor = next_chapter_start_idx
            continue

        intro_text = "\n".join(lines[chapter_start_idx:section_positions[0][0]]).strip()
        prepend_intro = (
            intro_text
            if intro_text and not looks_like_toc_noise(intro_text, chapter_cfg["sections"])
            else ""
        )

        for section_idx, (section_start_idx, section_raw_title) in enumerate(section_positions):
            section_end_idx = (
                section_positions[section_idx + 1][0]
                if section_idx + 1 < len(section_positions)
                else next_chapter_start_idx
            )
            next_title = (
                section_positions[section_idx + 1][1]
                if section_idx + 1 < len(section_positions)
                else (toc[chapter_index + 1]["chapter_title"] if chapter_index + 1 < len(toc) else "End of book")
            )
            section_text = "\n".join(lines[section_start_idx:section_end_idx]).strip()
            if section_idx == 0 and prepend_intro:
                section_text = f"{prepend_intro}\n\n{section_text}".strip()
            if not section_text:
                continue
            chapters.append(
                Chapter(
                    chapter_num=chapter_num,
                    chapter_title=chapter_title,
                    chapter_start=section_raw_title,
                    chapter_end=str(next_title),
                    text=section_text,
                )
            )
        cursor = next_chapter_start_idx

    return chapters or None


def split_markdown_into_book_chapters(book: BookSpec, markdown_text: str) -> Optional[List[Chapter]]:
    config = MC_PRIMARY_CHAPTERS.get(book.book_id)
    if not config:
        return None
    lines = markdown_text.splitlines()
    headings = find_heading_positions(lines)
    chapter_starts: List[Tuple[int, int, str]] = []
    used_indices: set[int] = set()
    for chapter_num, chapter_title in config:
        chapter_key = normalize_heading_key(chapter_title)
        match = next(
            (
                (idx, raw_title)
                for idx, raw_title, key in headings
                if idx not in used_indices and key == chapter_key
            ),
            None,
        )
        if not match:
            continue
        chapter_starts.append((match[0], chapter_num, chapter_title))
        used_indices.add(match[0])
    if not chapter_starts:
        return None
    chapter_starts.sort(key=lambda item: item[0])
    chapters: List[Chapter] = []
    for idx, (start_line, chapter_num, chapter_title) in enumerate(chapter_starts):
        end_line = chapter_starts[idx + 1][0] if idx + 1 < len(chapter_starts) else len(lines)
        next_title = chapter_starts[idx + 1][2] if idx + 1 < len(chapter_starts) else "End of book"
        chapter_text = "\n".join(lines[start_line:end_line]).strip()
        if not chapter_text:
            continue
        chapters.append(
            Chapter(
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                chapter_start=chapter_title,
                chapter_end=next_title,
                text=chapter_text,
            )
        )
    return chapters or None


def split_markdown_into_chapters(markdown_text: str) -> List[Chapter]:
    lines = markdown_text.splitlines()
    chapter_map = extract_toc_chapter_map(lines)
    chapter_starts: List[Tuple[int, int, str]] = []

    for idx, line in enumerate(lines):
        if not re.match(r"^#\s+", line):
            continue
        title = re.sub(r"^#\s+", "", line).strip()
        if title.upper().startswith("CHAPTER ") or title.upper().startswith("VOLUME "):
            continue
        key = normalize_heading_key(title)
        if key in chapter_map:
            chapter_num, canonical_title = chapter_map[key]
            chapter_starts.append((idx, chapter_num, canonical_title))

    if not chapter_starts:
        current_title = "Full Text"
        return [
            Chapter(
                chapter_num=1,
                chapter_title=current_title,
                chapter_start=current_title,
                chapter_end="End of book",
                text=markdown_text.strip(),
            )
        ]

    chapters: List[Chapter] = []
    intro_text = "\n".join(lines[: chapter_starts[0][0]]).strip()
    if intro_text:
        chapters.append(
            Chapter(
                chapter_num=1,
                chapter_title="Introduction",
                chapter_start="Start of book",
                chapter_end=chapter_starts[0][2],
                text=intro_text,
            )
        )

    for idx, (start_line, chapter_num, chapter_title) in enumerate(chapter_starts):
        end_line = chapter_starts[idx + 1][0] if idx + 1 < len(chapter_starts) else len(lines)
        next_title = chapter_starts[idx + 1][2] if idx + 1 < len(chapter_starts) else "End of book"
        chapter_text = "\n".join(lines[start_line:end_line]).strip()
        if not chapter_text:
            continue
        chapters.append(
            Chapter(
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                chapter_start=chapter_title,
                chapter_end=next_title,
                text=chapter_text,
            )
        )
    merged_chapters: List[Chapter] = []
    for chapter in chapters:
        if (
            merged_chapters
            and merged_chapters[-1].chapter_num == chapter.chapter_num
            and merged_chapters[-1].chapter_title == chapter.chapter_title
        ):
            merged_chapters[-1] = Chapter(
                chapter_num=chapter.chapter_num,
                chapter_title=chapter.chapter_title,
                chapter_start=merged_chapters[-1].chapter_start,
                chapter_end=chapter.chapter_end,
                text=f"{merged_chapters[-1].text}\n\n{chapter.text}".strip(),
            )
            continue
        merged_chapters.append(chapter)
    return merged_chapters


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
    text = str(body.get("response") or "").strip()
    if text:
        return text
    return str(body.get("thinking") or "").strip()


def get_semantic_chunker() -> "SemanticChunker":
    global _SEMANTIC_CHUNKER
    if _SEMANTIC_CHUNKER is not None:
        return _SEMANTIC_CHUNKER
    if SemanticChunker is None:
        raise PipelineError(
            "Chonkie is not installed for this Python environment. "
            "Install with: pip install 'chonkie[semantic]'"
        )
    _SEMANTIC_CHUNKER = SemanticChunker(
        embedding_model=CHONKIE_EMBEDDING_MODEL,
        chunk_size=CHONKIE_CHUNK_SIZE,
        min_chunk_size=CHONKIE_MIN_CHUNK_SIZE,
        threshold=CHONKIE_THRESHOLD,
    )
    return _SEMANTIC_CHUNKER


def split_large_block(text: str, max_chars: int) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    pieces: List[str] = []
    current: List[str] = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_len = len(paragraph) + 2
        if current and current_len + paragraph_len > max_chars:
            pieces.append("\n\n".join(current).strip())
            current = [paragraph]
            current_len = paragraph_len
        else:
            current.append(paragraph)
            current_len += paragraph_len
    if current:
        pieces.append("\n\n".join(current).strip())
    return pieces or [text.strip()]


def split_chapter_text_for_model(chapter_text: str, max_chars: int = 4500) -> List[str]:
    lines = chapter_text.splitlines()
    section_blocks: List[str] = []
    current_lines: List[str] = []
    for line in lines:
        if line.startswith("#") and current_lines:
            section_blocks.append("\n".join(current_lines).strip())
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        section_blocks.append("\n".join(current_lines).strip())

    segments: List[str] = []
    current_segment: List[str] = []
    current_len = 0
    for block in section_blocks:
        if not block:
            continue
        block_len = len(block) + 2
        if block_len > max_chars:
            if current_segment:
                segments.append("\n\n".join(current_segment).strip())
                current_segment = []
                current_len = 0
            segments.extend(split_large_block(block, max_chars=max_chars))
            continue
        if current_segment and current_len + block_len > max_chars:
            segments.append("\n\n".join(current_segment).strip())
            current_segment = [block]
            current_len = block_len
        else:
            current_segment.append(block)
            current_len += block_len
    if current_segment:
        segments.append("\n\n".join(current_segment).strip())
    normalized_segments = [segment for segment in segments if segment.strip()]
    if not normalized_segments:
        return [chapter_text.strip()]

    rebalanced_segments: List[str] = []
    idx = 0
    while idx < len(normalized_segments):
        current = normalized_segments[idx].strip()
        if len(current) < 300 and idx + 1 < len(normalized_segments):
            candidate = f"{current}\n\n{normalized_segments[idx + 1].strip()}".strip()
            if len(candidate) <= max_chars:
                rebalanced_segments.append(candidate)
                idx += 2
                continue
        if rebalanced_segments and len(current) < 160:
            candidate = f"{rebalanced_segments[-1]}\n\n{current}".strip()
            if len(candidate) <= max_chars:
                rebalanced_segments[-1] = candidate
                idx += 1
                continue
        rebalanced_segments.append(current)
        idx += 1
    return rebalanced_segments or [chapter_text.strip()]


def extract_chunks_from_payload(payload: Any) -> List[str]:
    if isinstance(payload, dict):
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            return [str(item).strip() for item in chunks if str(item).strip()]
        response = payload.get("response")
        if isinstance(response, str) and response.strip():
            nested = extract_json_value(response)
            return extract_chunks_from_payload(nested)
        return []
    if isinstance(payload, list):
        collected: List[str] = []
        for item in payload:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    collected.append(text)
                continue
            collected.extend(extract_chunks_from_payload(item))
        return collected
    return []


def sanitize_chunk_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("{") and cleaned.endswith("}"):
        for loader in (json.loads, ast.literal_eval):
            try:
                payload = loader(cleaned)
            except Exception:
                continue
            if isinstance(payload, dict):
                nested_text = payload.get("text")
                if isinstance(nested_text, str) and nested_text.strip():
                    cleaned = nested_text.strip()
                    break
    cleaned = cleaned.replace("\\n", "\n").strip()
    if (
        len(re.findall(r"\b\w+\b", cleaned)) < 12
        and not re.search(r"[.!?。！？]", cleaned)
        and re.fullmatch(r"[<>{}\[\]\w\s\-:,./']+", cleaned)
    ):
        return ""
    return cleaned


def chunk_word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def chunk_cjk_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "food_safety": (
        "pathogen",
        "pathogens",
        "bacteria",
        "bacterial",
        "spore",
        "spores",
        "toxin",
        "toxins",
        "pasteur",
        "steril",
        "contamin",
        "safe",
        "safety",
        "hygiene",
        "botuli",
        "canning safety",
    ),
    "chemical_reaction": (
        "maillard",
        "caramel",
        "oxid",
        "reaction",
        "react",
        "browning",
        "pyrolysis",
        "decomposition",
        "polymer",
        "acid-base",
        "acid base",
    ),
    "flavor_sensory": (
        "flavor",
        "aroma",
        "taste",
        "sensory",
        "mouthfeel",
        "palate",
        "fragrance",
        "odor",
        "smell",
        "texture",
        "crisp",
        "juicy",
        "tender",
    ),
}


TOPIC_FALLBACK_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("water_activity", ("water activity", "humidity", "moisture", "dehydrat", "drying", "humid", "boundary layer")),
    ("heat_transfer", ("heat", "thermal", "temperature", "convection", "conduction", "radiation", "steam", "draft")),
    ("equipment_physics", ("oven", "grill", "broiler", "microwave", "pressure cooker", "canner", "vacuum", "fan", "valve", "pump")),
    ("physical_change", ("boil", "evapor", "condens", "freeze", "melt", "phase", "crystal", "vapor pressure")),
]


def refine_annotation_topics(chunk_text: str, topics: Sequence[str]) -> List[str]:
    text = chunk_text.lower()
    refined: List[str] = []
    for topic in topics:
        if topic in {"food_safety", "chemical_reaction", "flavor_sensory"}:
            if not any(keyword in text for keyword in TOPIC_KEYWORDS[topic]):
                continue
        if topic not in refined:
            refined.append(topic)
    if refined:
        return refined[:2]
    for topic, keywords in TOPIC_FALLBACK_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return [topic]
    return [topics[0]] if topics else ["physical_change"]


def chunk_size_ok(text: str, min_words: int = 120, max_words: int = 320, min_cjk: int = 180, max_cjk: int = 400) -> bool:
    words = chunk_word_count(text)
    cjk = chunk_cjk_count(text)
    if cjk > 0:
        return min_cjk <= cjk <= max_cjk
    return min_words <= words <= max_words


def split_oversized_chunk(text: str, max_words: int = 260, max_cjk: int = 400) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_word_count(text) <= max_words and chunk_cjk_count(text) <= max_cjk:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]

    pieces: List[str] = []
    current: List[str] = []
    current_text = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current_text}\n\n{paragraph}"
        if current and (
            chunk_word_count(candidate) > max_words or chunk_cjk_count(candidate) > max_cjk
        ):
            pieces.append(current_text.strip())
            current = [paragraph]
            current_text = paragraph
        else:
            current.append(paragraph)
            current_text = candidate
    if current_text.strip():
        pieces.append(current_text.strip())

    refined: List[str] = []
    for piece in pieces:
        if piece != text and (chunk_word_count(piece) > max_words or chunk_cjk_count(piece) > max_cjk):
            refined.extend(split_oversized_chunk(piece, max_words=max_words, max_cjk=max_cjk))
        else:
            refined.append(piece)
    return [piece for piece in refined if piece.strip()]


def split_oversized(chunks: Sequence[str], max_len: int = 800) -> List[str]:
    result: List[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) <= max_len:
            result.append(chunk)
            continue
        if chunk.count("<tr") >= 2:
            sentences = [part.strip() for part in re.split(r"(?=<tr)", chunk) if part.strip()]
        elif chunk.count("> [figure") >= 2:
            sentences = [part.strip() for part in re.split(r"(?=>\s*\[figure)", chunk) if part.strip()]
        else:
            sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", chunk) if part.strip()]
        if len(sentences) <= 1:
            sentences = [part.strip() for part in re.split(r"\n\s*\n", chunk) if part.strip()]
        if len(sentences) <= 1:
            sentences = [part.strip() for part in chunk.splitlines() if part.strip()]
        if len(sentences) <= 1 and "<tr" in chunk:
            sentences = [part.strip() for part in re.split(r"(?=<tr)", chunk) if part.strip()]
        if len(sentences) <= 1 and "> [figure" in chunk:
            sentences = [part.strip() for part in re.split(r"(?=>\s*\[figure)", chunk) if part.strip()]
        if len(sentences) <= 1:
            words = chunk.split()
            sentences = []
            buf_words: List[str] = []
            for word in words:
                candidate = " ".join(buf_words + [word]).strip()
                if len(candidate) > max_len and buf_words:
                    sentences.append(" ".join(buf_words).strip())
                    buf_words = [word]
                else:
                    buf_words.append(word)
            if buf_words:
                sentences.append(" ".join(buf_words).strip())
        buf = ""
        for sentence in sentences:
            candidate = f"{buf} {sentence}".strip() if buf else sentence
            if len(candidate) > max_len and buf:
                result.append(buf.strip())
                buf = sentence
            else:
                buf = candidate
        if buf:
            result.append(buf.strip())
    return result


def rebalance_short_chunks(chunks: Sequence[str], min_len: int = CHONKIE_MIN_CHUNK_SIZE, max_len: int = 800) -> List[str]:
    normalized = [chunk.strip() for chunk in chunks if chunk.strip()]
    merged: List[str] = []
    idx = 0
    while idx < len(normalized):
        current = normalized[idx]
        if idx + 1 < len(normalized) and len(current) < min_len:
            candidate = f"{current}\n\n{normalized[idx + 1]}".strip()
            if len(candidate) <= max_len:
                merged.append(candidate)
                idx += 2
                continue
        if merged and len(current) < min_len:
            candidate = f"{merged[-1]}\n\n{current}".strip()
            if len(candidate) <= max_len or len(current) < 120:
                merged[-1] = candidate
                idx += 1
                continue
        if len(current) < 50 and current.lstrip().startswith("#"):
            idx += 1
            continue
        merged.append(current)
        idx += 1
    return merged


def normalize_chunk_sizes(chunks: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for chunk in chunks:
        cleaned = sanitize_chunk_text(chunk)
        if not cleaned:
            continue
        normalized.extend(split_oversized_chunk(cleaned))
    merged: List[str] = []
    buffer = ""
    for chunk in normalized:
        chunk = chunk.strip()
        if not chunk:
            continue
        if not buffer:
            buffer = chunk
            continue
        candidate = f"{buffer}\n\n{chunk}".strip()
        if chunk_size_ok(buffer):
            merged.append(buffer)
            buffer = chunk
            continue
        if chunk_word_count(candidate) <= 300 and chunk_cjk_count(candidate) <= 440:
            buffer = candidate
        else:
            merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)
    rebalanced: List[str] = []
    idx = 0
    while idx < len(merged):
        current = merged[idx].strip()
        if not current:
            idx += 1
            continue
        words = chunk_word_count(current)
        cjk = chunk_cjk_count(current)
        is_small_english = cjk == 0 and words < 90
        is_small_cjk = cjk > 0 and cjk < 140
        if (is_small_english or is_small_cjk) and idx + 1 < len(merged):
            candidate = f"{current}\n\n{merged[idx + 1].strip()}".strip()
            if chunk_word_count(candidate) <= 300 and chunk_cjk_count(candidate) <= 440:
                rebalanced.append(candidate)
                idx += 2
                continue
        if (
            rebalanced
            and ((is_small_english and chunk_word_count(f'{rebalanced[-1]}\n\n{current}') <= 300)
            or (is_small_cjk and chunk_cjk_count(f'{rebalanced[-1]}\n\n{current}') <= 440))
        ):
            rebalanced[-1] = f"{rebalanced[-1]}\n\n{current}".strip()
        else:
            rebalanced.append(current)
        idx += 1
    oversized_split = split_oversized(rebalanced, max_len=1200)
    compacted = rebalance_short_chunks(oversized_split, min_len=180, max_len=1200)
    final_chunks = split_oversized(compacted, max_len=1200)
    return [chunk for chunk in final_chunks if chunk.strip()]


def chapter_identity(chapter: Chapter) -> Tuple[int, str, str, str]:
    return (
        int(chapter.chapter_num),
        str(chapter.chapter_title),
        str(chapter.chapter_start),
        str(chapter.chapter_end),
    )


def split_segment_with_qwen(
    chapter: Chapter,
    segment_text: str,
    segment_label: str,
    *,
    max_depth: int = 2,
) -> List[str]:
    sanitized_segment = sanitize_chunk_text(segment_text)
    if not sanitized_segment:
        return []
    if len(sanitized_segment) < 180 and chunk_word_count(sanitized_segment) < 35:
        return [sanitized_segment]

    last_error: Optional[Exception] = None
    for _attempt in range(3):
        try:
            prompt = SPLIT_PROMPT_TEMPLATE.format(
                chapter_title=chapter.chapter_title,
                chapter_start=chapter.chapter_start,
                chapter_end=chapter.chapter_end,
                chapter_segment=segment_label,
                chapter_text=sanitized_segment,
            )
            raw = ollama_generate(SPLIT_MODEL, prompt, timeout=600)
            parsed = extract_json_value(raw)
            chunks = extract_chunks_from_payload(parsed)
            if chunks:
                return [chunk for chunk in chunks if chunk.strip()]
            last_error = PipelineError(f"Unexpected split response for chapter {chapter.chapter_title}")
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)

    if max_depth > 0 and len(sanitized_segment) > 1600:
        subsegments = split_chapter_text_for_model(sanitized_segment, max_chars=max(1600, len(sanitized_segment) // 2))
        if len(subsegments) > 1:
            collected: List[str] = []
            total_subsegments = len(subsegments)
            for idx, subsegment in enumerate(subsegments, start=1):
                sub_label = f"{segment_label} / retry {idx}/{total_subsegments}"
                collected.extend(
                    split_segment_with_qwen(chapter, subsegment, sub_label, max_depth=max_depth - 1)
                )
            if collected:
                return collected

    fallback_chunks = split_oversized_chunk(sanitized_segment, max_words=220, max_cjk=360)
    fallback_chunks = split_oversized(fallback_chunks, max_len=1200)
    fallback_chunks = [chunk.strip() for chunk in fallback_chunks if chunk.strip()]
    if fallback_chunks:
        return fallback_chunks
    if last_error is not None:
        raise last_error
    return [sanitized_segment]


def split_chapter_with_qwen_model(chapter: Chapter) -> List[str]:
    chapter_segments = split_chapter_text_for_model(chapter.text)
    combined_chunks: List[str] = []
    total_segments = len(chapter_segments)
    for idx, chapter_segment_text in enumerate(chapter_segments, start=1):
        segment_label = f"part {idx}/{total_segments}" if total_segments > 1 else "full chapter"
        segment_chunks = split_segment_with_qwen(chapter, chapter_segment_text, segment_label)
        combined_chunks.extend(segment_chunks)
    return normalize_chunk_sizes(combined_chunks)


def split_chapter_with_chonkie(chapter_text: str, chapter_title: str) -> List[str]:
    chunker = get_semantic_chunker()
    raw_chunks = chunker.chunk(chapter_text)
    texts: List[str] = []
    for chunk in raw_chunks:
        text = sanitize_chunk_text(str(getattr(chunk, "text", "") or ""))
        if len(text) >= CHONKIE_MIN_CHUNK_SIZE:
            texts.append(text)
    merged: List[str] = []
    buffer = ""
    for text in texts:
        candidate = f"{buffer}\n\n{text}".strip() if buffer else text
        if buffer and len(candidate) >= 500:
            merged.append(buffer)
            buffer = text
            continue
        buffer = candidate
    if buffer:
        merged.append(buffer)
    normalized = [chunk.strip() for chunk in merged if len(chunk.strip()) >= CHONKIE_MIN_CHUNK_SIZE]
    rebalanced: List[str] = []
    idx = 0
    while idx < len(normalized):
        current = normalized[idx]
        if idx + 1 < len(normalized) and (
            looks_like_toc_noise(current)
            or (current.lstrip().startswith("#") and chunk_word_count(current) < 40)
        ):
            rebalanced.append(f"{current}\n\n{normalized[idx + 1]}".strip())
            idx += 2
            continue
        rebalanced.append(current)
        idx += 1
    if not normalized:
        fallback = sanitize_chunk_text(chapter_text)
        if len(fallback) >= CHONKIE_MIN_CHUNK_SIZE:
            return [fallback]
    oversized_split = split_oversized(rebalanced, max_len=800)
    compacted = rebalance_short_chunks(oversized_split, min_len=CHONKIE_MIN_CHUNK_SIZE, max_len=800)
    final_chunks = split_oversized(compacted, max_len=800)
    return rebalance_short_chunks(final_chunks, min_len=CHONKIE_MIN_CHUNK_SIZE, max_len=800)


def split_chapter_with_model(book: BookSpec, chapter: Chapter) -> List[str]:
    if book.book_id in BOOKS_USE_QWEN_SPLIT:
        return split_chapter_with_qwen_model(chapter)
    return split_chapter_with_chonkie(chapter.text, chapter.chapter_title)


def summarize_step4_chunks(chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [len(str(item.get("full_text", "") or "")) for item in chunks if str(item.get("full_text", "") or "").strip()]
    if not lengths:
        return {
            "total_chunks": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "lt50_chars": 0,
            "lt150_chars": 0,
        }
    return {
        "total_chunks": len(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 2),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "lt50_chars": sum(1 for value in lengths if value < 50),
        "lt150_chars": sum(1 for value in lengths if value < 150),
    }


def repair_short_step4_chunks(chunks: Sequence[Dict[str, Any]], min_chars: int = 150, max_chars: int = 1400) -> List[Dict[str, Any]]:
    repaired: List[Dict[str, Any]] = [dict(item) for item in chunks]
    idx = 0
    while idx < len(repaired):
        current = repaired[idx]
        text = str(current.get("full_text", "") or "").strip()
        if not text or len(text) >= min_chars:
            idx += 1
            continue

        prev_idx = idx - 1 if idx > 0 else None
        next_idx = idx + 1 if idx + 1 < len(repaired) else None
        merged = False

        def same_section(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            return (
                a.get("chapter_num") == b.get("chapter_num")
                and a.get("chapter_title") == b.get("chapter_title")
                and a.get("chapter_start") == b.get("chapter_start")
                and a.get("chapter_end") == b.get("chapter_end")
                and a.get("source_book") == b.get("source_book")
            )

        heading_like = (
            len(text) <= 120
            and "\n" not in text
            and not re.search(r"[.!?。！？:;]", text)
        )

        if prev_idx is not None and same_section(repaired[prev_idx], current):
            candidate = f"{repaired[prev_idx]['full_text'].rstrip()}\n\n{text}".strip()
            if len(candidate) <= max_chars:
                repaired[prev_idx]["full_text"] = candidate
                repaired.pop(idx)
                merged = True
        if not merged and next_idx is not None and next_idx < len(repaired) and same_section(current, repaired[next_idx]):
            candidate = f"{text}\n\n{repaired[next_idx]['full_text'].lstrip()}".strip()
            if len(candidate) <= max_chars or (heading_like and len(candidate) <= 5000):
                repaired[next_idx]["full_text"] = candidate
                repaired.pop(idx)
                merged = True
        if not merged:
            idx += 1

    for new_idx, item in enumerate(repaired):
        item["chunk_idx"] = new_idx
    return repaired


def run_step4_chunk(book: BookSpec, book_dir: Path) -> Tuple[Path, int]:
    merged_path = book_dir / "raw_merged.md"
    chunks_path = book_dir / "chunks_raw.json"
    failed_path = book_dir / "failed_chapters.json"
    quality_path = book_dir / "step4_quality.json"
    merged_text = merged_path.read_text(encoding="utf-8", errors="ignore")
    cleaned_merged_text = clean_merged_text_for_chunking(book, merged_text)
    chapters = (
        build_toc_section_chapters(book, cleaned_merged_text)
        or split_markdown_into_book_chapters(book, cleaned_merged_text)
        or split_markdown_into_chapters(cleaned_merged_text)
    )
    existing_chunks = load_json_list(chunks_path)
    failed_chapters: List[Dict[str, Any]] = []
    done_keys = set()
    for item in existing_chunks:
        if "chapter_num" not in item:
            continue
        if "chapter_start" in item and "chapter_end" in item:
            done_keys.add(
                (
                    int(item["chapter_num"]),
                    str(item["chapter_title"]),
                    str(item["chapter_start"]),
                    str(item["chapter_end"]),
                )
            )
    next_chunk_idx = (max((int(item["chunk_idx"]) for item in existing_chunks), default=-1) + 1)
    print(f"[step4] chapter chunking for {len(chapters)} chapter(s)")
    for chapter in chapters:
        chapter_key = chapter_identity(chapter)
        if chapter_key in done_keys:
            continue
        print(
            f"  [step4] chapter {chapter.chapter_num}: "
            f"{chapter.chapter_title} | {chapter.chapter_start} -> {chapter.chapter_end}"
        )
        try:
            chapter_chunks = split_chapter_with_model(book, chapter)
            for chunk_text in chapter_chunks:
                existing_chunks.append(
                    {
                        "chunk_idx": next_chunk_idx,
                        "full_text": chunk_text,
                        "chapter_num": chapter.chapter_num,
                        "chapter_title": chapter.chapter_title,
                        "chapter_start": chapter.chapter_start,
                        "chapter_end": chapter.chapter_end,
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
                    "chapter_start": chapter.chapter_start,
                    "chapter_end": chapter.chapter_end,
                    "error": str(exc),
                    "updated_at": now_iso(),
                }
            )
            save_json(failed_path, failed_chapters)
    save_json(failed_path, failed_chapters)
    if not chunks_path.exists():
        save_json(chunks_path, existing_chunks)
    existing_chunks = repair_short_step4_chunks(existing_chunks)
    save_json(chunks_path, existing_chunks)
    quality = summarize_step4_chunks(existing_chunks)
    quality["failed_chapters"] = len(failed_chapters)
    save_json(quality_path, quality)
    print(
        "[step4] quality "
        f"total={quality['total_chunks']} avg_chars={quality['avg_chars']} "
        f"min={quality['min_chars']} max={quality['max_chars']} "
        f"lt50={quality['lt50_chars']} lt150={quality['lt150_chars']} "
        f"failed={quality['failed_chapters']}"
    )
    if quality["total_chunks"] <= 0:
        raise PipelineError("step4 produced 0 chunks")
    if quality["lt50_chars"] > 0:
        raise PipelineError(f"step4 produced {quality['lt50_chars']} chunks shorter than 50 chars")
    return chunks_path, len(existing_chunks)


def annotate_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    prompt = ANNOTATE_PROMPT_TEMPLATE.format(
        topics=", ".join(VALID_TOPICS),
        chapter_title=chunk.get("chapter_title", ""),
        chapter_start=chunk.get("chapter_start", ""),
        chapter_end=chunk.get("chapter_end", ""),
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
    topics = refine_annotation_topics(chunk["full_text"], topics)
    return {
        "summary": summary[:50],
        "topics": topics[:2],
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
