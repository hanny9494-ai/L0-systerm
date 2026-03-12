#!/usr/bin/env python3
"""Merge MinerU markdown with page-aware Qwen vision output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PLACEHOLDER_RE = re.compile(r"!\[.*?\]\(([^)]+)\)")
PAGE_IMAGE_RE = re.compile(r"(?:^|/)p(\d+)_img\d+\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def iter_pdf_pages(content_list: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(content_list, dict):
        pdf_info = content_list.get("pdf_info") or []
    elif isinstance(content_list, list):
        pdf_info = content_list
    else:
        pdf_info = []
    for page in pdf_info:
        if isinstance(page, dict):
            yield page


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


def discover_content_lists(paths: Sequence[str | Path]) -> List[Path]:
    discovered: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*_content_list.json")))
        elif path.exists():
            discovered.append(path)
    seen: set[Path] = set()
    deduped: List[Path] = []
    for path in discovered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def load_part_offsets(content_list_paths: Sequence[Path]) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    for path in content_list_paths:
        part_id = path.stem.removesuffix("_content_list")
        progress_path = path.parents[2] / "mineru_parts_progress.json"
        progress = load_json(progress_path, {"parts": []})
        for part in progress.get("parts", []):
            if str(part.get("part_id") or "") != part_id:
                continue
            offsets[part_id] = int(part.get("page_start") or 1)
            break
    return offsets


def build_image_page_map(content_list_paths: Sequence[Path]) -> Dict[str, int]:
    part_offsets = load_part_offsets(content_list_paths)
    image_page_map: Dict[str, int] = {}
    for path in content_list_paths:
        part_id = path.stem.removesuffix("_content_list")
        page_start = part_offsets.get(part_id, 1)
        content_list = load_json(path, {})
        for page in iter_pdf_pages(content_list):
            page_num = int(page.get("page_num") or 0)
            if page_num <= 0:
                page_num = page_start + int(page.get("page_idx") or 0)
            image_paths: List[str] = []
            collect_image_paths(page, image_paths)
            for image_name in image_paths:
                image_page_map.setdefault(Path(image_name).name, page_num)
    return image_page_map


def format_table_item(table: Dict[str, Any], page_num: int, book_page_label: Optional[str]) -> Optional[str]:
    markdown = str((table or {}).get("markdown") or "").strip()
    if not markdown:
        return None
    title = str((table or {}).get("title") or "").strip()
    notes = str((table or {}).get("notes") or "").strip()
    suffix = f"pdf-page {page_num}"
    if book_page_label:
        suffix += f", book-page {book_page_label}"
    header = f"<!-- qwen3-vl-plus {suffix} table"
    if title:
        header += f": {title}"
    header += " -->"
    block = f"{header}\n{markdown}"
    if notes:
        block += f"\n\n> Note: {notes}"
    return block


def format_figure_item(figure: Dict[str, Any], page_num: int, book_page_label: Optional[str]) -> Optional[str]:
    description = str((figure or {}).get("description") or "").strip()
    if not description:
        return None
    figure_type = str((figure or {}).get("type") or "figure").strip() or "figure"
    suffix = f"pdf-page {page_num}"
    if book_page_label:
        suffix += f", book-page {book_page_label}"
    return f"> [{figure_type} {suffix}] {description}"


def format_text_block_item(content: str, page_num: int, book_page_label: Optional[str]) -> Optional[str]:
    text = str(content or "").strip()
    if not text:
        return None
    suffix = f"pdf-page {page_num}"
    if book_page_label:
        suffix += f", book-page {book_page_label}"
    return f"> [text_block {suffix}] {text}"


def normalize_page_elements(parsed: Any) -> List[Dict[str, str]]:
    if isinstance(parsed, list):
        return [
            {
                "type": str(item.get("type") or "").strip().lower(),
                "content": str(item.get("content") or "").strip(),
            }
            for item in parsed
            if isinstance(item, dict) and str(item.get("content") or "").strip()
        ]
    if isinstance(parsed, dict):
        raw_elements = parsed.get("elements")
        if isinstance(raw_elements, list):
            return [
                {
                    "type": str(item.get("type") or "").strip().lower(),
                    "content": str(item.get("content") or "").strip(),
                    "figure_type": str(item.get("figure_type") or "").strip().lower(),
                }
                for item in raw_elements
                if isinstance(item, dict) and str(item.get("content") or "").strip()
            ]
    return []


def build_page_items(entry: Dict[str, Any]) -> List[str]:
    if entry.get("error"):
        return []
    parsed = entry.get("parsed") or {}
    page_num = int(entry.get("pdf_page_num") or entry.get("page_num") or 0)
    book_page_label = entry.get("book_page_label")
    items: List[str] = []
    elements = normalize_page_elements(parsed)
    if elements:
        for element in elements:
            element_type = element.get("type") or ""
            content = element.get("content") or ""
            if element_type == "table":
                item = format_table_item({"markdown": content, "title": "", "notes": ""}, page_num, book_page_label)
            elif element_type == "figure":
                item = format_figure_item(
                    {"type": element.get("figure_type") or "figure", "description": content},
                    page_num,
                    book_page_label,
                )
            elif element_type == "text_block":
                item = format_text_block_item(content, page_num, book_page_label)
            else:
                item = format_text_block_item(content, page_num, book_page_label)
            if item:
                items.append(item)
        return items
    for table in parsed.get("tables") or []:
        item = format_table_item(table, page_num, book_page_label)
        if item:
            items.append(item)
    for figure in parsed.get("figures") or []:
        item = format_figure_item(figure, page_num, book_page_label)
        if item:
            items.append(item)
    for text_block in parsed.get("text_blocks") or []:
        item = format_text_block_item(str(text_block), page_num, book_page_label)
        if item:
            items.append(item)
    page_note = str(parsed.get("page_note") or "").strip()
    if page_note:
        suffix = f"pdf-page {page_num}"
        if book_page_label:
            suffix += f", book-page {book_page_label}"
        items.append(f"> [{suffix}] {page_note}")
    return items


def summarize_page_items(items: Sequence[str]) -> str:
    snippets: List[str] = []
    for item in items:
        text = re.sub(r"<!--.*?-->", "", item, flags=re.DOTALL)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            snippets.append(text)
        if len(snippets) >= 3:
            break
    summary = " ".join(snippets).strip()
    if len(summary) > 360:
        summary = summary[:357].rstrip() + "..."
    return summary


def build_summary_fallback(page_num: int, occurrence: int, items: Sequence[str], book_page_label: Optional[str]) -> str:
    suffix = f"pdf-page {page_num}"
    if book_page_label:
        suffix += f", book-page {book_page_label}"
    summary = summarize_page_items(items)
    if summary:
        return (
            f"> [第{occurrence}个图片/表格，来自同页内容汇总，{suffix}] "
            f"同页视觉内容已超过逐项分配数量，请参考本页已提取内容。摘要：{summary}"
        )
    return f"> [第{occurrence}个图片/表格，来自同页内容汇总，{suffix}] 同页已有视觉内容，但当前摘要为空。"


def build_warning_fallback(page_num: Optional[int], reason: Optional[str], book_page_label: Optional[str]) -> str:
    if page_num is None:
        page_label = "页码未知"
    else:
        page_label = f"页码{page_num}"
        if book_page_label:
            page_label += f" / 书页{book_page_label}"
    reason_label = {
        "qwen_error": "qwen识别失败",
        "no_qwen_page": "qwen结果缺失",
        "no_visual_items": "该页未提取到可分配视觉内容",
        "no_page_mapping": "MinerU页码映射缺失",
    }.get(reason or "", "内容待补充")
    return f"> [图片/表格，{page_label}，内容待补充] {reason_label}。"


def build_vision_page_map(vision_path: Path) -> Dict[int, Dict[str, Any]]:
    data = load_json(vision_path, {"pages": []})
    page_map: Dict[int, Dict[str, Any]] = {}
    for entry in data.get("pages", []):
        if not isinstance(entry, dict):
            continue
        page_num = int(entry.get("pdf_page_num") or entry.get("page_num") or 0)
        if page_num <= 0:
            continue
        page_map[page_num] = entry
    return page_map


def parse_placeholder_page(image_ref: str, image_page_map: Dict[str, int]) -> Tuple[Optional[int], str]:
    image_name = Path(image_ref).name
    direct_match = PAGE_IMAGE_RE.search(image_ref)
    if direct_match:
        return int(direct_match.group(1)), image_name
    return image_page_map.get(image_name), image_name


def merge_mineru_qwen(
    mineru_path: str | Path,
    vision_path: str | Path,
    content_list_paths: Sequence[str | Path],
    output_path: str | Path,
    report_path: str | Path | None = None,
) -> Dict[str, Any]:
    mineru_path = Path(mineru_path)
    vision_path = Path(vision_path)
    output_path = Path(output_path)
    content_paths = discover_content_lists(content_list_paths)
    image_page_map = build_image_page_map(content_paths)
    vision_page_map = build_vision_page_map(vision_path)
    page_items = {page_num: build_page_items(entry) for page_num, entry in vision_page_map.items()}
    page_cursors = {page_num: 0 for page_num in page_items}

    mineru_text = mineru_path.read_text(encoding="utf-8", errors="ignore")
    replaced_parts: List[str] = []
    missed_list: List[Dict[str, Any]] = []
    cursor = 0
    replaced_count = 0
    direct_replacements = 0
    fallback_replacements = 0
    total_placeholders = 0
    page_occurrences: Dict[int, int] = {}

    for idx, match in enumerate(PLACEHOLDER_RE.finditer(mineru_text), start=1):
        replaced_parts.append(mineru_text[cursor : match.start()])
        image_ref = match.group(1)
        page_num, image_name = parse_placeholder_page(image_ref, image_page_map)
        if page_num is not None:
            page_occurrences[page_num] = page_occurrences.get(page_num, 0) + 1
        replacement = None
        reason = None
        fallback_mode = None
        book_page_label = None
        if page_num is None:
            reason = "no_page_mapping"
        else:
            vision_entry = vision_page_map.get(page_num)
            if vision_entry:
                book_page_label = vision_entry.get("book_page_label")
            if not vision_entry:
                reason = "no_qwen_page"
            elif vision_entry.get("error"):
                reason = "qwen_error"
            else:
                items = page_items.get(page_num) or []
                cursor_idx = page_cursors.get(page_num, 0)
                if cursor_idx < len(items):
                    replacement = items[cursor_idx]
                    page_cursors[page_num] = cursor_idx + 1
                    direct_replacements += 1
                elif items:
                    reason = "page_items_exhausted"
                    fallback_mode = "page_summary"
                    replacement = build_summary_fallback(
                        page_num=page_num,
                        occurrence=page_occurrences.get(page_num, cursor_idx + 1),
                        items=items,
                        book_page_label=book_page_label,
                    )
                else:
                    reason = "no_visual_items"
        if not replacement and reason and reason != "page_items_exhausted":
            fallback_mode = "warning_note"
            replacement = build_warning_fallback(page_num, reason, book_page_label)
        if replacement:
            replaced_parts.append("\n" + replacement + "\n")
            replaced_count += 1
            if fallback_mode:
                fallback_replacements += 1
        else:
            replaced_parts.append(match.group(0))
        if reason:
            missed_list.append(
                {
                    "placeholder_idx": idx,
                    "image_name": image_name,
                    "page_num": page_num,
                    "reason": reason,
                    "fallback_mode": fallback_mode,
                }
            )
        total_placeholders += 1
        cursor = match.end()

    replaced_parts.append(mineru_text[cursor:])
    merged_text = "".join(replaced_parts)
    merged_text = re.sub(r"\n{4,}", "\n\n\n", merged_text).strip() + "\n"
    output_path.write_text(merged_text, encoding="utf-8")
    residual_placeholders = len(PLACEHOLDER_RE.findall(merged_text))

    misses_by_reason: Dict[str, int] = {}
    for miss in missed_list:
        reason = str(miss["reason"])
        misses_by_reason[reason] = misses_by_reason.get(reason, 0) + 1

    unused_qwen: List[Dict[str, Any]] = []
    for page_num, entry in sorted(vision_page_map.items()):
        items = page_items.get(page_num) or []
        consumed = page_cursors.get(page_num, 0)
        remaining = items[consumed:]
        if not remaining:
            continue
        unused_qwen.append(
            {
                "page_num": page_num,
                "book_page_label": entry.get("book_page_label"),
                "unused_count": len(remaining),
                "items": remaining,
            }
        )

    report = {
        "mineru_path": str(mineru_path),
        "vision_path": str(vision_path),
        "content_lists": [str(path) for path in content_paths],
        "output_path": str(output_path),
        "total_placeholders": total_placeholders,
        "replaced_placeholders": replaced_count,
        "direct_replacements": direct_replacements,
        "fallback_replacements": fallback_replacements,
        "residual_placeholders": residual_placeholders,
        "misses_by_reason": misses_by_reason,
        "missed_list": missed_list,
        "unused_qwen_pages": len(unused_qwen),
        "unused_qwen_items": sum(item["unused_count"] for item in unused_qwen),
        "unused_qwen": unused_qwen,
        "vision_pages": len(vision_page_map),
        "vision_pages_with_errors": sorted(
            page_num for page_num, entry in vision_page_map.items() if entry.get("error")
        ),
    }

    if report_path is None:
        report_path = output_path.with_name(f"{output_path.stem}_report.json")
    report_path = Path(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mineru", required=True, help="Path to raw_mineru.md")
    parser.add_argument("--vision", required=True, help="Path to raw_vision.json")
    parser.add_argument(
        "--content-list",
        required=True,
        action="append",
        dest="content_lists",
        help="Path to a MinerU content_list JSON or a directory containing them. Repeatable.",
    )
    parser.add_argument("--output", required=True, help="Path to raw_merged.md")
    parser.add_argument("--report", help="Optional path for merge quality report JSON")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    report = merge_mineru_qwen(
        mineru_path=args.mineru,
        vision_path=args.vision,
        content_list_paths=args.content_lists,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(
        {
            "replaced_placeholders": report["replaced_placeholders"],
            "direct_replacements": report["direct_replacements"],
            "fallback_replacements": report["fallback_replacements"],
            "residual_placeholders": report["residual_placeholders"],
            "unused_qwen_items": report["unused_qwen_items"],
            "vision_pages_with_errors": report["vision_pages_with_errors"],
            "report_path": str(Path(args.report) if args.report else Path(args.output).with_name(f'{Path(args.output).stem}_report.json')),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
