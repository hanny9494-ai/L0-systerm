#!/usr/bin/env python3
"""Repair missing or failed Qwen vision pages for a single book."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


def load_batch_module():
    script_path = Path("/tmp/L0-systerm/scripts/batch_mc_stage1.py")
    spec = importlib.util.spec_from_file_location("batch_mc_stage1", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--book-dir", required=True)
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is required")

    mod = load_batch_module()
    pdf_path = Path(args.pdf)
    book_dir = Path(args.book_dir)
    vision_path = book_dir / "raw_vision.json"
    pages_dir = book_dir / "pages_150dpi"

    page_paths = mod.render_pdf_pages(pdf_path, pages_dir, dpi=mod.PNG_DPI)
    target_pages = mod.build_visual_target_pages(book_dir)
    page_meta_map = mod.build_page_metadata_map(book_dir)
    existing = mod.load_json(vision_path, {"pages": []})
    page_results = {int(item["page_num"]): item for item in existing.get("pages", []) if "page_num" in item}

    missing_pages = [
        page_num
        for page_num in target_pages
        if (page_num not in page_results)
        or page_results[page_num].get("error")
        or page_results[page_num].get("parsed") is None
    ]
    print(
        f"[step2] repairing {len(missing_pages)} page(s) out of {len(target_pages)} target page(s)",
        flush=True,
    )

    def persist() -> None:
        mod.save_json(
            vision_path,
            {
                "expected_pages": len(page_paths),
                "target_pages": target_pages,
                "updated_at": mod.now_iso(),
                "pages": [page_results[idx] for idx in sorted(page_results)],
            },
        )

    for idx, page_num in enumerate(missing_pages, start=1):
        png_path = page_paths[page_num - 1]
        print(f"[step2] page {page_num} ({idx}/{len(missing_pages)})", flush=True)
        page_meta = page_meta_map.get(
            page_num,
            {"pdf_page_num": page_num, "book_page_label": None, "book_page_candidates": []},
        )
        try:
            result = mod.recognize_page_visual(api_key, png_path, page_num)
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

    print("[step2] repair complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
