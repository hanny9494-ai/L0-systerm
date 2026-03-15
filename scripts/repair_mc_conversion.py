#!/usr/bin/env python3
"""Retry MC vision extraction and rebuild merged markdown for selected books."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


SOURCE_MAP = {
    "vol2": Path("/Users/jeff/Documents/厨书数据库/工具科学书/Volume 2 - Techniques and Equipment.pdf"),
    "vol3": Path("/Users/jeff/Documents/厨书数据库/工具科学书/Volume 3 - Animals and Plants.pdf"),
    "vol4": Path("/Users/jeff/Documents/厨书数据库/工具科学书/Volume 4 - Ingredients and Preparations.pdf"),
}


def load_batch_module():
    script_path = Path("/tmp/L0-systerm/scripts/batch_mc_stage1.py")
    spec = importlib.util.spec_from_file_location("batch_mc_stage1", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("books", nargs="+", choices=sorted(SOURCE_MAP.keys()))
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--force-vision", action="store_true")
    args = parser.parse_args()

    mod = load_batch_module()
    for book in args.books:
        book_dir = Path(f"/Users/jeff/l0-knowledge-engine/output/mc/{book}")
        print(f"=== {book} ===", flush=True)
        if not args.skip_vision:
            mod.run_step2_vision(SOURCE_MAP[book], book_dir, force=args.force_vision)
        mod.run_step3_merge(book_dir)
        print(f"=== {book} done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
