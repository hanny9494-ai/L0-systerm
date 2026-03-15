# MC Stage1 Handoff

## Scope

This handoff covers the Modernist Cuisine Stage1 pipeline for:

- `mc_vol2`
- `mc_vol3`
- `mc_vol4`

The main script is:

- `/tmp/L0-systerm/scripts/batch_mc_stage1.py`

Related helper scripts:

- `/tmp/L0-systerm/scripts/merge_mineru_qwen.py`
- `/tmp/L0-systerm/scripts/repair_missing_vision_pages.py`
- `/tmp/L0-systerm/scripts/repair_mc_conversion.py`

Outputs live under:

- `/Users/jeff/l0-knowledge-engine/output/mc`

## Pipeline

Per book:

1. `Step0`: EPUB to PDF if needed
2. `Step1`: MinerU extraction, with PDF splitting and resume support
3. `Step2`: Qwen vision extraction
4. `Step3`: MinerU + Qwen merge
5. `Step4`: TOC-aware subsection split, then `qwen3.5:2b`
6. `Step5`: `qwen3.5:9b` summary + topics annotation

## Current Step4 Design

All MC books now use the same split strategy:

1. Trim front matter and shared multi-volume TOC
2. Locate the book body anchor
3. Locate subsection boundaries from `MC_SECTION_TOC`
4. Send each subsection to `qwen3.5:2b`
5. Normalize chunk sizes
6. Run a post-check and write `step4_quality.json`

This is implemented in:

- `MC_SECTION_TOC`
- `build_toc_section_chapters()`
- `split_chapter_with_qwen_model()`
- `repair_short_step4_chunks()`
- `run_step4_chunk()`

## Quality Gates

Step4 writes:

- `chunks_raw.json`
- `failed_chapters.json`
- `step4_quality.json`

Current Step4 gate:

- `total_chunks > 0`
- `lt50_chars == 0`
- short chunks are auto-merged where possible

Step5 writes:

- `stage1/chunks_smart.json`
- `stage1/annotation_failures.json`

Completion should mean:

- `len(chunks_smart.json) == len(chunks_raw.json)`
- `annotation_failures.json == []`

## Resume Semantics

`infer_resume_status()` was tightened so status is inferred from valid outputs, not just file existence.

Key rule:

- `completed` only if `chunks_smart.json` count matches `chunks_raw.json` count and there are no annotation failures

## Dependencies

Required services and tools:

- MinerU API
- DashScope `qwen3-vl-plus`
- Ollama at `http://localhost:11434`
- `qwen3.5:2b`
- `qwen3.5:9b`

Python packages:

- `requests`
- `fitz` / PyMuPDF
- optional `chonkie[semantic]`

Note:

- Chonkie support remains in the script, but MC vol2/3/4 Step4 is currently routed through TOC-aware `2b`, not Chonkie.

## Book TOC Config

The TOC boundaries are stored in `MC_SECTION_TOC`.

Current configured books:

- `mc_vol2`
- `mc_vol3`
- `mc_vol4`

If a new MC volume or similar structured book is added, the expected path is:

1. Add TOC config
2. Verify front-matter anchor
3. Run Step4
4. Check `step4_quality.json`
5. Only then run Step5

## Single-Step Execution

There is no dedicated `--book` or `--step` CLI yet. The reliable pattern is module-level invocation.

Example: rerun Step4 for `mc_vol4`

```bash
cd /tmp/L0-systerm
python3 -u -c "from pathlib import Path; import scripts.batch_mc_stage1 as mod; output_root=Path('/Users/jeff/l0-knowledge-engine/output/mc'); book=next(book for book in mod.normalize_books(mod.DEFAULT_BOOKS) if book.book_id=='mc_vol4'); book_dir=output_root / book.slug; progress_path=output_root / 'batch_progress.json'; progress=mod.load_batch_progress(progress_path); chunks_path,total_chunks=mod.run_step4_chunk(book, book_dir); mod.set_book_status(progress, progress_path, book.book_id, status='step4_done', total_chunks=total_chunks)"
```

Example: rerun Step5 for `mc_vol3`

```bash
cd /tmp/L0-systerm
python3 -u -c "from pathlib import Path; import scripts.batch_mc_stage1 as mod; book_dir=Path('/Users/jeff/l0-knowledge-engine/output/mc/vol3'); out_path,total=mod.run_step5_annotate(book_dir); print(out_path, total)"
```

## Current Output Snapshot

As of `2026-03-15`, the practical state is:

- `vol2`: Step4 repaired to pass quality gate; Step5 rerun in progress or should be rerun from the repaired `485` chunks
- `vol3`: Step4 and Step5 complete on the current `502` chunks
- `vol4`: Step4 repaired to `707` chunks and passed quality gate; Step5 has been started and most chunks are annotated, but final completion should be verified from `chunks_smart.json` and `annotation_failures.json`

Always trust the output files more than stale logs or stale batch state.

## Known Open Items

1. `batch_progress.json` can lag behind actual output files if a step is run manually outside the main batch loop.
2. There is still no first-class CLI for `--book` and `--step`; reruns currently use module-level Python entrypoints.
3. `scripts/__pycache__/` is local noise and should not be committed.
4. Old failure logs may remain after a successful rerun unless explicitly reset.

## Recommended Next Cleanup

1. Add a proper CLI:
   - `--book-id`
   - `--start-step`
   - `--stop-step`
2. Add a `20` minute watchdog:
   - if `chunks_raw.json` or `chunks_smart.json` does not grow for `20` minutes, print the active subsection and fail fast
3. Centralize state repair:
   - recompute `batch_progress.json` from output files
4. Add a dedicated `retry_failed_annotations()` helper for Step5

## Handoff Checklist

Before handing to the next operator:

1. Check `batch_progress.json`
2. Check each book's `step4_quality.json`
3. Check `annotation_failures.json`
4. Confirm `len(chunks_smart.json) == len(chunks_raw.json)` for finished books
5. Only then mark the book `completed`
