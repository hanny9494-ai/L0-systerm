# MC Stage1 Status Snapshot

Date:

- `2026-03-15`

## Book Status

### vol2

- Step4 repaired from `488` chunks to `485` chunks
- `step4_quality.json` now passes:
  - `lt50_chars = 0`
  - `lt150_chars = 0`
- Step5 rerun is in progress against the repaired `485` chunks

Key files:

- `/Users/jeff/l0-knowledge-engine/output/mc/vol2/chunks_raw.json`
- `/Users/jeff/l0-knowledge-engine/output/mc/vol2/step4_quality.json`
- `/Users/jeff/l0-knowledge-engine/output/mc/logs/vol2_step5_rerun.log`

### vol3

- Step4 complete
- Step5 complete
- `chunks_raw.json = 502`
- `chunks_smart.json = 502`
- `annotation_failures.json = []`
- stale Step4 failure state was cleaned and `step4_quality.json` was written

Key files:

- `/Users/jeff/l0-knowledge-engine/output/mc/vol3/chunks_raw.json`
- `/Users/jeff/l0-knowledge-engine/output/mc/vol3/step4_quality.json`
- `/Users/jeff/l0-knowledge-engine/output/mc/vol3/stage1/chunks_smart.json`

### vol4

- Step4 repaired and passed quality gate
- final Step4 stats:
  - `707` chunks
  - `lt50_chars = 0`
  - `lt150_chars = 0`
  - `failed_chapters = 0`
- Step5 completed with partial annotation success:
  - `chunks_smart.json = 703`
  - `annotation_failures.json = 4`

Remaining failures:

- `chunk_idx 50`
- `chunk_idx 94`
- `chunk_idx 697`
- `chunk_idx 698`

Key files:

- `/Users/jeff/l0-knowledge-engine/output/mc/vol4/chunks_raw.json`
- `/Users/jeff/l0-knowledge-engine/output/mc/vol4/step4_quality.json`
- `/Users/jeff/l0-knowledge-engine/output/mc/vol4/stage1/chunks_smart.json`
- `/Users/jeff/l0-knowledge-engine/output/mc/vol4/stage1/annotation_failures.json`

## Remaining Work

1. Wait for `vol2` Step5 rerun to finish and update `batch_progress.json` to `completed`.
2. Retry the `4` failed annotations for `vol4`.
3. Once `vol4` failures are cleared, mark `mc_vol4` as `completed`.

## Handoff Files

- Main handoff:
  - `/tmp/L0-systerm/docs/mc_stage1_handoff.md`
- Status snapshot:
  - `/tmp/L0-systerm/docs/mc_stage1_status_20260315.md`
