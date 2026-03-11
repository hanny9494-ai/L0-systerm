import json
from pathlib import Path
from datetime import datetime

f = Path("./output/stage1/progress_smart.json")
if f.exists():
    d = json.load(open(f))
    completed = d.get("completed_chapters", [])
    chunks = d.get("chunks", [])
    last = d.get("last_update", "N/A")
    pct = len(completed) / 23 * 100
    print(f"[{datetime.now().strftime('%H:%M:%S')}]")
    print(f"完成章节: {len(completed)} / 23  ({pct:.1f}%)")
    print(f"Chunks:   {len(chunks)}")
    print(f"最后更新: {last}")
    if completed:
        print(f"已完成:   {completed}")
else:
    print("⏳ 进度文件不存在，任务可能还未写入")
