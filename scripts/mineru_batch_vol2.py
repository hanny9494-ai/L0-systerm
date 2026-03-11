#!/usr/bin/env python3
"""
MinerU 分批上传 Vol2 三个分片，合并为 raw_merged.md
用法: python3 mineru_batch_vol2.py
"""

import os, sys, time, zipfile, requests
from pathlib import Path
from io import BytesIO

BASE_URL      = "https://mineru.net/api/v4"
POLL_INTERVAL = 5
MAX_WAIT      = 1800  # 30分钟

TOKEN = os.environ.get("MINERU_API_KEY", "")
if not TOKEN:
    sys.exit("❌ 未设置 MINERU_API_KEY")

PARTS = [
    Path("/Users/jeff/Documents/厨书数据库/工具科学书/vol2_part1.pdf"),
    Path("/Users/jeff/Documents/厨书数据库/工具科学书/vol2_part2.pdf"),
    Path("/Users/jeff/Documents/厨书数据库/工具科学书/vol2_part3.pdf"),
]

OUT_DIR   = Path("/Users/jeff/l0-knowledge-engine/output/mc/vol2/mineru_parts")
MERGED_MD = Path("/Users/jeff/l0-knowledge-engine/output/mc/vol2/raw_merged.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def auth_headers():
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def request_upload_url(filename: str) -> tuple[str, str]:
    """返回 (upload_url, batch_id)"""
    resp = requests.post(
        f"{BASE_URL}/file-urls/batch",
        headers=auth_headers(),
        json={
            "enable_formula": True,
            "enable_table":   True,
            "language":       "en",
            "files": [{"name": filename, "is_ocr": False, "data_id": filename}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        sys.exit(f"❌ 请求上传URL失败: {data}")
    batch_id   = data["data"]["batch_id"]
    upload_url = data["data"]["file_urls"][0]
    print(f"  ✅ batch_id={batch_id}")
    return upload_url, batch_id

def upload_file(upload_url: str, pdf_path: Path):
    print(f"  📤 上传 {pdf_path.name} ({pdf_path.stat().st_size/1e6:.0f}MB)...")
    with open(pdf_path, "rb") as f:
        resp = requests.put(upload_url, data=f, timeout=600)
    resp.raise_for_status()
    print(f"  ✅ 上传完成 HTTP {resp.status_code}")

def poll_batch(batch_id: str, label: str) -> str:
    """
    正确endpoint: GET /api/v4/extract-results/batch/{batch_id}
    响应结构: data.extract_result[0].state / full_zip_url
    """
    url   = f"{BASE_URL}/extract-results/batch/{batch_id}"
    start = time.time()
    print(f"  ⏳ 等待MinerU处理 {label}...")
    while time.time() - start < MAX_WAIT:
        resp = requests.get(url, headers=auth_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            # 任务还没注册好，继续等
            time.sleep(POLL_INTERVAL)
            continue

        file_res = data["data"]["extract_result"][0]
        state    = file_res.get("state", "")

        if state == "done":
            zip_url = file_res.get("full_zip_url", "")
            if zip_url:
                print(f"\n  ✅ {label} 处理完成")
                return zip_url
        elif state == "failed":
            sys.exit(f"❌ {label} 失败: {file_res.get('err_msg')}")
        else:
            prog    = file_res.get("extract_progress", {})
            elapsed = int(time.time() - start)
            print(
                f"     {state}  "
                f"{prog.get('extracted_pages','?')}/{prog.get('total_pages','?')}页  "
                f"({elapsed}s)",
                end="\r"
            )
        time.sleep(POLL_INTERVAL)

    sys.exit(f"❌ {label} 超时 (>{MAX_WAIT}s)")

def download_md(zip_url: str, out_dir: Path, stem: str) -> Path:
    print(f"  📥 下载结果ZIP...")
    resp = requests.get(zip_url, timeout=300)
    resp.raise_for_status()
    md_path = None
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".md"):
                md_path = out_dir / f"{stem}.md"
                md_path.write_bytes(zf.read(name))
                print(f"  ✅ 保存: {md_path.name} ({md_path.stat().st_size//1024}KB)")
    if not md_path:
        sys.exit("❌ ZIP中无.md文件")
    return md_path

def process_part(pdf_path: Path, idx: int) -> Path:
    label   = f"Part{idx+1} ({pdf_path.name})"
    out_dir = OUT_DIR / f"part{idx+1}"
    out_dir.mkdir(exist_ok=True)
    md_cache = out_dir / f"{pdf_path.stem}.md"

    if md_cache.exists():
        print(f"\n[{label}] 缓存命中 → {md_cache.name} ({md_cache.stat().st_size//1024}KB)")
        return md_cache

    print(f"\n{'='*55}")
    print(f"  处理 {label}")
    print(f"{'='*55}")

    upload_url, batch_id = request_upload_url(pdf_path.name)
    upload_file(upload_url, pdf_path)
    zip_url = poll_batch(batch_id, label)
    md_path = download_md(zip_url, out_dir, pdf_path.stem)
    return md_path

def main():
    print("=" * 55)
    print("MinerU 分批上传 Vol2 (3份)")
    print("=" * 55)

    md_files = []
    for i, part in enumerate(PARTS):
        if not part.exists():
            sys.exit(f"❌ 文件不存在: {part}")
        md = process_part(part, i)
        md_files.append(md)

    # 合并三份MD
    print(f"\n{'='*55}")
    print(f"合并 {len(md_files)} 份MD → raw_merged.md")
    MERGED_MD.parent.mkdir(parents=True, exist_ok=True)
    combined = []
    for i, md in enumerate(md_files):
        text = md.read_text(encoding="utf-8").strip()
        combined.append(f"\n\n<!-- ===== Part {i+1}: {md.name} ===== -->\n\n{text}")

    MERGED_MD.write_text("\n".join(combined), encoding="utf-8")
    size_kb = MERGED_MD.stat().st_size // 1024
    print(f"✅ 合并完成: raw_merged.md ({size_kb}KB)")
    print(f"\n下一步:")
    print(f"  python3 stage1_mc_vol2.py --skip-extract")

if __name__ == "__main__":
    main()
