#!/usr/bin/env python3
"""
MinerU API 客户端
=================
流程: 本地 PDF → 请求上传 URL → 上传文件 → 轮询进度 → 下载 ZIP → 解压 MD

用法:
  # 解析单个 PDF
  python3 mineru_api.py --pdf /path/to/book.pdf --out /path/to/output/

  # 解析并直接输出 MD 文本到 stdout（用于管道）
  python3 mineru_api.py --pdf book.pdf --stdout

环境变量:
  MINERU_API_KEY  你的 Bearer token
"""

import os
import sys
import time
import zipfile
import requests
import argparse
from pathlib import Path
from io import BytesIO

# ──────────────────────────────────────────────
BASE_URL = "https://mineru.net/api/v4"
POLL_INTERVAL = 5   # 秒
MAX_WAIT      = 600  # 最多等 10 分钟


def get_token() -> str:
    token = os.environ.get("MINERU_API_KEY", "")
    if not token:
        sys.exit("❌ 未设置 MINERU_API_KEY，请先: export MINERU_API_KEY=你的key")
    return token


def headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ──────────────────────────────────────────────
# 步骤 1: 请求预签名上传 URL
# ──────────────────────────────────────────────
def request_upload_url(token: str, filename: str) -> tuple[str, str]:
    """
    返回 (upload_url, task_id)
    文档: POST /file-urls/batch
    """
    resp = requests.post(
        f"{BASE_URL}/file-urls/batch",
        headers=headers(token),
        json={
            "enable_formula": True,
            "enable_table": True,
            "language": "en",
            "files": [{"name": filename, "is_ocr": False, "data_id": filename}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 0:
        sys.exit(f"❌ 请求上传 URL 失败: {data}")

    file_info = data["data"]["files"][0]
    batch_id  = data["data"]["batch_id"]
    upload_url = file_info["url"]

    print(f"  ✅ 获取上传 URL  batch_id={batch_id}")
    return upload_url, batch_id


# ──────────────────────────────────────────────
# 步骤 2: 上传文件
# ──────────────────────────────────────────────
def upload_file(upload_url: str, pdf_path: Path):
    """直接 PUT 到预签名 URL，不带 Content-Type"""
    size_mb = pdf_path.stat().st_size / 1e6
    print(f"  📤 上传 {pdf_path.name}  ({size_mb:.1f} MB)...")
    with open(pdf_path, "rb") as f:
        resp = requests.put(upload_url, data=f, timeout=300)
    resp.raise_for_status()
    print(f"  ✅ 上传完成  HTTP {resp.status_code}")


# ──────────────────────────────────────────────
# 步骤 3: 轮询批次进度
# ──────────────────────────────────────────────
def poll_batch(token: str, batch_id: str) -> str:
    """
    轮询直到 state=done，返回 full_zip_url。
    文档: GET /extract/batch/result/{batch_id}
    """
    print(f"  ⏳ 等待解析完成 (batch_id={batch_id})...")
    url = f"{BASE_URL}/extract/batch/result/{batch_id}"
    start = time.time()

    while time.time() - start < MAX_WAIT:
        resp = requests.get(url, headers=headers(token), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            sys.exit(f"❌ 查询失败: {data}")

        results = data["data"]["extract_result"]
        file_res = results[0]
        state    = file_res["state"]

        if state == "done":
            zip_url = file_res["full_zip_url"]
            print(f"  ✅ 解析完成！")
            return zip_url
        elif state == "failed":
            sys.exit(f"❌ 解析失败: {file_res.get('err_msg')}")
        else:
            progress = file_res.get("extract_progress", {})
            done_pg  = progress.get("extracted_pages", "?")
            total_pg = progress.get("total_pages", "?")
            print(f"     {state}  {done_pg}/{total_pg} 页  ({int(time.time()-start)}s 已过)", end="\r")
            time.sleep(POLL_INTERVAL)

    sys.exit(f"❌ 超时（>{MAX_WAIT}s），请检查 MinerU 控制台")


# ──────────────────────────────────────────────
# 步骤 4: 下载 ZIP，提取 MD
# ──────────────────────────────────────────────
def download_and_extract(zip_url: str, out_dir: Path, pdf_stem: str) -> Path:
    """
    下载 ZIP，把 .md 文件保存到 out_dir/{pdf_stem}.md
    ZIP 内通常包含: {name}.md, {name}_content_list.json, images/
    """
    print(f"  📥 下载结果 ZIP...")
    resp = requests.get(zip_url, timeout=120)
    resp.raise_for_status()

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = None

    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        print(f"  📦 ZIP 内容: {zf.namelist()[:10]}")
        for name in zf.namelist():
            if name.endswith(".md"):
                # 统一保存为 {pdf_stem}.md
                md_path = out_dir / f"{pdf_stem}.md"
                md_path.write_bytes(zf.read(name))
                print(f"  ✅ MD 已保存: {md_path}  ({md_path.stat().st_size/1e3:.0f} KB)")
            elif name.startswith("images/") or name.endswith((".png",".jpg")):
                img_out = out_dir / "images" / Path(name).name
                img_out.parent.mkdir(exist_ok=True)
                img_out.write_bytes(zf.read(name))
            elif name.endswith(".json"):
                json_out = out_dir / f"{pdf_stem}_content_list.json"
                json_out.write_bytes(zf.read(name))

    if not md_path:
        sys.exit("❌ ZIP 中未找到 .md 文件")

    return md_path


# ──────────────────────────────────────────────
# 完整流程
# ──────────────────────────────────────────────
def parse_pdf(pdf_path: Path, out_dir: Path, stdout: bool = False) -> Path:
    token = get_token()

    print(f"\n{'='*52}")
    print(f"  MinerU API 解析: {pdf_path.name}")
    print(f"{'='*52}")

    # 1. 请求上传 URL
    upload_url, batch_id = request_upload_url(token, pdf_path.name)

    # 2. 上传
    upload_file(upload_url, pdf_path)

    # 3. 轮询
    zip_url = poll_batch(token, batch_id)

    # 4. 下载解压
    md_path = download_and_extract(zip_url, out_dir, pdf_path.stem)

    if stdout:
        print("\n" + "─"*52)
        print(md_path.read_text(encoding="utf-8"))
    else:
        print(f"\n✅ 完成！MD 路径: {md_path}")

    return md_path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MinerU API — PDF → Markdown")
    ap.add_argument("--pdf",    required=True,  help="输入 PDF 路径")
    ap.add_argument("--out",    default="./mineru_output", help="输出目录")
    ap.add_argument("--stdout", action="store_true",      help="同时打印 MD 到 stdout")
    args = ap.parse_args()

    parse_pdf(
        pdf_path=Path(args.pdf),
        out_dir=Path(args.out),
        stdout=args.stdout,
    )
