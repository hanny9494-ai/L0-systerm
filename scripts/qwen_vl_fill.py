#!/usr/bin/env python3
"""
Qwen-VL 补全脚本：识别 raw_merged.md 中的图片占位符，替换为表格/图片描述
流程：
  1. PDF → PNG（每页）
  2. 逐页 Qwen-VL 识别（带进度条）
  3. 将识别到的表格/图片描述替换 raw_merged.md 中的 ![](...) 占位符
  4. 输出 raw_merged_vl.md

用法: python3 qwen_vl_fill.py
断点续跑: 中途中断后重跑会跳过已识别的页
"""

import os, re, sys, json, time, base64, subprocess
from pathlib import Path

# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════
PDF_PATH    = Path("/Users/jeff/Documents/厨书数据库/工具科学书/Volume 2 - Techniques and Equipment.pdf")
RAW_MERGED  = Path("/Users/jeff/l0-knowledge-engine/output/mc/vol2/raw_merged.md")
WORK_DIR    = Path("/Users/jeff/l0-knowledge-engine/output/mc/vol2/qwen_vl_work")
PAGES_DIR   = WORK_DIR / "pages_png"
RESULTS_F   = WORK_DIR / "vl_results.json"
OUTPUT_MD   = Path("/Users/jeff/l0-knowledge-engine/output/mc/vol2/raw_merged_vl.md")

DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
VL_MODEL      = "qwen-vl-plus-latest"
DPI           = 72

PROMPT = """请完整提取这一页的所有内容，输出 Markdown 格式：
- 正文：直接输出文字，保留段落结构
- 表格：用标准 Markdown 表格（| 列1 | 列2 |），保留所有数值和单位
- 图表/示意图：用1-2句话描述类型和关键信息
- 操作步骤图/照片：简述内容
不要加页码或额外说明，直接输出内容。"""

# ══════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════
def log(msg):
    print(f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def progress_bar(done: int, total: int, width: int = 35) -> str:
    pct    = done / total if total else 0
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total} ({pct*100:.1f}%)"

def pdf_to_png(pdf_path: Path, pages_dir: Path, dpi: int = 150) -> list[Path]:
    """PDF每页转PNG，已存在则跳过"""
    try:
        import fitz
    except ImportError:
        log("安装 pymupdf...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pymupdf", "-q"], check=True)
        import fitz

    pages_dir.mkdir(parents=True, exist_ok=True)
    doc   = fitz.open(str(pdf_path))
    total = doc.page_count
    mat   = fitz.Matrix(dpi / 72, dpi / 72)

    existing = len(list(pages_dir.glob("page_*.png")))
    if existing == total:
        log(f"PNG已全部存在({total}页)，跳过转换")
        doc.close()
        return sorted(pages_dir.glob("page_*.png"))

    log(f"PDF转PNG: {total}页 (DPI={dpi})，已有{existing}页...")
    for i, page in enumerate(doc):
        img_path = pages_dir / f"page_{i+1:04d}.png"
        if not img_path.exists():
            pix = page.get_pixmap(matrix=mat)
            pix.save(str(img_path))
        if (i + 1) % 50 == 0:
            log(f"  转换进度: {i+1}/{total}")
    doc.close()
    log(f"PNG转换完成: {total}页")
    return sorted(pages_dir.glob("page_*.png"))

def qwen_vl_recognize(img_path: Path) -> dict:
    """调用Qwen-VL识别单页"""
    import urllib.request
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    payload = json.dumps({
        "model": VL_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT}
            ]
        }]
    }).encode()

    req = urllib.request.Request(
        DASHSCOPE_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {DASHSCOPE_KEY}",
            "Content-Type": "application/json",
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())

    content   = result["choices"][0]["message"]["content"]
    has_table = bool(re.search(r'^\|.+\|', content, re.MULTILINE))
    has_img   = bool(re.search(r'图|figure|photo|示意|diagram|照片|shows?|depicts?', content, re.IGNORECASE))
    tables    = re.findall(r'(?:^\|.+\n)+', content, re.MULTILINE)

    return {
        "content":     content,
        "has_table":   has_table,
        "has_image":   has_img,
        "table_count": len(tables),
        "char_count":  len(content),
    }

def run_vl_all_pages(pages: list[Path]) -> dict[int, dict]:
    """逐页识别，断点续跑，带实时进度条"""
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 加载已有结果
    if RESULTS_F.exists():
        done = json.loads(RESULTS_F.read_text(encoding="utf-8"))
        done = {int(k): v for k, v in done.items()}
        log(f"断点续跑: 已完成{len(done)}/{len(pages)}页")
    else:
        done = {}

    total      = len(pages)
    errors     = 0
    t_start    = time.time()
    page_times = []

    for i, img_path in enumerate(pages):
        page_num = i + 1
        if page_num in done:
            continue

        t_page = time.time()
        for attempt in range(3):
            try:
                result = qwen_vl_recognize(img_path)
                done[page_num] = result
                break
            except Exception as e:
                wait = (attempt + 1) * 3
                if attempt < 2:
                    time.sleep(wait)
                else:
                    done[page_num] = {
                        "content": "", "has_table": False,
                        "has_image": False, "table_count": 0,
                        "char_count": 0, "error": str(e)
                    }
                    errors += 1

        elapsed_page = time.time() - t_page
        page_times.append(elapsed_page)

        completed     = len(done)
        remaining     = total - completed
        avg_t         = sum(page_times[-20:]) / len(page_times[-20:])
        eta_sec       = remaining * avg_t
        eta_str       = f"{eta_sec/60:.1f}min" if eta_sec > 60 else f"{eta_sec:.0f}s"
        tables_so_far = sum(1 for r in done.values() if r.get("has_table"))
        has_table_tag = " 📊" if done[page_num].get("has_table") else ""

        bar = progress_bar(completed, total)
        print(
            f"\r{bar}  ETA:{eta_str}  表格:{tables_so_far}  错误:{errors}"
            f"  [{elapsed_page:.1f}s{has_table_tag}]     ",
            end="", flush=True
        )

        # 每20页保存 + 打印checkpoint
        if completed % 20 == 0 or completed == total:
            RESULTS_F.write_text(
                json.dumps(done, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print()  # 换行
            elapsed_total = time.time() - t_start
            log(f"  ✓ {completed}/{total}页  已用:{elapsed_total/60:.1f}min  "
                f"表格页:{tables_so_far}  错误:{errors}")

    print()
    return done

# ══════════════════════════════════════════════════════════
# 合并
# ══════════════════════════════════════════════════════════
def extract_tables(content: str) -> list[str]:
    tables = re.findall(r'(?:^\|.+\n)+', content, re.MULTILINE)
    return [t.strip() for t in tables if len(t.strip().split('\n')) >= 2]

def extract_img_descs(content: str) -> list[str]:
    clean = re.sub(r'(?:^\|.+\n)+', '', content, flags=re.MULTILINE)
    descs = []
    for para in re.split(r'\n{2,}', clean):
        para = para.strip()
        if para and re.search(r'图|figure|photo|示意|diagram|照片|shows?|depicts?', para, re.IGNORECASE):
            descs.append(para)
    return descs

def merge_vl_into_md(md_text: str, vl_results: dict[int, dict]) -> tuple[str, int, int]:
    table_pool = []
    img_pool   = []
    for page_num in sorted(vl_results.keys()):
        r = vl_results[page_num]
        content = r.get("content", "")
        if not content:
            continue
        for t in extract_tables(content):
            table_pool.append({"page": page_num, "content": t})
        for d in extract_img_descs(content):
            img_pool.append({"page": page_num, "content": d})

    log(f"Qwen-VL内容池: 表格{len(table_pool)}个, 图片描述{len(img_pool)}个")

    t_idx = i_idx = replaced = 0

    def replace_one(m):
        nonlocal t_idx, i_idx, replaced
        if t_idx < len(table_pool):
            repl = f"\n\n{table_pool[t_idx]['content']}\n\n"
            t_idx += 1
            replaced += 1
            return repl
        elif i_idx < len(img_pool):
            repl = f"\n\n> 📷 {img_pool[i_idx]['content']}\n\n"
            i_idx += 1
            replaced += 1
            return repl
        return m.group(0)

    merged   = re.sub(r'!\[.*?\]\([^)]*\)', replace_one, md_text)
    residual = len(re.findall(r'!\[.*?\]\([^)]*\)', merged))
    return merged, replaced, residual

# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def main():
    if not DASHSCOPE_KEY:
        sys.exit("❌ 未设置 DASHSCOPE_API_KEY")
    if not PDF_PATH.exists():
        sys.exit(f"❌ PDF不存在: {PDF_PATH}")
    if not RAW_MERGED.exists():
        sys.exit(f"❌ raw_merged.md不存在: {RAW_MERGED}")

    log("=" * 60)
    log(f"Qwen-VL 补全 raw_merged.md  模型: {VL_MODEL}")
    log("=" * 60)
    t0 = time.time()

    md_text      = RAW_MERGED.read_text(encoding="utf-8")
    placeholders = len(re.findall(r'!\[.*?\]\([^)]*\)', md_text))
    log(f"raw_merged.md: {len(md_text):,}字符，{placeholders}个图片占位符")
    log(f"预计耗时: ~{488 * 4 / 60:.0f}分钟 (488页 × ~4s/页)")

    # Step1: PDF → PNG
    log("\n[Step 1] PDF转PNG...")
    pages = pdf_to_png(PDF_PATH, PAGES_DIR, dpi=DPI)
    log(f"共{len(pages)}页PNG")

    # Step2: 识别
    log(f"\n[Step 2] {VL_MODEL} 逐页识别...")
    vl_results = run_vl_all_pages(pages)

    total_pages  = len(vl_results)
    table_pages  = sum(1 for r in vl_results.values() if r.get("has_table"))
    error_pages  = sum(1 for r in vl_results.values() if r.get("error"))
    log(f"识别完成: {total_pages}页 | 含表格:{table_pages}页 | 错误:{error_pages}页")

    # Step3: 合并
    log("\n[Step 3] 合并到MD...")
    merged_text, replaced, residual = merge_vl_into_md(md_text, vl_results)
    merged_text = re.sub(r'\n{4,}', '\n\n\n', merged_text)
    OUTPUT_MD.write_text(merged_text, encoding="utf-8")

    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"✅ 完成！")
    log(f"  输出: {OUTPUT_MD}")
    log(f"  文件大小: {OUTPUT_MD.stat().st_size//1024}KB")
    log(f"  替换占位符: {replaced}/{placeholders}")
    log(f"  残留占位符: {residual}")
    log(f"  含表格页: {table_pages}/{total_pages} ({table_pages/total_pages*100:.0f}%)")
    log(f"  总耗时: {elapsed/60:.1f}分钟")
    log(f"\n下一步:")
    log(f"  cp {OUTPUT_MD} {RAW_MERGED}")
    log(f"  python3 stage1_mc_vol2.py --skip-extract")

if __name__ == "__main__":
    main()
