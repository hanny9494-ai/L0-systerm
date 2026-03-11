#!/usr/bin/env python3
"""
MC Stage 1 — Vol 2 Pipeline
PDF提取 → 智能切分 → 标注 → chunks_smart.json

用法:
  python3 stage1_mc_vol2.py                              # 完整三步
  python3 stage1_mc_vol2.py --skip-extract               # 跳过Step1（已有raw_merged.md）
  python3 stage1_mc_vol2.py --skip-extract --skip-split  # 只跑标注
  python3 stage1_mc_vol2.py --only-report                # 查看已有结果统计

依赖:
  pip install requests pymupdf openai
"""

import os, re, sys, json, time, shutil, argparse, subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════
PDF_PATH   = Path("/Users/jeff/Documents/厨书数据库/工具科学书/Volume 2 - Techniques and Equipment.pdf")
BASE_OUT   = Path("/Users/jeff/l0-knowledge-engine/output/mc/vol2")
WORK_DIR   = BASE_OUT / "extract_work"
RAW_MERGED = BASE_OUT / "raw_merged.md"
STAGE1_DIR = BASE_OUT / "stage1"
CHUNKS_RAW   = STAGE1_DIR / "chunks_raw.json"
CHUNKS_SMART = STAGE1_DIR / "chunks_smart.json"
PROGRESS_F   = STAGE1_DIR / "annotate_progress.json"

# 原始工具脚本
ENGINE_DIR   = Path("/Users/jeff/l0-knowledge-engine")
MINERU_PY    = ENGINE_DIR / "mineru_api.py"
QWEN_VL_PY   = ENGINE_DIR / "qwen_vision_compare.py"
MERGE_PY     = ENGINE_DIR / "merge_mineru_qwen.py"

# Ollama
OLLAMA_URL     = "http://localhost:11434/api/generate"
ANNOTATE_MODEL = "qwen2.5:7b"   # 按 `ollama list` 实际填写

# DashScope（qwen_vision_compare.py 需要）
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

VALID_TOPICS = [
    "heat_transfer", "chemical_reaction", "physical_change", "water_activity",
    "protein_science", "lipid_science", "carbohydrate", "enzyme", "flavor_sensory",
    "fermentation", "food_safety", "emulsion_colloid", "color_pigment", "equipment_physics"
]
SOURCE_BOOK = "modernist_cuisine_vol2"

# merge_mineru_qwen.py 里的三行硬编码路径（原文完全匹配）
MERGE_MINERU_LINE = "MINERU_MD  = Path('/Users/jeff/l0-knowledge-engine/output/comparison/mineru/full.md')"
MERGE_JSON_LINE   = "RAW_JSON   = Path('/Users/jeff/l0-knowledge-engine/output/comparison/raw_results.json')"
MERGE_OUT_LINE    = "OUTPUT_MD  = Path('/Users/jeff/l0-knowledge-engine/output/comparison/ch13_merged.md')"

# qwen_vision_compare.py 里的两行硬编码路径
QWEN_PAGES_LINE  = "PAGES_DIR  = Path('/Users/jeff/l0-knowledge-engine/output/comparison/pages')"
QWEN_MINERU_LINE = "MINERU_MD  = Path('/Users/jeff/l0-knowledge-engine/output/comparison/mineru/full.md')"
QWEN_OUTPUT_LINE = "OUTPUT_DIR = Path('/Users/jeff/l0-knowledge-engine/output/comparison')"


# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════
def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

def ensure_dirs():
    for d in [STAGE1_DIR, BASE_OUT, WORK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def safe_json(text: str):
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1).strip()
    for pat in [r"\{[\s\S]+\}", r"\[[\s\S]+\]"]:
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}

def run_patched(script: Path, replacements: dict) -> subprocess.CompletedProcess:
    """替换脚本中的硬编码路径行，写临时文件运行"""
    src = script.read_text(encoding="utf-8")
    for old, new in replacements.items():
        if old in src:
            src = src.replace(old, new)
        else:
            log(f"  ⚠ 替换目标未找到（脚本可能已更新）: {old[:70]}", "WARN")
    tmp = script.parent / f"_tmp_{script.name}"
    tmp.write_text(src, encoding="utf-8")
    env = os.environ.copy()
    try:
        r = subprocess.run(
            [sys.executable, str(tmp)],
            capture_output=True, text=True, env=env
        )
    finally:
        tmp.unlink(missing_ok=True)
    return r

def pdf_to_pages_png(pdf_path: Path, out_dir: Path, dpi: int = 150) -> Path:
    """将PDF转为PNG页面目录（qwen_vision_compare.py需要PAGES_DIR）"""
    try:
        import fitz
    except ImportError:
        log("  安装 pymupdf...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pymupdf", "-q"], check=True)
        import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("p*.png"))

    doc = fitz.open(str(pdf_path))
    total = doc.page_count

    if len(existing) == total:
        log(f"  PNG已存在({total}页)，跳过转换")
        doc.close()
        return out_dir

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    log(f"  PDF转PNG: {total}页 (DPI={dpi})...")
    for i, page in enumerate(doc):
        img_path = out_dir / f"p{i+1:04d}.png"
        if not img_path.exists():
            pix = page.get_pixmap(matrix=mat)
            pix.save(str(img_path))
        if (i + 1) % 100 == 0:
            log(f"  转换进度: {i+1}/{total}")
    doc.close()
    log(f"  PNG转换完成: {total}页 → {out_dir}")
    return out_dir

def ollama_call(model: str, prompt: str, timeout: int = 90) -> str:
    """调用本地Ollama，qwen3系列自动关闭thinking"""
    payload = {"model": model, "prompt": prompt, "stream": False}
    if "qwen3" in model.lower():
        payload["options"] = {"think": False}
    result = subprocess.run(
        ["curl", "-s", "--max-time", str(timeout),
         "-X", "POST", OLLAMA_URL,
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=timeout + 5
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"Ollama无响应: {result.stderr[:200]}")
    return json.loads(result.stdout).get("response", "")

def ollama_retry(model: str, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            return ollama_call(model, prompt)
        except Exception as e:
            wait = (attempt + 1) * 5
            if attempt < retries - 1:
                log(f"  Ollama重试 {attempt+1}/{retries}，等{wait}s: {e}", "WARN")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Ollama {model} 连续{retries}次失败: {e}")


# ══════════════════════════════════════════════════════════
# Step 1: PDF提取（调用 mineru_api.py + qwen_vision_compare.py + merge_mineru_qwen.py）
# ══════════════════════════════════════════════════════════
def step1_extract():
    log("=" * 60)
    log("Step 1: PDF提取 (MinerU + Qwen-VL + 合并)")
    log("=" * 60)
    t0 = time.time()

    if not PDF_PATH.exists():
        log(f"PDF不存在: {PDF_PATH}", "ERROR")
        sys.exit(1)

    # 工作目录布局（对应原始脚本期望的路径结构）
    mineru_out_dir = WORK_DIR / "mineru"   # mineru_api.py --out 指向这里
    pages_dir      = WORK_DIR / "pages"    # qwen_vision_compare.py 读取PNG的目录
    raw_json       = WORK_DIR / "raw_results.json"

    mineru_out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1A: MinerU ──────────────────────────────────────
    # mineru_api.py 接口: --pdf PDF_PATH --out OUT_DIR
    # 输出: OUT_DIR/{pdf_stem}.md
    expected_md = mineru_out_dir / f"{PDF_PATH.stem}.md"

    if expected_md.exists():
        log(f"  MinerU缓存命中: {expected_md.name} ({expected_md.stat().st_size//1024}KB)")
        mineru_md = expected_md
    else:
        log(f"  运行 mineru_api.py ...")
        r = subprocess.run(
            [sys.executable, str(MINERU_PY),
             "--pdf", str(PDF_PATH),
             "--out", str(mineru_out_dir)],
            capture_output=False,  # 让MinerU的进度直接打印到终端
            text=True
        )
        if r.returncode != 0:
            log("MinerU失败", "ERROR")
            sys.exit(1)
        # 找实际输出的.md（名字跟pdf stem一致）
        md_files = sorted(mineru_out_dir.rglob("*.md"))
        if not md_files:
            log("MinerU未生成.md文件", "ERROR")
            sys.exit(1)
        mineru_md = max(md_files, key=lambda f: f.stat().st_size)
        log(f"  MinerU完成: {mineru_md.name} ({mineru_md.stat().st_size//1024}KB)")

    # ── 1B: PDF → PNG（qwen_vision_compare.py需要PNG目录）──
    if not DASHSCOPE_KEY:
        log("  ⚠ DASHSCOPE_API_KEY未设置，跳过Qwen-VL视觉识别", "WARN")
        qwen_ok = False
    else:
        pages_dir = pdf_to_pages_png(PDF_PATH, pages_dir, dpi=150)

        # qwen_vision_compare.py 硬编码了 PAGES_DIR / MINERU_MD / OUTPUT_DIR
        # 用 run_patched 替换这三个路径变量
        log(f"  运行 qwen_vision_compare.py ({PDF_PATH.stem})...")
        r = run_patched(QWEN_VL_PY, {
            QWEN_PAGES_LINE:  f"PAGES_DIR  = Path('{pages_dir}')",
            QWEN_MINERU_LINE: f"MINERU_MD  = Path('{mineru_md}')",
            QWEN_OUTPUT_LINE: f"OUTPUT_DIR = Path('{WORK_DIR}')",
        })
        # qwen脚本直接打印进度到stdout，转发出来
        if r.stdout:
            for line in r.stdout.splitlines():
                log(f"  [qwen] {line}")
        if r.returncode != 0:
            log(f"  Qwen-VL失败: {r.stderr[:300]}", "WARN")
            log("  继续使用纯MinerU输出")
            qwen_ok = False
        else:
            # qwen脚本输出 raw_results.json 到 OUTPUT_DIR
            qwen_ok = raw_json.exists()
            if qwen_ok:
                log(f"  Qwen-VL完成: {raw_json}")

    # ── 1C: 合并 ────────────────────────────────────────
    # merge_mineru_qwen.py 硬编码了 MINERU_MD / RAW_JSON / OUTPUT_MD
    if qwen_ok:
        log("  运行 merge_mineru_qwen.py ...")
        r = run_patched(MERGE_PY, {
            MERGE_MINERU_LINE: f"MINERU_MD  = Path('{mineru_md}')",
            MERGE_JSON_LINE:   f"RAW_JSON   = Path('{raw_json}')",
            MERGE_OUT_LINE:    f"OUTPUT_MD  = Path('{RAW_MERGED}')",
        })
        if r.stdout:
            for line in r.stdout.splitlines():
                log(f"  [merge] {line}")
        if r.returncode != 0:
            log(f"  Merge失败，fallback到纯MinerU: {r.stderr[:200]}", "WARN")
            shutil.copy(mineru_md, RAW_MERGED)
        else:
            log("  Merge完成")
    else:
        log("  使用纯MinerU输出（无Qwen-VL合并）")
        shutil.copy(mineru_md, RAW_MERGED)

    # ── 统计视觉内容占比 ──
    visual_stats = {"visual_pct": 0, "table_pages": 0, "img_pages": 0, "total_pages": 0}
    if raw_json.exists():
        try:
            raw_data = json.loads(raw_json.read_text(encoding="utf-8"))
            qwen_results = [r for r in raw_data.get("qwen", []) if "error" not in r]
            total_pages  = len(qwen_results)
            table_pages  = sum(1 for r in qwen_results if r.get("stats", {}).get("has_table"))
            img_pages    = sum(1 for r in qwen_results if r.get("stats", {}).get("img_refs", 0) > 0)
            visual_pct   = (table_pages + img_pages) / total_pages * 100 if total_pages else 0
            visual_stats = {"visual_pct": visual_pct, "table_pages": table_pages,
                            "img_pages": img_pages, "total_pages": total_pages}
        except Exception:
            pass

    size_kb = RAW_MERGED.stat().st_size / 1024
    log(f"Step 1完成 [{time.time()-t0:.0f}s]")
    log(f"  输出: raw_merged.md ({size_kb:.0f}KB)")
    if visual_stats["total_pages"]:
        log(f"  视觉: 表格页{visual_stats['table_pages']} + "
            f"图片页{visual_stats['img_pages']} / "
            f"{visual_stats['total_pages']}页 = {visual_stats['visual_pct']:.0f}%")
    return visual_stats


# ══════════════════════════════════════════════════════════
# Step 2: 智能切分
# ══════════════════════════════════════════════════════════
def detect_chapter(text: str) -> int | None:
    for p in [r'\bChapter\s+(\d+)\b', r'\bCHAPTER\s+(\d+)\b', r'^#{1,2}\s+(\d+)[.\s]']:
        m = re.search(p, text[:600], re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None

def rule_split(text: str, target_words: int = 400) -> list[str]:
    """按标题粗切 + 段落细切，目标约400词/chunk"""
    sections = re.split(r'\n(?=#{1,3} )', text)
    chunks = []
    for section in sections:
        if len(section.strip()) < 60:
            continue
        if len(section.split()) <= target_words * 1.4:
            chunks.append(section.strip())
        else:
            paras = section.split('\n\n')
            buf = ""
            for para in paras:
                candidate = (buf + "\n\n" + para).strip() if buf else para.strip()
                if len(candidate.split()) > target_words and buf:
                    chunks.append(buf.strip())
                    buf = para.strip()
                else:
                    buf = candidate
            if buf.strip():
                chunks.append(buf.strip())
    return [c for c in chunks if len(c.split()) >= 25]

def step2_split():
    log("=" * 60)
    log("Step 2: 智能切分")
    log("=" * 60)
    t0 = time.time()

    if not RAW_MERGED.exists():
        log(f"raw_merged.md不存在: {RAW_MERGED}", "ERROR")
        sys.exit(1)

    text = RAW_MERGED.read_text(encoding="utf-8")
    log(f"输入: {len(text):,}字符")

    raw_chunks = rule_split(text, target_words=400)
    log(f"切分完成: {len(raw_chunks)} chunks")

    chunks_out = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks_out.append({
            "chunk_idx":   i,
            "full_text":   chunk_text,
            "chapter_num": detect_chapter(chunk_text),
            "source_book": SOURCE_BOOK,
            "char_count":  len(chunk_text),
            "word_count":  len(chunk_text.split()),
        })

    avg_words = sum(c["word_count"] for c in chunks_out) // len(chunks_out) if chunks_out else 0
    result = {
        "chunks":       chunks_out,
        "total_chunks": len(chunks_out),
        "split_model":  "rule-based",
        "avg_words":    avg_words,
        "source":       str(RAW_MERGED),
        "created_at":   datetime.now().isoformat(),
    }
    CHUNKS_RAW.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Step 2完成 [{time.time()-t0:.1f}s]")
    log(f"  总chunk数: {len(chunks_out)}  (OFC基准: 1427)")
    log(f"  平均词数: {avg_words}")
    return len(chunks_out)


# ══════════════════════════════════════════════════════════
# Step 3: 标注
# ══════════════════════════════════════════════════════════
ANNOTATE_PROMPT = """你是烹饪科学专家。分析以下文本块，返回JSON标注。

文本:
{text}

只返回JSON，不要任何其他内容:
{{
  "topics": ["从合法列表选1-3个"],
  "summary": "50字以内中文摘要，点明核心科学内容"
}}

合法topics（只能从这14个中选）:
heat_transfer, chemical_reaction, physical_change, water_activity,
protein_science, lipid_science, carbohydrate, enzyme, flavor_sensory,
fermentation, food_safety, emulsion_colloid, color_pigment, equipment_physics"""

def annotate_one(chunk: dict) -> dict:
    prompt = ANNOTATE_PROMPT.format(text=chunk["full_text"][:900])
    try:
        resp   = ollama_retry(ANNOTATE_MODEL, prompt)
        parsed = safe_json(resp)
        raw_topics = parsed.get("topics", [])
        if isinstance(raw_topics, str):
            raw_topics = [raw_topics]
        valid_topics = [t for t in raw_topics if t in VALID_TOPICS] or ["physical_change"]
        summary = str(parsed.get("summary", ""))[:100]
        return {**chunk, "topics": valid_topics, "summary": summary}
    except Exception as e:
        return {**chunk, "topics": ["physical_change"], "summary": "", "_error": str(e)}

def step3_annotate():
    log("=" * 60)
    log("Step 3: 标注 (topics + summary)")
    log("=" * 60)
    t0 = time.time()

    if not CHUNKS_RAW.exists():
        log(f"chunks_raw.json不存在: {CHUNKS_RAW}", "ERROR")
        sys.exit(1)

    data   = json.loads(CHUNKS_RAW.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    total  = len(chunks)
    log(f"待标注: {total} chunks  模型: {ANNOTATE_MODEL}")
    log(f"预计耗时: ~{total * 4.5 / 60:.0f}分钟")

    # 断点续跑
    if PROGRESS_F.exists():
        done_list = json.loads(PROGRESS_F.read_text(encoding="utf-8"))
        done_map  = {c["chunk_idx"]: c for c in done_list}
        log(f"断点续跑: 已完成 {len(done_map)}/{total}，剩余{total - len(done_map)}")
    else:
        done_map = {}

    annotated = list(done_map.values())
    errors    = 0

    for i, chunk in enumerate(chunks):
        if chunk["chunk_idx"] in done_map:
            continue

        result = annotate_one(chunk)
        annotated.append(result)
        if "_error" in result:
            errors += 1

        done_so_far = len(annotated)
        if done_so_far % 50 == 0 or i == total - 1:
            PROGRESS_F.write_text(json.dumps(annotated, ensure_ascii=False), encoding="utf-8")
            elapsed  = time.time() - t0
            new_done = done_so_far - len(done_map)
            speed    = new_done / elapsed if elapsed > 0 else 1
            eta      = (total - done_so_far) / speed if speed > 0 else 0
            log(f"  进度: {done_so_far}/{total} | 错误: {errors} | "
                f"速度: {speed:.1f}/s | ETA: {eta/60:.1f}min")

    # Topics统计
    topic_counter = Counter()
    for c in annotated:
        for t in c.get("topics", []):
            topic_counter[t] += 1

    final = {
        "chunks":               sorted(annotated, key=lambda x: x["chunk_idx"]),
        "total_chunks":         len(annotated),
        "split_model":          data.get("split_model", "rule-based"),
        "annotate_model":       ANNOTATE_MODEL,
        "topics_distribution":  dict(topic_counter.most_common()),
        "annotate_errors":      errors,
        "source_book":          SOURCE_BOOK,
        "created_at":           datetime.now().isoformat(),
        "elapsed_seconds":      round(time.time() - t0, 1),
    }
    CHUNKS_SMART.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    PROGRESS_F.unlink(missing_ok=True)

    log(f"Step 3完成 [{time.time()-t0:.0f}s]")
    log(f"  总chunk数: {len(annotated)}  (OFC基准: 1427)")
    log(f"  标注错误: {errors}")
    log(f"\n  Topics分布:")
    max_v = max(topic_counter.values()) if topic_counter else 1
    for topic, count in topic_counter.most_common():
        bar = "█" * max(1, count * 28 // max_v)
        log(f"    {topic:<25} {count:4d}  {bar}")

    return len(annotated), dict(topic_counter)


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="MC Vol2 Stage1 Pipeline")
    parser.add_argument("--skip-extract",  action="store_true", help="跳过Step1（已有raw_merged.md）")
    parser.add_argument("--skip-split",    action="store_true", help="跳过Step2（已有chunks_raw.json）")
    parser.add_argument("--skip-annotate", action="store_true", help="跳过Step3")
    parser.add_argument("--only-report",   action="store_true", help="只打印已有结果统计")
    args = parser.parse_args()

    ensure_dirs()
    t_start = time.time()

    log("=" * 60)
    log("MC Stage 1 — Vol 2 Pipeline 启动")
    log(f"  PDF:     {PDF_PATH.name}")
    log(f"  MinerU:  {'✓ 已配置' if os.environ.get('MINERU_API_KEY') else '✗ 未配置 (需 MINERU_API_KEY)'}")
    log(f"  Qwen-VL: {'✓ 已配置' if DASHSCOPE_KEY else '✗ 未配置 (需 DASHSCOPE_API_KEY)'}")
    log(f"  Ollama:  {ANNOTATE_MODEL}")
    log("=" * 60)

    if args.only_report:
        for f, label in [(CHUNKS_SMART, "chunks_smart.json"), (CHUNKS_RAW, "chunks_raw.json")]:
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                log(f"\n{label}: {d.get('total_chunks')} chunks")
                if "topics_distribution" in d:
                    log("Topics分布:")
                    for t, c in d["topics_distribution"].items():
                        log(f"  {t:<25} {c}")
                return
        log("暂无结果文件", "WARN")
        return

    visual_stats = {}
    if not args.skip_extract:
        visual_stats = step1_extract()
    else:
        log("⏭  跳过Step1 (--skip-extract)")

    n_chunks = 0
    if not args.skip_split:
        n_chunks = step2_split()
    else:
        log("⏭  跳过Step2 (--skip-split)")
        if CHUNKS_RAW.exists():
            n_chunks = json.loads(CHUNKS_RAW.read_text()).get("total_chunks", 0)

    topic_dist = {}
    if not args.skip_annotate:
        n_chunks, topic_dist = step3_annotate()
    else:
        log("⏭  跳过Step3 (--skip-annotate)")

    # ── 最终汇报 ──
    total_min = (time.time() - t_start) / 60
    diff = n_chunks - 1427
    log("\n" + "=" * 60)
    log("✅  Stage 1 完成 — 汇报")
    log("=" * 60)
    log(f"  总chunk数:    {n_chunks}  (OFC基准1427，{'+' if diff >= 0 else ''}{diff})")
    if visual_stats.get("total_pages"):
        log(f"  视觉内容占比: {visual_stats['visual_pct']:.0f}%  "
            f"(表格页{visual_stats['table_pages']}, "
            f"图片页{visual_stats['img_pages']} / "
            f"{visual_stats['total_pages']}页)")
    if topic_dist:
        top3 = sorted(topic_dist.items(), key=lambda x: -x[1])[:3]
        log(f"  Top3 topics:  {', '.join(f'{k}({v})' for k, v in top3)}")
    log(f"  总耗时:       {total_min:.1f} 分钟")
    log(f"  输出文件:     {CHUNKS_SMART}")
    log("=" * 60)


if __name__ == "__main__":
    main()
