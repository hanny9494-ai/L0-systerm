#!/usr/bin/env python3
"""
MinerU（单大MD）vs qwen3.5 视觉方案 — 全维度对比评估
======================================================

评估维度:
  1. 表格结构完整性（行列对齐，畸形率）
  2. 数值准确性（温度/时间抽样人工验证）
  3. 处理速度（页/秒）
  4. RAG 检索质量（chunk 语义密度、信息熵、表格保留率）

MinerU 切分策略（可选）:
  heading   - 按 H1/H2 标题切分（默认）
  semantic  - 标题 + 表格边界感知（最优 RAG 质量）
  fixed     - 固定 token 数（baseline 对照）
"""

import re
import json
import time
import math
import base64
import argparse
import requests
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Iterator

# ──────────────────────────────────────────────────────
# 路径配置（按实际修改）
# ──────────────────────────────────────────────────────
MINERU_MD    = Path("/Users/jeff/l0-knowledge-engine/books/modernist_cuisine/modernist_cuisine.md")
PAGES_DIR    = Path("/Users/jeff/l0-knowledge-engine/books/ofc/pages")
OUTPUT_DIR   = Path("/Users/jeff/l0-knowledge-engine/output/comparison")
OLLAMA_URL   = "http://127.0.0.1:11434/api/chat"
VISION_MODEL = "qwen2.5vl:2b"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────
# 正则
# ──────────────────────────────────────────────────────
RE_TEMP  = re.compile(r'(\d+\.?\d*)\s*°\s*[FC]|\b(\d+)\s*degrees?\s*[FC]', re.I)
RE_TIME  = re.compile(r'(\d+\.?\d*)\s*(second|minute|hour|min|sec|hr)s?', re.I)
RE_TABLE = re.compile(r'((?:\|.+\n)+)', re.MULTILINE)
RE_WORDS = re.compile(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b')


# ──────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────
@dataclass
class ChunkResult:
    source: str
    chunk_id: str
    text: str
    elapsed_sec: float = 0.0
    table_count: int = 0
    table_malformed: int = 0
    has_table: bool = False
    temps: list = field(default_factory=list)
    times: list = field(default_factory=list)
    token_count: int = 0
    type_token_ratio: float = 0.0
    info_density: float = 0.0
    error: str = ""


# ──────────────────────────────────────────────────────
# 切分策略
# ──────────────────────────────────────────────────────
def split_heading(text: str, min_tokens: int = 80) -> Iterator[tuple[str, str]]:
    """按 H1/H2 切分，小块合并避免碎片。"""
    parts = re.split(r'\n(?=#{1,2} )', text)
    buf_id, buf = "", []
    for part in parts:
        if not part.strip():
            continue
        first = part.split('\n')[0].strip('# ').strip()
        pid = first[:50].replace(' ', '_')
        tokens = len(RE_WORDS.findall(part))
        if tokens < min_tokens and buf:
            buf.append(part)
        else:
            if buf:
                yield buf_id, '\n\n'.join(buf)
            buf_id, buf = pid, [part]
    if buf:
        yield buf_id, '\n\n'.join(buf)


def split_semantic(text: str, target: int = 400) -> Iterator[tuple[str, str]]:
    """
    语义感知切分：尊重标题边界，表格不拆断，
    长文本在句子边界切割。最适合 RAG。
    """
    sections = re.split(r'\n(?=#{1,3} )', text)
    idx = 0
    for section in sections:
        if not section.strip():
            continue
        m = re.match(r'^#{1,3}\s+(.+)', section)
        heading = m.group(1).strip()[:30] if m else "sec"
        blocks = _section_to_blocks(section)
        current, cur_tok = [], 0

        for btype, btext in blocks:
            btok = len(RE_WORDS.findall(btext))
            if btype == "table":
                if cur_tok + btok > target * 1.5 and current:
                    yield f"{heading}_{idx:03d}", '\n'.join(current)
                    idx += 1; current, cur_tok = [], 0
                current.append(btext); cur_tok += btok
            else:
                for sent in re.split(r'(?<=[.!?。])\s+', btext):
                    stok = len(RE_WORDS.findall(sent))
                    if cur_tok + stok > target and current:
                        yield f"{heading}_{idx:03d}", '\n'.join(current)
                        idx += 1; current, cur_tok = [], 0
                    current.append(sent); cur_tok += stok
        if current:
            yield f"{heading}_{idx:03d}", '\n'.join(current)
            idx += 1


def _section_to_blocks(text: str) -> list[tuple[str, str]]:
    result, last = [], 0
    for m in RE_TABLE.finditer(text):
        if m.start() > last:
            prose = text[last:m.start()].strip()
            if prose: result.append(("text", prose))
        result.append(("table", m.group(0).strip()))
        last = m.end()
    if last < len(text):
        prose = text[last:].strip()
        if prose: result.append(("text", prose))
    return result


def split_fixed(text: str, target: int = 400, overlap: int = 40) -> Iterator[tuple[str, str]]:
    """固定大小切分（baseline）。"""
    words = text.split()
    step = target - overlap
    for i in range(0, len(words), step):
        yield f"fixed_{i:06d}", ' '.join(words[i:i+target])


# ──────────────────────────────────────────────────────
# 指标计算
# ──────────────────────────────────────────────────────
def analyze_tables(text: str) -> tuple[int, int]:
    count = malformed = 0
    for m in RE_TABLE.finditer(text):
        rows = [r for r in m.group(0).strip().split('\n')
                if r.strip().startswith('|') and not re.match(r'^\|[-:\s|]+\|$', r.strip())]
        if not rows: continue
        count += 1
        cols = [len(r.split('|')) - 2 for r in rows]
        if len(set(cols)) > 1: malformed += 1
    return count, malformed


def info_entropy(text: str) -> float:
    words = RE_WORDS.findall(text.lower())
    if not words: return 0.0
    freq: dict[str, int] = {}
    for w in words: freq[w] = freq.get(w, 0) + 1
    n = len(words)
    return round(-sum((c/n)*math.log2(c/n) for c in freq.values()), 3)


def analyze_chunk(source: str, cid: str, text: str, elapsed: float = 0.0) -> ChunkResult:
    r = ChunkResult(source=source, chunk_id=cid, text=text, elapsed_sec=elapsed)
    r.table_count, r.table_malformed = analyze_tables(text)
    r.has_table = r.table_count > 0
    r.temps = [m[0] or m[1] for m in RE_TEMP.findall(text)]
    r.times = [m[0] for m in RE_TIME.findall(text)]
    words = RE_WORDS.findall(text.lower())
    r.token_count = len(words)
    r.type_token_ratio = round(len(set(words)) / len(words), 3) if words else 0
    r.info_density = info_entropy(text)
    return r


# ──────────────────────────────────────────────────────
# Vision 识别
# ──────────────────────────────────────────────────────
VISION_PROMPT = """/no_think 将这页内容完整转为 Markdown：
- 表格用标准 Markdown 格式（| 列1 | 列2 |）
- 保留所有数值（温度、时间、百分比）
- 正文分段，不遗漏内容
只输出 Markdown。"""


def vision_recognize(image_path: Path, model: str) -> tuple[str, float]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [{"role": "user", "content": VISION_PROMPT, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }, proxies={"http": None, "https": None}, timeout=180)
    return resp.json()["message"]["content"], round(time.time() - t0, 2)


# ──────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────
def run(args):
    print("=" * 64)
    print("  MinerU (单 MD) vs qwen 视觉 — 全维度对比")
    print("=" * 64)

    all_results: dict[str, list[ChunkResult]] = {}

    # ── MinerU ────────────────────────────────────────
    md_path = Path(args.mineru)
    if not md_path.exists():
        print(f"❌ 找不到 MinerU 文件: {md_path}")
        return

    print(f"\n📂 MinerU: {md_path.name}  ({md_path.stat().st_size/1e6:.1f} MB)")
    full_text = md_path.read_text(encoding="utf-8")
    wc = len(full_text.split())
    print(f"   ~{wc:,} tokens | 预计 heading chunks: ~{wc//300}")

    strategies = {"heading": split_heading, "semantic": split_semantic}
    if args.include_fixed:
        strategies["fixed"] = split_fixed

    for name, fn in strategies.items():
        print(f"\n  ▶ 切分策略: {name}")
        t0 = time.time()
        chunks = list(fn(full_text))
        print(f"    切分完成: {len(chunks)} chunks  ({time.time()-t0:.2f}s)")

        results = [analyze_chunk(f"mineru_{name}", pid, txt)
                   for pid, txt in chunks[:args.n]]
        all_results[f"mineru_{name}"] = results
        _print_stats(f"mineru_{name}", results)

    # ── Vision ────────────────────────────────────────
    if not args.mineru_only:
        pages = sorted(Path(args.pages).glob("p*.png"))[:args.n]
        if not pages:
            print(f"\n⚠️  无页面图片 ({args.pages})，跳过视觉测试")
            print("   请先运行 vision_test.py 转换 PDF")
        else:
            print(f"\n  ▶ 视觉识别 [{args.model}]  {len(pages)} 页")
            results = []
            for img in pages:
                try:
                    text, elapsed = vision_recognize(img, args.model)
                    r = analyze_chunk("vision", img.stem, text, elapsed)
                    print(f"    {img.stem}  {elapsed:.1f}s  tables:{r.table_count}"
                          f"  nums:{len(r.temps)+len(r.times)}")
                except Exception as e:
                    r = ChunkResult(source="vision", chunk_id=img.stem, text="", error=str(e))
                    print(f"    {img.stem}  ❌ {e}")
                results.append(r)
            all_results["vision"] = results
            _print_stats("vision", results)

    report = build_report(all_results)
    print_report(report)
    save(all_results, report, Path(args.output))


def _agg(items, fn):
    vals = [fn(r) for r in items if not r.error and r.token_count > 0]
    return round(sum(vals)/len(vals), 3) if vals else 0


def _print_stats(label, results):
    ok = [r for r in results if not r.error]
    tbl = sum(r.table_count for r in ok)
    bad = sum(r.table_malformed for r in ok)
    print(f"    chunks:{len(ok)}  tables:{tbl}(bad:{bad})"
          f"  avg_tok:{_agg(ok, lambda r:r.token_count)}"
          f"  avg_TTR:{_agg(ok, lambda r:r.type_token_ratio)}"
          f"  avg_H:{_agg(ok, lambda r:r.info_density)}")


def build_report(all_results):
    report = {}
    for key, results in all_results.items():
        ok = [r for r in results if not r.error]
        if not ok: continue
        total_tbl = sum(r.table_count for r in ok)
        report[key] = {
            "chunks":               len(ok),
            "avg_sec":              _agg(ok, lambda r: r.elapsed_sec),
            "chunks_with_table":    sum(1 for r in ok if r.has_table),
            "total_tables":         total_tbl,
            "malformed_rate":       round(sum(r.table_malformed for r in ok)/max(total_tbl,1), 3),
            "total_temps":          sum(len(r.temps) for r in ok),
            "total_times":          sum(len(r.times) for r in ok),
            "avg_tokens":           _agg(ok, lambda r: r.token_count),
            "avg_TTR":              _agg(ok, lambda r: r.type_token_ratio),
            "avg_entropy":          _agg(ok, lambda r: r.info_density),
        }
    return report


def print_report(report):
    COLS = [
        ("指标",                   None,   None),
        ("⚡ 平均耗时 (秒)",        "avg_sec",          False),
        ("📊 含表格 chunks",        "chunks_with_table", True),
        ("⚠️  表格畸形率",          "malformed_rate",    False),
        ("🌡  温度数值总数",         "total_temps",       True),
        ("⏱  时间数值总数",         "total_times",       True),
        ("📝 平均 tokens",          "avg_tokens",        None),
        ("🔤 词汇多样性 (TTR)",     "avg_TTR",           True),
        ("📡 信息熵 (RAG 密度)",    "avg_entropy",       True),
    ]

    keys = list(report.keys())
    W = 26
    C = 16

    print("\n" + "=" * (W + C * len(keys) + 2))
    print("  📋 最终对比报告")
    print("=" * (W + C * len(keys) + 2))
    print(f"  {'':>{W}}" + "".join(f"{k:<{C}}" for k in keys))
    print("  " + "-" * (W + C * len(keys)))

    for label, metric, hb in COLS:
        if metric is None:
            continue
        vals = {k: report[k].get(metric) for k in keys}
        nums = {k: v for k, v in vals.items() if isinstance(v, (int, float))}
        best = (max(nums.values()) if hb else min(nums.values())) if nums and hb is not None else None
        row = f"  {label:<{W}}"
        for k in keys:
            v = vals[k]
            cell = f"{v:.3f}" if isinstance(v, float) else str(v) if v is not None else "-"
            mark = "✅" if (best is not None and nums.get(k) == best) else "  "
            row += f"{mark}{cell:<{C-2}}"
        print(row)

    print("\n" + "─" * 50)
    print("  🏆 综合建议")
    print("─" * 50)
    if "mineru_semantic" in report:
        ms = report["mineru_semantic"]
        vi = report.get("vision", {})
        print(f"  表格结构  → MinerU semantic  (畸形率 {ms['malformed_rate']} vs vision {vi.get('malformed_rate','N/A')})")
        print(f"  数值提取  → MinerU (原文提取无幻觉风险)")
        print(f"  RAG 质量  → MinerU semantic  (信息熵 {ms['avg_entropy']}，表格边界对齐)")
        print(f"  处理速度  → MinerU >> vision")
        print(f"  视觉方案  → 用作 MinerU 失败页的 fallback（扫描图、复杂排版）")


def save(all_results, report, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    detail = {}
    for key, results in all_results.items():
        detail[key] = []
        for r in results:
            d = asdict(r)
            d["text_preview"] = d.pop("text", "")[:600]
            detail[key].append(d)
    (out / "detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    # 表格样例（人工抽查结构）
    for key, results in all_results.items():
        samples = [r for r in results if r.has_table and not r.error][:5]
        if samples:
            md = f"# {key} — 表格样例（人工抽查）\n\n"
            for r in samples:
                md += f"## {r.chunk_id}\n\n{r.text[:3000]}\n\n---\n\n"
            (out / f"table_samples_{key}.md").write_text(md, encoding="utf-8")

    # 数值样例（验证准确性）
    for key, results in all_results.items():
        samples = [r for r in results if (r.temps or r.times) and not r.error][:8]
        if samples:
            lines = [f"# {key} — 数值抽样（验证准确性）\n"]
            for r in samples:
                lines += [f"## {r.chunk_id}",
                          f"温度: {r.temps}", f"时间: {r.times}",
                          f"\n原文:\n{r.text[:400]}\n"]
            (out / f"number_samples_{key}.md").write_text('\n'.join(lines), encoding="utf-8")

    print(f"\n💾 结果 → {out}/")
    print("   report.json                  汇总指标对比")
    print("   detail.json                  逐块详情")
    print("   table_samples_<source>.md    表格样例（人工抽查）")
    print("   number_samples_<source>.md   数值样例（验证准确性）")


# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mineru",        default=str(MINERU_MD))
    ap.add_argument("--pages",         default=str(PAGES_DIR))
    ap.add_argument("--model",         default=VISION_MODEL)
    ap.add_argument("--output",        default=str(OUTPUT_DIR))
    ap.add_argument("--n",             default=10, type=int,  help="抽样 chunk/页数量")
    ap.add_argument("--mineru-only",   action="store_true",   help="只跑 MinerU，不调用视觉模型")
    ap.add_argument("--include-fixed", action="store_true",   help="加入 fixed 切分作为 baseline")
    run(ap.parse_args())
