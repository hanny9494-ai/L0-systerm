#!/usr/bin/env python3
"""
Stage 3B — 因果链蒸馏
功能：对303条已有原理补充 proposition_type / causal_chain / boundary_zones
输入：l0_principles_fixed.jsonl + question_chunk_matches.json
输出：l0_principles_v2.jsonl（带因果链的完整版）
"""

import json
import time
import argparse
import sys
import re
from pathlib import Path
import requests

# ── 配置 ─────────────────────────────────────────────────────────────────────
API_BASE    = "http://1.95.142.151:3000"
API_KEY     = "Bearer"
MODEL       = "claude-sonnet-4-6"   # Sonnet够用，省成本；复杂原理可换Opus
MAX_TOKENS  = 1800
SLEEP_SEC   = 0.4   # 限速
CHUNK_CHARS = 2400  # chunk_preview截断长度

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是食品科学知识工程师，专门负责结构化烹饪科学原理。

给你一条已提取的烹饪科学原理（scientific_statement）和对应的原始书本片段。
完成以下四个分析任务，输出严格的JSON，不要输出任何其他内容。

━━━ TASK 1: 命题类型 ━━━
判断 proposition_type（四选一）：
- fact_atom          → 最小不可分事实，单一数值或关系（无因果序列）
- causal_chain       → 完整因果序列 A→B→C（有触发、过程、结果）
- compound_condition → 多个条件必须同时满足才产生结果（n≥2个前提）
- mathematical_law   → 定量数学关系（含公式、比例、对数、平方律等）

注意：一条原理只能属于一种类型。如果同时有因果链和数学关系，
优先标记为 mathematical_law；compound_condition 优先级最高。

━━━ TASK 2: 因果链步骤 ━━━
提取 causal_chain_steps（3-6步），每步前缀：
- "触发：" → 触发条件（输入/外部变量）
- "过程：" → 中间机制（分子/物理/化学过程）
- "结果：" → 最终结果（可观察的烹饪效果）

如果是 fact_atom，只写 ["事实：<陈述内容>"]
如果是 compound_condition，写 ["条件1：...", "条件2：...", "条件N：...", "结果：..."]
如果是 mathematical_law，写 ["变量：...", "关系：<公式>", "含义：...", "烹饪应用：..."]

━━━ TASK 3: 复合命题检测 ━━━
如果 scientific_statement 实际上包含 2 个或以上独立的原子事实：
- needs_split = true
- sub_principles 列出每个子命题（1-2句话）

判断标准：子命题之间用"；另外"/"同时"/"此外"分隔，
或者涉及完全不同的机制（例如美拉德和花青素变色是两种机制）。
如果只是同一机制的补充说明，不要拆分。

━━━ TASK 4: 边界区间 ━━━
如果存在多个临界值，每个临界值对应不同的效果：
- boundary_zones 列出每个区间

只有真正的"分段效果"才填，连续变化不要强行分段。

━━━ 输出格式（严格JSON） ━━━
{
  "proposition_type": "causal_chain",
  "causal_chain_steps": [
    "触发：加热温度超过50°C",
    "过程：肌球蛋白氢键开始断裂",
    "过程：蛋白质空间构象展开",
    "结果：持水力下降约30%",
    "结果：肉质纤维收紧变硬"
  ],
  "causal_chain_text": "温度↑ → 氢键断裂 → 构象展开 → 持水↓ → 变硬",
  "reasoning_type": "single_hop",
  "needs_split": false,
  "sub_principles": [],
  "boundary_zones": [
    {"range": "50-55°C", "effect": "肌球蛋白开始变性"},
    {"range": "65-80°C", "effect": "肌动蛋白变性，收缩显著"}
  ],
  "parallel_chains": [],
  "confidence": 0.92
}"""

USER_TEMPLATE = """原理内容：
{scientific_statement}

领域：{domain}

原始书本来源片段（供参考）：
{chunks_preview}"""


# ── API调用 ────────────────────────────────────────────────────────────────────
def call_claude(scientific_statement: str, domain: str, chunks_preview: str) -> dict:
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": USER_TEMPLATE.format(
                scientific_statement=scientific_statement,
                domain=domain,
                chunks_preview=chunks_preview[:CHUNK_CHARS]
            )
        }]
    }
    resp = requests.post(
        f"{API_BASE}/v1/messages",
        headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
        json=payload,
        timeout=60
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()

    # 清洗markdown代码块
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    return json.loads(raw)


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_chunks_map(matches_path: str) -> dict:
    """question_id → chunks_preview 字符串"""
    with open(matches_path, encoding="utf-8") as f:
        matches = json.load(f)
    mapping = {}
    for m in matches:
        previews = []
        for c in m.get("top_chunks", [])[:3]:
            preview = c.get("preview") or c.get("full_text", "")[:300]
            if preview:
                previews.append(preview)
        mapping[m["question_id"]] = "\n---\n".join(previews)
    return mapping


def load_done_ids(v2_path: str) -> set:
    done = set()
    p = Path(v2_path)
    if not p.exists():
        return done
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            # 标记：原始ID或拆分来源ID
            done.add(rec.get("split_from") or rec["principle_id"])
        except Exception:
            pass
    return done


# ── 拆分处理 ──────────────────────────────────────────────────────────────────
def make_sub_principle(base: dict, sub_stmt: str, idx: int,
                       chunks_map: dict) -> dict:
    """对子命题再跑一次Stage3B"""
    new_id = f"{base['principle_id']}-{chr(65 + idx)}"
    chunks  = chunks_map.get(base.get("question_id", ""), "")
    try:
        sub_result = call_claude(sub_stmt, base.get("domain", ""), chunks)
        time.sleep(SLEEP_SEC)
    except Exception as e:
        print(f"    ⚠️ 子命题{new_id}调用失败: {e}")
        sub_result = {"proposition_type": "fact_atom",
                      "causal_chain_steps": [f"事实：{sub_stmt}"],
                      "causal_chain_text": sub_stmt,
                      "reasoning_type": "single_hop",
                      "needs_split": False,
                      "sub_principles": [],
                      "boundary_zones": [],
                      "parallel_chains": [],
                      "confidence": 0.5}

    return {
        **base,
        "principle_id": new_id,
        "scientific_statement": sub_stmt,
        "split_from": base["principle_id"],
        **{k: sub_result[k] for k in [
            "proposition_type", "causal_chain_steps", "causal_chain_text",
            "reasoning_type", "boundary_zones", "parallel_chains",
            "needs_split", "confidence"
        ] if k in sub_result}
    }


def process_one(principle: dict, result: dict, chunks_map: dict) -> list:
    """返回1条或多条原理记录"""
    if not result.get("needs_split"):
        merged = {
            **principle,
            **{k: result[k] for k in [
                "proposition_type", "causal_chain_steps", "causal_chain_text",
                "reasoning_type", "boundary_zones", "parallel_chains",
                "needs_split", "confidence"
            ] if k in result}
        }
        return [merged]

    # 需要拆分
    subs = result.get("sub_principles", [])
    if not subs:
        # 声称需要拆分但没给子命题，作为普通处理
        merged = {**principle, **result}
        return [merged]

    out = []
    for i, sub in enumerate(subs):
        stmt = sub if isinstance(sub, str) else sub.get("statement", str(sub))
        out.append(make_sub_principle(principle, stmt, i, chunks_map))
    return out


# ── 质检报告 ──────────────────────────────────────────────────────────────────
def generate_report(v2_path: str, report_path: str):
    records = []
    for line in open(v2_path, encoding="utf-8"):
        if line.strip():
            records.append(json.loads(line))

    from collections import Counter
    type_counter = Counter(r.get("proposition_type", "unknown") for r in records)
    split_records = [r for r in records if r.get("split_from")]
    needs_review  = [r for r in records
                     if r.get("needs_split") and len(r.get("sub_principles", [])) == 0]

    lines = [
        "═" * 60,
        "Stage 3B 质检报告",
        "═" * 60,
        "",
        f"总原理数：{len(records)}条",
        f"  其中拆分产生：{len(split_records)}条",
        "",
        "命题类型分布：",
    ]
    for ptype, cnt in type_counter.most_common():
        lines.append(f"  {ptype:<25} {cnt:>4}条  ({cnt/len(records)*100:.0f}%)")

    lines += [
        "",
        "需要Jeff人工确认的记录：",
        f"  causal_chain步骤<2步（链太短）：",
    ]
    short_chain = [r for r in records
                   if r.get("proposition_type") == "causal_chain"
                   and len(r.get("causal_chain_steps", [])) < 2]
    for r in short_chain:
        lines.append(f"    {r['principle_id']}: {r['scientific_statement'][:60]}...")

    lines += [
        "",
        f"  mathematical_law但无数学符号：",
    ]
    bad_math = [r for r in records
                if r.get("proposition_type") == "mathematical_law"
                and not any(c in r.get("causal_chain_text", "")
                            for c in ["∝", "=", "²", "%", "÷", "×", "log"])]
    for r in bad_math:
        lines.append(f"    {r['principle_id']}: {r['scientific_statement'][:60]}...")

    lines += ["", "═" * 60]

    report = "\n".join(lines)
    Path(report_path).write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已写入：{report_path}")


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 3B 因果链蒸馏")
    parser.add_argument("--input",   default="output/stage3/l0_principles_fixed.jsonl")
    parser.add_argument("--matches", default="output/stage2/question_chunk_matches.json")
    parser.add_argument("--output",  default="output/stage3/l0_principles_v2.jsonl")
    parser.add_argument("--report",  default="output/stage3/stage3b_report.txt")
    parser.add_argument("--report-only", action="store_true", help="只生成报告不运行蒸馏")
    parser.add_argument("--dry-run",     action="store_true", help="只处理前5条测试")
    args = parser.parse_args()

    if args.report_only:
        generate_report(args.output, args.report)
        return

    # 加载数据
    print("加载数据...")
    principles = []
    for line in Path(args.input).open(encoding="utf-8"):
        if line.strip():
            principles.append(json.loads(line))
    print(f"  原理：{len(principles)}条")

    chunks_map = load_chunks_map(args.matches)
    print(f"  chunk映射：{len(chunks_map)}题")

    done_ids = load_done_ids(args.output)
    todo = [p for p in principles if p["principle_id"] not in done_ids]
    print(f"  已完成：{len(done_ids)}，待处理：{len(todo)}")

    if args.dry_run:
        todo = todo[:5]
        print("  [dry-run] 只处理前5条")

    # 主循环
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total_out = 0
    errors = 0

    with open(args.output, "a", encoding="utf-8") as out_f:
        for i, p in enumerate(todo):
            pid = p["principle_id"]
            stmt = p.get("scientific_statement", "").strip()

            if not stmt:
                print(f"[{i+1}/{len(todo)}] SKIP {pid} (statement为空)")
                continue

            print(f"[{i+1}/{len(todo)}] {pid}", end=" ", flush=True)

            chunks = chunks_map.get(p.get("question_id", ""), "")

            try:
                result = call_claude(stmt, p.get("domain", ""), chunks)
                records = process_one(p, result, chunks_map)

                for rec in records:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    total_out += 1

                split_note = f" → {len(records)}条" if len(records) > 1 else ""
                print(f"✓ [{result.get('proposition_type','?')}]{split_note}")

            except json.JSONDecodeError as e:
                errors += 1
                print(f"✗ JSON解析失败: {e}")
                # 写fallback记录，保留原始字段
                fallback = {**p, "proposition_type": "unknown",
                            "causal_chain_steps": [], "stage3b_error": str(e)}
                out_f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
                out_f.flush()

            except Exception as e:
                errors += 1
                print(f"✗ 调用失败: {e}")
                time.sleep(2)  # 遇到错误多等一下

            time.sleep(SLEEP_SEC)

    print(f"\n完成：写出{total_out}条，错误{errors}条")
    print(f"输出：{args.output}")

    # 自动生成报告
    generate_report(args.output, args.report)


if __name__ == "__main__":
    main()
