#!/usr/bin/env python3
"""
Stage 3: Claude Opus 蒸馏 L0 原理
每道题取 top 3 chunks 合并送入 Claude，抽取科学原理
"""

import json
import os
import time
import re
import sys
import argparse
import requests
from pathlib import Path
from datetime import datetime

# ── 配置 ──────────────────────────────────────────────────────────────────────
API_ENDPOINT = "http://1.95.142.151:3000/v1/messages"
API_KEY      = "sk-7g6RsJ5li3UUa3JXfHTzpvdsPimJPOre3S5eyN7WWdrrt33E"
MODEL        = "claude-opus-4.6"   # 点号，非连字符

INPUT_MATCHES = "/Users/jeff/l0-knowledge-engine/output/stage2/question_chunk_matches.json"
INPUT_CHUNKS  = "/Users/jeff/l0-knowledge-engine/output/stage1/chunks_smart.json"
# chunk_id 格式: "chunk_1327" → chunks_smart.json["chunks"][1327]

OUT_DIR       = Path("/Users/jeff/l0-knowledge-engine/output/stage3")
OUT_JSONL     = OUT_DIR / "l0_principles.jsonl"
OUT_PROGRESS  = OUT_DIR / "progress.json"
OUT_FAILED    = OUT_DIR / "failed.json"

TIMEOUT_SEC   = 90
MAX_RETRIES   = 3
SAVE_EVERY    = 20

# 成本估算 ($/M tokens)
COST_IN  = 15.0
COST_OUT = 75.0

# domain 序号计数器
domain_seq: dict[str, int] = {}


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def load_json(path: str) -> any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_progress() -> dict:
    if OUT_PROGRESS.exists():
        return load_json(str(OUT_PROGRESS))
    return {"completed": [], "total_in": 0, "total_out": 0}

def save_progress(progress: dict):
    with open(OUT_PROGRESS, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_failed() -> list:
    if OUT_FAILED.exists():
        return load_json(str(OUT_FAILED))
    return []

def save_failed(failed: list):
    with open(OUT_FAILED, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

def append_jsonl(record: dict):
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def infer_domain(question_id: str, question_text: str) -> str:
    """从问题ID或文本推断 domain"""
    qid = question_id.upper()
    if "HT"  in qid: return "HT"   # 热传导
    if "FM"  in qid: return "FM"   # 流体力学
    if "EM"  in qid: return "EM"   # 电磁
    if "CH"  in qid: return "CH"   # 化学
    if "BIO" in qid: return "BIO"  # 生物
    if "SEN" in qid: return "SEN"  # 感知
    if "MT"  in qid: return "MT"   # 材料
    if "OPT" in qid: return "OPT"  # 光学
    # 文本关键词兜底
    text = question_text.lower()
    for kw, dm in [("热","HT"),("温","HT"),("冷","HT"),("流","FM"),("压","FM"),
                   ("电","EM"),("磁","EM"),("化学","CH"),("反应","CH"),
                   ("生物","BIO"),("细胞","BIO"),("感","SEN"),("材料","MT"),("光","OPT")]:
        if kw in text:
            return dm
    return "GEN"

def next_seq(domain: str) -> str:
    domain_seq[domain] = domain_seq.get(domain, 0) + 1
    return f"{domain_seq[domain]:03d}"

def build_prompt(question_text: str, chunks: list[dict]) -> str:
    texts = []
    for i, c in enumerate(chunks[:3], 1):
        ft = c.get("full_text") or c.get("text") or c.get("content") or ""
        texts.append(f"[文本{i}] {ft[:2000]}")  # 单段最多2000字
    ref = "\n\n".join(texts)
    return f"""分析后抽取原理。

问题: {question_text}

参考文本:
{ref}

格式:
<thinking>分析三段文本，找出最相关的科学机制，确认数值</thinking>
<principle>
{{"principle_name": "中文名", "mechanism": "physics/chemistry/biology/sensory", "scientific_statement": "含数值的可证伪陈述", "boundary_conditions": ["条件:数值"], "citation_quote": "原文<30词"}}
</principle>"""

def call_api(prompt: str) -> dict:
    """调用 Claude API，返回 {content, in_tokens, out_tokens}"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",   # OneAPI/NewAPI 代理用 Bearer
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()
    content = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            content += block["text"]
    usage = data.get("usage", {})
    return {
        "content": content,
        "in_tokens":  usage.get("input_tokens", 0),
        "out_tokens": usage.get("output_tokens", 0),
    }

def parse_response(content: str) -> dict:
    """从 Claude 响应中提取 thinking + principle JSON"""
    thinking = ""
    tm = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
    if tm:
        thinking = tm.group(1).strip()

    principle = {}
    pm = re.search(r"<principle>\s*(\{.*?\})\s*</principle>", content, re.DOTALL)
    if pm:
        try:
            principle = json.loads(pm.group(1))
        except json.JSONDecodeError:
            # 尝试宽松解析
            raw = pm.group(1)
            for key in ["principle_name","mechanism","scientific_statement","boundary_conditions","citation_quote"]:
                km = re.search(rf'"{key}"\s*:\s*"([^"]*)"', raw)
                if km:
                    principle[key] = km.group(1)
            bc = re.search(r'"boundary_conditions"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
            if bc:
                items = re.findall(r'"([^"]+)"', bc.group(1))
                principle["boundary_conditions"] = items

    return {"thinking": thinking, "principle": principle}


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 3: 蒸馏 L0 原理")
    parser.add_argument("--limit",   type=int, default=0,
                        help="只处理前 N 题（0 = 全量）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不调用 API，只打印 prompt 预览")
    parser.add_argument("--preview", type=int, default=3,
                        help="dry-run 时展示前 N 题 prompt（默认 3）")
    parser.add_argument("--reset",   action="store_true",
                        help="清空 progress/failed，从头跑（配合 --limit 使用）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset:
        for p in [OUT_PROGRESS, OUT_FAILED, OUT_JSONL]:
            if p.exists():
                p.unlink()
        print("🗑️  已清空 progress / failed / jsonl\n")

    label = f"[试跑 {args.limit} 题]" if args.limit else "[全量]"
    if args.dry_run:
        label = f"[DRY-RUN 预览 {args.preview} 题]"
    print(f"🚀 Stage 3 {label}\n")

    print("📂 加载输入文件...")
    matches_raw = load_json(INPUT_MATCHES)
    chunks_raw  = load_json(INPUT_CHUNKS)

    # chunks_smart.json 结构: {"chunks": [...], "split_model": ..., ...}
    # 每个 chunk 无 chunk_id 字段，按数组下标对应 "chunk_0", "chunk_1", ...
    chunks_list = chunks_raw.get("chunks", chunks_raw) if isinstance(chunks_raw, dict) else chunks_raw
    chunk_map: dict[str, dict] = {}
    for idx, c in enumerate(chunks_list):
        chunk_map[f"chunk_{idx}"] = c

    print(f"   chunks 总数: {len(chunk_map)}")

    # question_chunk_matches.json 是 list，每项有 question_id / top_chunks
    questions = matches_raw if isinstance(matches_raw, list) else list(matches_raw.values())
    print(f"   题目总数: {len(questions)}")

    progress = load_progress()
    completed_set = set(progress.get("completed", []))
    failed_list   = load_failed()
    failed_set    = {f["question_id"] for f in failed_list}

    total_in  = progress.get("total_in",  0)
    total_out = progress.get("total_out", 0)

    # 初始化 domain_seq（从已写 JSONL 恢复）
    if OUT_JSONL.exists():
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    pid = rec.get("principle_id", "")
                    m = re.match(r"L0-([A-Z]+)-(\d+)", pid)
                    if m:
                        dom, seq = m.group(1), int(m.group(2))
                        domain_seq[dom] = max(domain_seq.get(dom, 0), seq)
                except:
                    pass

    success_count = len(completed_set)
    pending = [q for q in questions
               if (q.get("question_id") or q.get("id") or "") not in completed_set
               and (q.get("question_id") or q.get("id") or "") not in failed_set]

    # ── 限量 ──
    effective_limit = args.preview if args.dry_run else args.limit
    if effective_limit:
        pending = pending[:effective_limit]

    print(f"   已完成: {success_count}  本次处理: {len(pending)}  失败: {len(failed_set)}\n")

    if args.dry_run:
        print("=" * 60)
        print("📋 DRY-RUN：Prompt 预览（不调用 API）")
        print("=" * 60)
        for idx, q in enumerate(pending):
            qid   = q.get("question_id") or q.get("id") or f"Q-{idx:04d}"
            qtext = q.get("question_text") or q.get("text") or q.get("question") or ""
            top_chunks_meta = q.get("top_chunks", [])
            chunk_ids = [item["chunk_id"] for item in top_chunks_meta[:3] if "chunk_id" in item]
            chunks = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
            domain = infer_domain(qid, qtext)
            prompt = build_prompt(qtext, chunks)
            print(f"\n── 题 {idx+1}: {qid}  domain={domain} ──")
            print(f"  问题: {qtext[:100]}")
            print(f"  Chunks: {chunk_ids}")
            # 只打印前 400 字的 prompt
            print(f"  Prompt 片段:\n{prompt[:400]}...")
        print("\n✅ Dry-run 完毕，确认无误后去掉 --dry-run 开始实际调用")
        return

    batch_count = 0

    for idx, q in enumerate(pending):
        qid   = q.get("question_id") or q.get("id") or f"Q-{idx:04d}"
        qtext = q.get("question_text") or q.get("text") or q.get("question") or ""

        # top_chunks 结构: [{"chunk_id": "chunk_1327", "chapter": 25, "score": ..., ...}, ...]
        top_chunks_meta = q.get("top_chunks", [])
        chunk_ids = [item["chunk_id"] for item in top_chunks_meta[:3] if "chunk_id" in item]

        chunks = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
        if not chunks:
            print(f"  ⚠️  [{qid}] 找不到 chunks，跳过")
            failed_list.append({"question_id": qid, "reason": "no_chunks_found"})
            save_failed(failed_list)
            continue

        # domain 优先用题目自带字段
        raw_domain = q.get("domain", "")
        domain = {
            "heat_transfer": "HT", "fluid_mechanics": "FM", "electromagnetism": "EM",
            "chemistry": "CH", "biology": "BIO", "sensory": "SEN",
            "materials": "MT", "optics": "OPT",
        }.get(raw_domain) or infer_domain(qid, qtext)
        seq    = next_seq(domain)
        pid    = f"L0-{domain}-{seq}"

        total_pending = len(pending)
        print(f"  [{idx+1}/{total_pending}] {qid} → {pid}", end="  ", flush=True)

        # 重试逻辑
        record = None
        last_err = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt = build_prompt(qtext, chunks)
                result = call_api(prompt)
                parsed = parse_response(result["content"])

                p = parsed["principle"]
                record = {
                    "principle_id":        pid,
                    "question_id":         qid,
                    "principle_name":      p.get("principle_name", ""),
                    "mechanism":           p.get("mechanism", ""),
                    "scientific_statement": p.get("scientific_statement", ""),
                    "boundary_conditions": p.get("boundary_conditions", []),
                    "citation_quote":      p.get("citation_quote", ""),
                    "_thinking":           parsed["thinking"],
                    "_chunks_used":        chunk_ids[:3],
                    "_tokens": {
                        "in":  result["in_tokens"],
                        "out": result["out_tokens"],
                    },
                }
                total_in  += result["in_tokens"]
                total_out += result["out_tokens"]
                break

            except Exception as e:
                last_err = str(e)
                print(f"\n    ⚠️  attempt {attempt} failed: {last_err[:80]}", end="")
                if attempt < MAX_RETRIES:
                    time.sleep(5 * attempt)

        if record is None:
            print(f"  ✗ 3次失败")
            failed_list.append({"question_id": qid, "reason": last_err[:200]})
            save_failed(failed_list)
            # domain_seq 回退（该 pid 未使用）
            domain_seq[domain] = max(0, domain_seq.get(domain, 1) - 1)
            continue

        # 写入 JSONL
        append_jsonl(record)
        completed_set.add(qid)
        success_count += 1
        batch_count   += 1
        print(f"✓  [{record['principle_name'][:20]}]")

        # 每 20 条保存进度
        if batch_count % SAVE_EVERY == 0:
            progress["completed"]  = list(completed_set)
            progress["total_in"]   = total_in
            progress["total_out"]  = total_out
            save_progress(progress)
            cost = total_in/1e6*COST_IN + total_out/1e6*COST_OUT
            print(f"\n  💾 进度保存 | 已成功 {success_count} | token {total_in+total_out:,} | 估算 ${cost:.3f}\n")

    # 最终保存
    progress["completed"]  = list(completed_set)
    progress["total_in"]   = total_in
    progress["total_out"]  = total_out
    save_progress(progress)

    # ── 汇报 ──────────────────────────────────────────────────────────────────
    total_q    = len(questions)
    ran_q      = len(pending)
    fail_count = len(failed_list)
    cost_usd   = total_in/1e6*COST_IN + total_out/1e6*COST_OUT

    print("\n" + "="*60)
    if args.limit:
        print(f"✅ 试跑完成（前 {args.limit} 题）")
    else:
        print("✅ Stage 3 全量完成")
    print("="*60)
    print(f"  本次处理    : {ran_q} 题  (总题库 {total_q} 题)")
    print(f"  成功率      : {success_count}/{ran_q}  ({success_count/max(ran_q,1)*100:.1f}%)")
    print(f"  失败题目    : {fail_count} 条")
    if failed_list:
        for f in failed_list[:10]:
            print(f"    - {f['question_id']}: {f.get('reason','')[:60]}")
        if fail_count > 10:
            print(f"    ... 共 {fail_count} 条，见 failed.json")
    print(f"  Token 消耗  : input {total_in:,}  output {total_out:,}  total {total_in+total_out:,}")
    print(f"  成本估算    : ${cost_usd:.4f}  (input@${COST_IN}/M  output@${COST_OUT}/M)")
    print(f"\n  输出文件    : {OUT_JSONL}")

    # 各 domain 样例
    print("\n  各 domain 原理样例:")
    shown_domains: set[str] = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    dm  = rec["principle_id"].split("-")[1]
                    if dm not in shown_domains:
                        shown_domains.add(dm)
                        print(f"    [{dm}] {rec['principle_id']} {rec['principle_name']}")
                        print(f"         {rec['scientific_statement'][:80]}")
                except:
                    pass

    print("="*60)

    if args.limit:
        print(f"\n👀 检查输出: {OUT_JSONL}")
        print(f"   质量OK  → python3 stage3_distill.py                   # 全量（断点续跑，已完成的跳过）")
        print(f"   重跑    → python3 stage3_distill.py --reset --limit {args.limit}")


if __name__ == "__main__":
    main()
