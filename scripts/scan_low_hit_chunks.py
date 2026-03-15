#!/usr/bin/env python3
"""
补丁B：逆向补题 — 扫描MC低命中chunk，生成候选新题
功能：找到306题没有覆盖的MC知识区域，生成新题候选
输入：MC chunks + question_chunk_matches.json
输出：candidate_questions.json（待Jeff审核）

使用时机：MC Vol1-4全部完成Stage1-5之后运行
"""

import json
import time
import argparse
import re
import glob
from pathlib import Path
from collections import defaultdict
import requests
import numpy as np

API_BASE  = "http://1.95.142.151:3000"
API_KEY   = "Bearer"
MODEL     = "claude-sonnet-4-6"
THRESHOLD = 0.55   # 低于此相似度 → 低命中chunk

# 新domain列表（v2）
VALID_DOMAINS = [
    "protein_science", "carbohydrate", "lipid_science", "fermentation",
    "food_safety", "water_activity", "enzyme", "color_pigment",
    "equipment_physics", "maillard_caramelization", "oxidation_reduction",
    "salt_acid_chemistry", "taste_perception", "aroma_volatiles",
    "thermal_dynamics", "mass_transfer", "texture_rheology",
]

SCAN_PROMPT = """你是食品科学知识工程师。

以下是一段烹饪科学书籍的文字片段（来自Modernist Cuisine）。
请判断这段文字是否包含值得提炼为"L0科学原理"的核心知识。

L0科学原理的标准：
1. 有明确的科学机制（不只是操作步骤或历史背景）
2. 有可量化的参数（温度/时间/浓度/比例等）或清晰的因果关系
3. 对烹饪决策有直接指导意义
4. 在我们现有306道题中没有被直接覆盖

排除：
- 纯历史介绍、名厨轶事
- 纯操作步骤（没有解释为什么）
- 已经非常通用的常识（水100°C沸腾等）

已有题目涵盖的domain：
蛋白质变性温度、淀粉糊化、美拉德反应温度、发酵微生物、
脂肪氧化、水分活度、酶活性、色素变化、食品安全参数...

请判断并输出严格JSON：
{
  "has_principle": true/false,
  "reason": "一句话说明为什么有/没有",
  "question_text": "如果有，该用什么问题来提取这个原理？（中文）",
  "domain": "对应哪个domain（从列表选择）",
  "key_parameters": ["关键数值或参数"],
  "priority": "high/medium/low",
  "novelty_note": "这个知识点在哪里是新的/独特的？"
}

domain列表：""" + ", ".join(VALID_DOMAINS) + """

待分析文字：
{chunk_text}"""


def call_claude(chunk_text: str) -> dict:
    payload = {
        "model": MODEL,
        "max_tokens": 600,
        "messages": [{
            "role": "user",
            "content": SCAN_PROMPT.format(chunk_text=chunk_text[:3000])
        }]
    }
    resp = requests.post(
        f"{API_BASE}/v1/messages",
        headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
        json=payload, timeout=45
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    return json.loads(raw)


def load_mc_chunks(mc_glob: str) -> list:
    """加载所有MC chunks"""
    chunks = []
    for path in sorted(glob.glob(mc_glob)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        chunk_list = data if isinstance(data, list) else data.get("chunks", [])
        source_vol = Path(path).parts[-3] if len(Path(path).parts) >= 3 else "mc_unknown"
        for c in chunk_list:
            c["_source_vol"] = source_vol
            c["_source_file"] = str(path)
        chunks.extend(chunk_list)
        print(f"  加载 {path}: {len(chunk_list)}个chunk")
    return chunks


def build_hit_score_map(matches_path: str) -> dict:
    """建立 chunk_id → 最高命中分数 的映射"""
    with open(matches_path, encoding="utf-8") as f:
        matches = json.load(f)
    hit_map = defaultdict(float)
    for m in matches:
        for c in m.get("top_chunks", []):
            cid = c.get("chunk_id", "")
            score = c.get("cosine", c.get("score", 0.0))
            hit_map[cid] = max(hit_map[cid], score)
    return dict(hit_map)


def is_noise_chunk(chunk: dict) -> bool:
    """过滤掉噪音chunk"""
    text = (chunk.get("full_text") or chunk.get("summary") or "").lower()
    noise_signals = [
        "references", "bibliography", "index", "copyright",
        "contents", "acknowledgment", "preface", "foreword",
        "figure ", "table ", "photo ", "image ",
        "recipe for", "serves 4", "makes about",
    ]
    if len(text) < 100:
        return True
    return any(s in text for s in noise_signals)


def get_chunk_text(chunk: dict) -> str:
    return (chunk.get("full_text") or
            chunk.get("summary") or
            chunk.get("text") or "").strip()


def main():
    parser = argparse.ArgumentParser(description="逆向补题：扫描MC低命中chunk")
    parser.add_argument("--mc-chunks",
                        default="output/mc/vol*/stage1/chunks_smart.json",
                        help="MC chunks文件glob路径")
    parser.add_argument("--matches",
                        default="output/stage2/question_chunk_matches.json")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="低命中阈值（默认0.55）")
    parser.add_argument("--output",
                        default="output/gap_analysis/candidate_questions.json")
    parser.add_argument("--max-chunks", type=int, default=200,
                        help="最多处理多少个低命中chunk")
    parser.add_argument("--dry-run", action="store_true",
                        help="只扫描不调用API（看统计数据）")
    args = parser.parse_args()

    print("=" * 60)
    print("补丁B：逆向补题扫描")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n[1/4] 加载MC chunks...")
    mc_chunks = load_mc_chunks(args.mc_chunks)
    print(f"  共 {len(mc_chunks)} 个MC chunks")

    print("\n[2/4] 建立命中分数映射...")
    hit_map = build_hit_score_map(args.matches)
    print(f"  映射了 {len(hit_map)} 个chunk的命中分数")

    # Step 2: 找低命中chunk
    print(f"\n[3/4] 过滤低命中chunk（阈值 < {args.threshold}）...")
    low_hit = []
    for chunk in mc_chunks:
        cid = chunk.get("chunk_id") or chunk.get("id") or ""
        if is_noise_chunk(chunk):
            continue
        score = hit_map.get(cid, 0.0)
        if score < args.threshold:
            chunk["_hit_score"] = score
            low_hit.append(chunk)

    print(f"  低命中chunk：{len(low_hit)} / {len(mc_chunks)}")

    # 按得分排序（得分越低越可能是盲区）
    low_hit.sort(key=lambda c: c["_hit_score"])

    # 限制处理数量
    to_scan = low_hit[:args.max_chunks]
    print(f"  本次处理：{len(to_scan)} 个")

    if args.dry_run:
        print("\n[dry-run] 跳过API调用，输出前10条chunk预览：")
        for c in to_scan[:10]:
            text = get_chunk_text(c)[:100]
            print(f"  [{c['_hit_score']:.3f}] {text}...")
        return

    # Step 3: 扫描每个低命中chunk
    print(f"\n[4/4] 扫描低命中chunk，识别候选新题...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    for i, chunk in enumerate(to_scan):
        text = get_chunk_text(chunk)
        if not text:
            continue

        cid = chunk.get("chunk_id") or chunk.get("id") or f"mc_chunk_{i}"
        print(f"  [{i+1}/{len(to_scan)}] {cid} (hit={chunk['_hit_score']:.3f})",
              end=" ", flush=True)

        try:
            result = call_claude(text)

            if result.get("has_principle") and result.get("priority") in ("high", "medium"):
                candidate = {
                    "source_chunk_id": cid,
                    "source_vol": chunk.get("_source_vol", ""),
                    "hit_score": chunk["_hit_score"],
                    "chunk_preview": text[:300],
                    "question_text": result.get("question_text", ""),
                    "domain": result.get("domain", ""),
                    "key_parameters": result.get("key_parameters", []),
                    "priority": result.get("priority", "medium"),
                    "novelty_note": result.get("novelty_note", ""),
                    "reason": result.get("reason", ""),
                    "status": "pending",  # pending / approved / rejected
                }
                candidates.append(candidate)
                print(f"✓ [{result['priority']}] {result.get('domain','')} "
                      f"→ {result.get('question_text','')[:40]}...")
            else:
                print(f"- 跳过 ({result.get('reason','')[:50]})")

        except Exception as e:
            print(f"✗ 错误: {e}")

        time.sleep(0.4)

    # 写出结果
    output = {
        "meta": {
            "total_mc_chunks": len(mc_chunks),
            "low_hit_chunks": len(low_hit),
            "scanned": len(to_scan),
            "candidates_found": len(candidates),
            "threshold": args.threshold,
        },
        "candidates": sorted(candidates,
                              key=lambda c: (c["priority"] == "high", -c["hit_score"]),
                              reverse=True)
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n完成：找到 {len(candidates)} 个候选新题")
    print(f"输出：{args.output}")
    print("\n下一步：打开 review_new_questions.html 审核候选题目")


if __name__ == "__main__":
    main()
