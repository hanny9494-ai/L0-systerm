#!/usr/bin/env python3
"""
Stage 2 v3: Embedding 语义匹配
用 qwen3-embedding:8b 对 chunk summary 和 question_text 向量化，
用余弦相似度做匹配，domain/章节加权。

用法：
    python3 stage2_embedding.py

输入：
    /Users/jeff/l0-knowledge-engine/output/stage1/chunks_smart.json
    /Users/jeff/l0-knowledge-engine/data/l0_question_master.json

中间产物（可复用）：
    /Users/jeff/l0-knowledge-engine/output/stage2/chunk_vectors.npy
    /Users/jeff/l0-knowledge-engine/output/stage2/chunk_meta.json

输出：
    /Users/jeff/l0-knowledge-engine/output/stage2/question_chunk_matches.json
"""

import json, os, time
import numpy as np
import requests
from collections import defaultdict

# ─────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────
BASE           = "/Users/jeff/l0-knowledge-engine"
CHUNKS_PATH    = f"{BASE}/output/stage1/chunks_smart.json"
QUESTIONS_PATH = f"{BASE}/data/l0_question_master.json"
OUT_DIR        = f"{BASE}/output/stage2"
CHUNK_VEC_PATH = f"{OUT_DIR}/chunk_vectors.npy"
CHUNK_META_PATH= f"{OUT_DIR}/chunk_meta.json"
OUTPUT_PATH    = f"{OUT_DIR}/question_chunk_matches.json"

# ─────────────────────────────────────────────
# Ollama 配置
# ─────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/embeddings"
EMBED_MODEL  = "qwen3-embedding:8b"
BATCH_SIZE   = 16   # 每批向量化数量

# ─────────────────────────────────────────────
# Domain → 优先章节 & 加权
# ─────────────────────────────────────────────
DOMAIN_PREFERRED_CHAPTERS = {
    "heat_transfer":     [13, 25],       # ch13=烹饪热传递/褐变, ch25=传热方式/风味
    "chemical_reaction": [25, 21, 19],   # ch25=褐变反应, ch21=糖化学, ch19=发酵化学
    "physical_change":   [13, 25, 26],   # ch13=热处理物理变化, ch25=烹饪, ch26=水化学
    "water_activity":    [26],           # ch26=水的化学性质
    "protein_science":   [7, 10, 11],    # ch7=乳蛋白, ch10=肉类蛋白, ch11=肉类加热
    "lipid_science":     [7, 20],        # ch7=乳脂肪, ch20=酱汁/乳化
    "carbohydrate":      [18, 19, 21],   # ch18=谷物豆类, ch19=面团淀粉, ch21=糖
    "enzyme":            [7, 9, 19],     # ch7=乳酶, ch9=鸡蛋, ch19=面包发酵酶
    "flavor_sensory":    [17, 25],       # ch17=香料风味化学, ch25=风味形成
    "fermentation":      [19, 23],       # ch19=面包发酵, ch23=酒精/醋酸发酵
    "food_safety":       [10, 11, 14],   # ch10=肉类安全, ch11=加热杀菌, ch14=果蔬储存
    "emulsion_colloid":  [20, 7],        # ch20=酱汁乳化/明胶, ch7=乳制品胶体
    "color_pigment":     [11, 14, 25],   # ch11=肉色, ch14=果蔬色素, ch25=褐变
    "equipment_physics": [13, 25],       # ch13=烹饪设备热传递, ch25=传热方式
}

DOMAIN_WEIGHT   = 1.3   # 同 domain chunk 的相似度乘数
CHAPTER_WEIGHT  = 1.1   # 优先章节的额外乘数
MATCH_THRESHOLD = 0.48  # matched 阈值
LOW_THRESHOLD   = 0.40  # low_confidence 阈值

# ─────────────────────────────────────────────
# Embedding 工具
# ─────────────────────────────────────────────
def get_embedding(text: str, retries: int = 3) -> np.ndarray | None:
    """单条文本向量化"""
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": EMBED_MODEL,
                "prompt": text,
            }, timeout=60)
            resp.raise_for_status()
            vec = resp.json().get("embedding")
            if vec:
                return np.array(vec, dtype=np.float32)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"    ⚠️  向量化失败: {e}")
    return None


def batch_embeddings(texts: list[str], desc: str = "") -> list[np.ndarray | None]:
    """批量向量化，带进度显示"""
    results = []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        for text in batch:
            vec = get_embedding(text)
            results.append(vec)
        done = min(i + BATCH_SIZE, total)
        print(f"  {desc} {done}/{total}", end="\r")
    print()
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. 加载 chunks ────────────────────────
    print(f"📂 加载 chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks = raw["chunks"] if isinstance(raw, dict) else raw
    print(f"   ✅ {len(chunks)} 条 chunks")

    # ── 2. 加载题目 ───────────────────────────
    print(f"📂 加载题目...")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        qs_raw = json.load(f)
    if isinstance(qs_raw, dict):
        questions = qs_raw.get("questions", list(qs_raw.values()))
    else:
        questions = qs_raw
    print(f"   ✅ {len(questions)} 道题目")

    # ── 3. chunk 向量化（有缓存则跳过）────────
    if os.path.exists(CHUNK_VEC_PATH) and os.path.exists(CHUNK_META_PATH):
        print(f"\n♻️  读取缓存 chunk 向量: {CHUNK_VEC_PATH}")
        chunk_vecs = np.load(CHUNK_VEC_PATH, allow_pickle=True)
        with open(CHUNK_META_PATH, "r", encoding="utf-8") as f:
            chunk_meta = json.load(f)
        print(f"   ✅ {len(chunk_meta)} 条")
    else:
        print(f"\n🔢 向量化 {len(chunks)} 条 chunk summary...")
        print(f"   模型: {EMBED_MODEL}")

        chunk_meta = []
        texts = []
        for ci, c in enumerate(chunks):
            summary = (c.get("summary") or "").strip()
            chapter = c.get("chapter_num")
            topics  = [t.lower().strip() for t in (c.get("topics") or [])]
            cid     = c.get("chunk_id", f"chunk_{ci}")
            ft_preview = (c.get("full_text") or "")[:100].replace("\n", " ")

            chunk_meta.append({
                "idx":      ci,
                "chunk_id": cid,
                "chapter":  int(chapter) if chapter is not None else None,
                "topics":   topics,
                "preview":  ft_preview,
                "has_summary": bool(summary),
            })
            # 没有summary则用full_text前200字兜底
            texts.append(summary if summary else (c.get("full_text") or "")[:200])

        vecs = batch_embeddings(texts, desc="chunk")

        # 替换失败的向量为零向量
        dim = next((v.shape[0] for v in vecs if v is not None), 768)
        chunk_vecs = np.array([
            v if v is not None else np.zeros(dim, dtype=np.float32)
            for v in vecs
        ], dtype=np.float32)

        np.save(CHUNK_VEC_PATH, chunk_vecs)
        with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
            json.dump(chunk_meta, f, ensure_ascii=False, indent=2)
        print(f"   ✅ 向量已缓存: {CHUNK_VEC_PATH}")

    # ── 4. 逐题匹配 ───────────────────────────
    print(f"\n🔍 开始匹配 {len(questions)} 道题...")
    results       = []
    domain_scores = defaultdict(list)
    no_match_count = 0

    for qi, q in enumerate(questions):
        qid    = q.get("question_id", f"Q-{qi:04d}")
        domain = q.get("domain", "unknown").lower().strip()
        qtext  = q.get("question_text", "")

        # 题目向量化
        q_vec = get_embedding(qtext)
        if q_vec is None:
            print(f"  ⚠️  [{qid}] 向量化失败，跳过")
            continue

        pref_ch = DOMAIN_PREFERRED_CHAPTERS.get(domain, [])

        # 计算所有 chunk 的加权相似度
        scored = []
        for meta in chunk_meta:
            # 跳过噪声章节（参考文献、版权页、纯列表）
            if meta["chapter"] in (15, 28, 29):
                continue

            base_sim = cosine_similarity(q_vec, chunk_vecs[meta["idx"]])

            # domain 加权
            weight = DOMAIN_WEIGHT if domain in meta["topics"] else 1.0

            # 章节加权
            if meta["chapter"] in pref_ch:
                weight *= CHAPTER_WEIGHT

            final_score = base_sim * weight

            scored.append({
                "chunk_id":  meta["chunk_id"],
                "chapter":   meta["chapter"],
                "score":     round(final_score, 4),
                "cosine":    round(base_sim, 4),
                "preview":   meta["preview"],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top3 = scored[:3]

        # 状态判断（基于原始余弦，不含加权）
        top_cosine = top3[0]["cosine"] if top3 else 0
        if top_cosine >= MATCH_THRESHOLD:
            status = "matched"
        elif top_cosine >= LOW_THRESHOLD:
            status = "low_confidence"
        else:
            status = "no_match"
            no_match_count += 1

        entry = {
            "question_id":   qid,
            "domain":        domain,
            "question_text": qtext,
            "top_chunks":    top3,
            "match_status":  status,
        }
        results.append(entry)
        domain_scores[domain].append(top3[0]["score"] if top3 else 0)

        if (qi + 1) % 30 == 0:
            print(f"   ... {qi+1}/{len(questions)} 完成")

    # ── 5. 写出 ───────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 输出: {OUTPUT_PATH}")

    # ── 6. 汇报 ───────────────────────────────
    total         = len(results)
    matched       = sum(1 for r in results if r["match_status"] == "matched")
    low_conf      = sum(1 for r in results if r["match_status"] == "low_confidence")

    print("\n" + "="*58)
    print("📊  Stage 2 v3 Embedding 匹配汇报")
    print("="*58)
    print(f"总题数:            {total}")
    print(f"✅ matched:        {matched}  ({matched/total*100:.1f}%)")
    print(f"⚠️  low_confidence: {low_conf}  ({low_conf/total*100:.1f}%)")
    print(f"❌ no_match:       {no_match_count}  ({no_match_count/total*100:.1f}%)")

    print("\n各 Domain 平均分（加权余弦）:")
    for d, scores in sorted(domain_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0
        bar = "█" * int(avg * 20)
        print(f"  {d:<22} n={len(scores):3d}  avg={avg:.3f}  {bar}")

    print("\n样例（top-1）:")
    for r in results[:5]:
        c = r["top_chunks"][0] if r["top_chunks"] else {}
        print(f"  [{r['question_id']}] {r['match_status']} score={c.get('score',0):.3f}")
        print(f"    Q: {r['question_text'][:50]}")
        print(f"    → ch{c.get('chapter')} {c.get('preview','')[:60]}")
        print()

    print("="*58)
    print("🎉 Stage 2 v3 完成！")
    print(f"\n💡 提示: chunk 向量已缓存，下次运行无需重新向量化")


if __name__ == "__main__":
    main()
