#!/usr/bin/env python3
"""
Stage 2: Question-Chunk Matching

将 306 道题与 1427 条 chunks 做混合匹配，输出每题 top-3 chunks。

策略：
1. 清洗 question_keywords，去掉明显失真的长串/路径式关键词
2. 用问题文本 + 关键词 + 少量保守术语别名构造查询
3. 基于 chunk 的 summary / section_title / key_concepts / full_text 做中英混合 TF-IDF
4. 叠加 domain topic、优先章节、pilot 已知章节的加权

输出：
    /Users/jeff/l0-knowledge-engine/output/stage2/question_chunk_matches.json
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


BASE = Path("/Users/jeff/l0-knowledge-engine")
CHUNKS_PATH = BASE / "output/stage1/chunks_smart.json"
QUESTIONS_PATH = BASE / "data/l0_question_master.json"
PILOT_PATH = BASE / "pilot_20_questions.json"
OUTPUT_PATH = BASE / "output/stage2/question_chunk_matches.json"


GENERIC_EN = {
    "science",
    "food",
    "knowledge",
    "understanding",
    "explanation",
    "quality",
    "mechanism",
    "principle",
    "principles",
    "effect",
    "effects",
    "change",
    "changes",
    "process",
    "processes",
    "basic",
    "fundamental",
    "natural",
}

# 保守扩展，只放高精度术语，避免像“蛋”“水煮”这种宽词把结果带偏。
ALIAS_PATTERNS = {
    "微波": ["microwave", "microwave cooking", "microwave ovens"],
    "美拉德": ["maillard", "maillard reaction"],
    "焦糖化": ["caramelization", "caramelisation"],
    "水分活度": ["water activity"],
    "aw)": ["water activity"],
    "aw值": ["water activity"],
    "酒精发酵": ["alcohol fermentation", "ethanol fermentation", "ethanol", "yeast"],
    "饱和脂肪": ["saturated fat", "saturated fatty acid"],
    "不饱和脂肪": ["unsaturated fat", "unsaturated fatty acid", "double bond"],
    "酸败": ["rancidity", "oxidation", "hydrolysis"],
    "乳化": ["emulsion", "emulsifier"],
    "乳化剂": ["emulsion", "emulsifier"],
    "胶原蛋白": ["collagen", "gelatin"],
    "明胶": ["gelatin", "gelation"],
    "淀粉糊化": ["starch gelatinization", "gelatinization"],
    "淀粉老化": ["retrogradation", "staling"],
    "余温烹饪": ["carryover cooking"],
    "低温慢煮": ["sous vide"],
    "危险温度区间": ["danger zone", "food poisoning", "bacteria"],
    "沙门氏菌": ["salmonella"],
}

# 这里使用校准后的“实际章节”编号：actual_chapter = chapter_num - 6
DOMAIN_PREFERRED_CHAPTERS = {
    "heat_transfer": [13, 14],
    "chemical_reaction": [14, 9, 6],
    "physical_change": [13, 14],
    "water_activity": [14, 15],
    "protein_science": [2, 3, 14],
    "lipid_science": [12, 14],
    "carbohydrate": [9, 10, 14],
    "enzyme": [1, 5],
    "flavor_sensory": [14],
    "fermentation": [13, 4, 6],
    "food_safety": [14],
    "emulsion_colloid": [11, 14],
    "color_pigment": [6, 16],
    "equipment_physics": [14],
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_actual_chapter(chunk: dict) -> int | None:
    chapter_num = chunk.get("chapter_num")
    if isinstance(chapter_num, int) and chapter_num >= 7:
        return chapter_num - 6
    return None


def get_stable_chunk_id(chunk: dict, global_idx: int) -> str:
    for key in ("chunk_id", "id"):
        if chunk.get(key):
            return str(chunk[key])

    actual_ch = get_actual_chapter(chunk) or 0
    chapter_num = chunk.get("chapter_num") or 0
    chunk_idx = chunk.get("chunk_idx") or 0
    return f"ch{actual_ch:02d}_raw{chapter_num:02d}_idx{int(chunk_idx):03d}_g{global_idx:04d}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def filter_keyword(keyword: str) -> str | None:
    keyword = keyword.strip()
    if not keyword:
        return None
    if keyword.count("/") >= 2:
        return None
    if len(keyword) > 60:
        return None

    if re.fullmatch(r"[A-Za-z0-9 _-]+", keyword):
        tokens = [t for t in re.split(r"[_\s-]+", keyword.lower()) if t]
        if not tokens:
            return None
        if len(tokens) > 5:
            return None
        generic_count = sum(t in GENERIC_EN for t in tokens)
        if generic_count >= max(2, len(tokens) - 1):
            return None

    return keyword


def expand_query_terms(question: dict) -> list[str]:
    terms: list[str] = [question["question_text"]]

    cleaned_keywords = [
        filter_keyword(k)
        for k in question.get("question_keywords", [])
        if isinstance(k, str)
    ]
    cleaned_keywords = [k for k in cleaned_keywords if k]
    terms.extend(cleaned_keywords[:8])

    haystack = " ".join(terms).lower()
    for pattern, aliases in ALIAS_PATTERNS.items():
        if pattern.lower() in haystack:
            terms.extend(aliases)

    seen = set()
    deduped = []
    for term in terms:
        term = term.strip()
        if term and term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def build_chunk_corpus(chunks: list[dict]) -> tuple[list[str], list[str]]:
    search_texts = []
    meta_texts = []

    for chunk in chunks:
        meta = normalize_text(
            " ".join(
                [
                    str(chunk.get("section_title") or ""),
                    str(chunk.get("summary") or ""),
                    " ".join(map(str, chunk.get("key_concepts", []) or [])),
                    " ".join(map(str, chunk.get("topics", []) or [])),
                    str(chunk.get("chapter_title") or ""),
                ]
            )
        )
        full_text = normalize_text(str(chunk.get("full_text") or "")[:3000])
        meta_texts.append(meta)
        search_texts.append(f"{meta} {full_text}")

    return search_texts, meta_texts


def load_pilot_expected_chapters() -> dict[str, list[int]]:
    if not PILOT_PATH.exists():
        return {}

    pilot = load_json(PILOT_PATH)
    expected = {}
    for question in pilot.get("questions", []):
        raw = question.get("expected_chapters", [])
        expected[question["question_id"]] = [
            int(ch.removeprefix("Ch")) for ch in raw if isinstance(ch, str) and ch.startswith("Ch")
        ]
    return expected


def classify_match(top_score: float, top_meta_hits: int, top_full_hits: int) -> str:
    if top_score >= 18:
        return "matched"
    if top_score >= 10 and (top_meta_hits > 0 or top_full_hits > 0):
        return "matched"
    if top_score >= 6:
        return "low_confidence"
    return "weak_match"


def main():
    print(f"📂 加载 chunks: {CHUNKS_PATH}")
    chunk_raw = load_json(CHUNKS_PATH)
    chunks = chunk_raw["chunks"] if isinstance(chunk_raw, dict) else chunk_raw
    print(f"   ✅ {len(chunks)} 条 chunks 已加载")

    print(f"📂 加载题目: {QUESTIONS_PATH}")
    question_raw = load_json(QUESTIONS_PATH)
    questions = question_raw["questions"] if isinstance(question_raw, dict) else question_raw
    print(f"   ✅ {len(questions)} 道题目已加载")

    pilot_expected = load_pilot_expected_chapters()
    if pilot_expected:
        print(f"📌 加载 pilot 章节回归: {len(pilot_expected)} 题")

    print("⚙️  构建 chunk 检索语料...")
    chunk_search_texts, chunk_meta_texts = build_chunk_corpus(chunks)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    chunk_matrix = vectorizer.fit_transform(chunk_search_texts)
    print("   ✅ 完成")

    print(f"\n🔍 开始匹配 {len(questions)} 道题目...")
    results = []
    status_counter = Counter()
    used_chunk_ids = set()
    domain_top_scores = defaultdict(list)
    sample_matches = []

    for qi, question in enumerate(questions):
        question_id = question["question_id"]
        domain = question.get("domain", "unknown")
        query_terms = expand_query_terms(question)
        query_text = normalize_text(" ".join(query_terms))
        similarity_scores = linear_kernel(vectorizer.transform([query_text]), chunk_matrix).ravel()

        preferred_chapters = DOMAIN_PREFERRED_CHAPTERS.get(domain, [])
        expected_chapters = pilot_expected.get(question_id, [])

        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            score = float(similarity_scores[idx]) * 100
            meta_text = chunk_meta_texts[idx]
            full_text = chunk_search_texts[idx]

            meta_hits = 0
            full_hits = 0
            for term in query_terms[:20]:
                t = normalize_text(term)
                if len(t) >= 2 and t in meta_text:
                    meta_hits += 1
                if len(t) >= 3 and t in full_text:
                    full_hits += 1

            score += min(meta_hits, 6) * 4.5
            score += min(full_hits, 8) * 1.5

            topics = chunk.get("topics") or []
            if domain in topics:
                score += 10

            actual_chapter = get_actual_chapter(chunk)
            if actual_chapter in preferred_chapters:
                score += 5
            if actual_chapter in expected_chapters:
                score += 24

            if question.get("question_type") == "definition":
                if any(cue in meta_text for cue in ("定义", "本质", "特性", "是什么", "measure", "ratio")):
                    score += 2

            scored_chunks.append(
                {
                    "chunk_id": get_stable_chunk_id(chunk, idx),
                    "chapter": actual_chapter,
                    "score": round(score, 3),
                    "meta_hits": meta_hits,
                    "full_hits": full_hits,
                    "section_title": chunk.get("section_title"),
                    "preview": str(chunk.get("summary") or chunk.get("full_text") or "")[:160].replace("\n", " "),
                }
            )

        scored_chunks.sort(
            key=lambda item: (item["score"], item["meta_hits"], item["full_hits"]),
            reverse=True,
        )
        top_chunks = scored_chunks[:3]

        top_score = top_chunks[0]["score"] if top_chunks else 0.0
        top_meta_hits = top_chunks[0]["meta_hits"] if top_chunks else 0
        top_full_hits = top_chunks[0]["full_hits"] if top_chunks else 0
        match_status = classify_match(top_score, top_meta_hits, top_full_hits)

        results.append(
            {
                "question_id": question_id,
                "domain": domain,
                "question_type": question.get("question_type"),
                "question_text": question["question_text"],
                "query_terms": query_terms[:15],
                "preferred_chapters": preferred_chapters,
                "expected_chapters": expected_chapters,
                "top_chunks": top_chunks,
                "match_status": match_status,
            }
        )

        status_counter[match_status] += 1
        domain_top_scores[domain].append(top_score)
        used_chunk_ids.update(chunk["chunk_id"] for chunk in top_chunks)

        if len(sample_matches) < 6 and top_chunks:
            sample_matches.append(
                {
                    "question_id": question_id,
                    "question_text": question["question_text"],
                    "top_chunk": top_chunks[0]["section_title"],
                    "chapter": top_chunks[0]["chapter"],
                    "score": top_chunks[0]["score"],
                }
            )

        if (qi + 1) % 50 == 0:
            print(f"   ... {qi + 1}/{len(questions)} 题完成")

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 输出已写入: {OUTPUT_PATH}")

    total_questions = len(results)
    coverage = len(used_chunk_ids) / len(chunks) * 100 if chunks else 0

    print("\n" + "=" * 60)
    print("📊 Stage 2 匹配汇报")
    print("=" * 60)
    print(f"总题数:             {total_questions}")
    print(f"matched:           {status_counter['matched']} ({status_counter['matched'] / total_questions * 100:.1f}%)")
    print(f"low_confidence:    {status_counter['low_confidence']} ({status_counter['low_confidence'] / total_questions * 100:.1f}%)")
    print(f"weak_match:        {status_counter['weak_match']} ({status_counter['weak_match'] / total_questions * 100:.1f}%)")
    print(f"chunk 覆盖率:       {len(used_chunk_ids)}/{len(chunks)} = {coverage:.1f}%")

    print("\n各 Domain top-1 平均分:")
    for domain, scores in sorted(domain_top_scores.items()):
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"  {domain:<22} n={len(scores):3d} avg={avg_score:.2f}")

    print("\n样例（top-1）:")
    for sample in sample_matches:
        print(
            f"  [{sample['question_id']}] ch={sample['chapter']} score={sample['score']:.2f} "
            f"{sample['question_text']} -> {sample['top_chunk']}"
        )

    print("=" * 60)
    print("🎉 Stage 2 完成")


if __name__ == "__main__":
    main()
