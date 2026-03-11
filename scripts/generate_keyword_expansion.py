#!/usr/bin/env python3
"""
生成关键词扩展映射表 keyword_expansion.json

Step 1: 从 l0_question_master.json 提取所有唯一关键词
Step 2: 对每个关键词，调用本地 LLM 生成3-5个书本摘要风格的同义/近义表达
Step 3: 输出 keyword_expansion.json

用法：
    python3 generate_keyword_expansion.py

输出：
    /Users/jeff/l0-knowledge-engine/data/keyword_expansion.json
"""

import json
import os
import re
import time
import requests
from collections import Counter

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
BASE           = "/Users/jeff/l0-knowledge-engine"
QUESTIONS_PATH = f"{BASE}/data/l0_question_master.json"
OUTPUT_PATH    = f"{BASE}/data/keyword_expansion.json"

# ─────────────────────────────────────────────
# 本地 LLM 配置（Ollama 默认接口）
# 如果你用的不是 Ollama，修改 call_llm() 函数即可
# ─────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:9b"   # 改成你本地实际的模型名

# ─────────────────────────────────────────────
# 调用本地 LLM
# ─────────────────────────────────────────────
def call_llm(prompt: str, timeout: int = 30) -> str:
    """调用 Ollama 本地 LLM，返回生成文本"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2, "think": False,
            "num_predict": 200,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"    ⚠️  LLM 调用失败: {e}")
        return ""


# ─────────────────────────────────────────────
# 解析 LLM 返回的 JSON 数组
# ─────────────────────────────────────────────
def parse_expansion(text: str) -> list[str]:
    """从 LLM 返回文本中提取 JSON 数组"""
    # 尝试直接解析
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x).strip() for x in result if x]
    except json.JSONDecodeError:
        pass

    # 提取第一个 [...] 块
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return [str(x).strip() for x in result if x]
        except json.JSONDecodeError:
            pass

    # 按行/逗号分割兜底
    items = re.split(r'[,\n、；;]+', text)
    items = [i.strip().strip('"\'[]') for i in items]
    items = [i for i in items if 2 <= len(i) <= 12 and i]
    return items[:5]


# ─────────────────────────────────────────────
# 生成扩展词的 Prompt
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = """你是食品科学领域的专家。
给定一个食品科学关键词，请生成3-5个在食品科学教材摘要中常见的同义词或近义表达。

要求：
- 必须是中文
- 每个词2-8个汉字
- 贴近教材/科普文章的实际用词风格（而非专业论文术语）
- 输出纯JSON数组，不要任何解释

示例：
输入：热传导率
输出：["热传导","导热性","热量传递","传热效率","导热系数"]

输入：体感温度
输出：["热量传递","接触导热","比热容","温度感知","冷热感觉"]

输入：{keyword}
输出："""


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    # 1. 加载题目，收集所有关键词
    print(f"📂 加载题目: {QUESTIONS_PATH}")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        qs_raw = json.load(f)
    if isinstance(qs_raw, dict):
        questions = qs_raw.get("questions", list(qs_raw.values()))
    else:
        questions = qs_raw

    all_keywords = []
    for q in questions:
        kws = q.get("question_keywords") or []
        all_keywords.extend(kws)

    # 去重，按频率排序
    kw_counter = Counter(all_keywords)
    unique_kws  = [kw for kw, _ in kw_counter.most_common()]
    print(f"   ✅ 共 {len(unique_kws)} 个唯一关键词（来自 {len(all_keywords)} 个原始关键词）")

    # 只处理中文关键词（含中文字符）
    zh_kws = [kw for kw in unique_kws if re.search(r'[\u4e00-\u9fff]', kw)]
    print(f"   📝 其中中文关键词: {len(zh_kws)} 个")

    # 2. 加载已有结果（支持断点续跑）
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            expansion = json.load(f)
        print(f"   ♻️  已有 {len(expansion)} 条，继续补充...")
    else:
        expansion = {}

    # 3. 逐个扩展
    todo = [kw for kw in zh_kws if kw not in expansion]
    print(f"\n🔍 需要扩展: {len(todo)} 个关键词")
    print(f"   预计耗时: {len(todo) * 2 // 60} 分钟（约2秒/个）\n")

    for i, kw in enumerate(todo):
        prompt = PROMPT_TEMPLATE.format(keyword=kw)
        raw    = call_llm(prompt)
        terms  = parse_expansion(raw)

        # 过滤：只保留中文、长度合理的词，且不与原词相同
        terms = [t for t in terms
                 if re.search(r'[\u4e00-\u9fff]', t)
                 and 2 <= len(t) <= 12
                 and t != kw]
        terms = list(dict.fromkeys(terms))[:5]  # 去重取前5

        expansion[kw] = terms

        status = "✓" if terms else "✗"
        print(f"  [{i+1}/{len(todo)}] {status} {kw} → {terms}")

        # 每50个保存一次
        if (i + 1) % 50 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(expansion, f, ensure_ascii=False, indent=2)
            print(f"  💾 已保存 ({len(expansion)} 条)")

        time.sleep(0.3)  # 避免过载

    # 4. 最终保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(expansion, f, ensure_ascii=False, indent=2)

    # 5. 汇报
    success = sum(1 for v in expansion.values() if v)
    print(f"\n{'='*50}")
    print(f"✅ 完成！共 {len(expansion)} 个关键词")
    print(f"   成功扩展: {success} ({success/len(expansion)*100:.1f}%)")
    print(f"   空结果:   {len(expansion)-success}")
    print(f"   输出: {OUTPUT_PATH}")

    # 打印样例
    print(f"\n样例：")
    for kw, terms in list(expansion.items())[:10]:
        print(f"  {kw} → {terms}")


if __name__ == "__main__":
    main()
