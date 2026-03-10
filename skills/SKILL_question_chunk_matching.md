# SKILL: Question-Chunk Semantic Matching

## 核心结论
- ❌ 中文关键词匹配英文 full_text → 0% 命中
- ❌ 字符串匹配 summary → 术语层次不对齐
- ✅ Embedding 语义匹配：question_text → cosine → chunk.summary

## 推荐参数
DOMAIN_WEIGHT=1.3, CHAPTER_WEIGHT=1.1, MATCH_THRESHOLD=0.48, NOISE_CHAPTERS={15,28,29}

## Ollama Embedding
```python
def get_embedding(text, model="qwen3-embedding:8b"):
    resp = requests.post("http://localhost:11434/api/embeddings",
                         json={"model": model, "prompt": text}, timeout=60)
    return np.array(resp.json()["embedding"], dtype=np.float32)
```

## 本次基准（306题 × 1427 chunks）
matched: 294/306=96.1%, avg cosine ~0.83, 向量化时间~15分钟
