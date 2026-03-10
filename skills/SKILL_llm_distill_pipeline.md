---
name: llm-distill-pipeline
description: >
  Build and run a multi-stage LLM knowledge distillation pipeline that extracts
  structured scientific principles from text corpora using Claude.
---

# LLM 知识蒸馏 Pipeline

将文本语料库（书籍、文档）通过多阶段 pipeline，最终蒸馏为带数值的可证伪科学原理 JSONL。

## 架构概览

```
Stage 1: 文本分块 (chunks_smart.json)
    ↓
Stage 2: 问题-chunk 向量匹配 (question_chunk_matches.json)
    ↓
Stage 3: Claude 蒸馏原理 → l0_principles.jsonl
```

---

## 关键数据结构

### 输入：question_chunk_matches.json
```json
[{
  "question_id": "L0-Q-HT-001",
  "domain": "heat_transfer",
  "question_text": "为什么水煮温度上限是100°C？",
  "top_chunks": [
    {"chunk_id": "chunk_1327", "chapter": 25, "score": 0.87, "preview": "..."},
    {"chunk_id": "chunk_545",  "chapter": 14, "score": 0.85, "preview": "..."}
  ],
  "match_status": "matched"
}]
```

### 输入：chunks_smart.json
```json
{
  "chunks": [
    {
      "chunk_idx": 0,
      "full_text": "原文内容...",
      "chapter_num": 7,
      "topics": ["protein_science"],
      "summary": "摘要"
    }
  ]
}
```
⚠️ **chunk 无 chunk_id 字段**，按数组下标对应：`chunk_N` → `chunks[N]`

### 输出：l0_principles.jsonl
```json
{
  "principle_id": "L0-HT-001",
  "question_id": "L0-Q-HT-001",
  "principle_name": "相变温度锁定效应",
  "mechanism": "physics",
  "scientific_statement": "液态水在1个标准大气压下于100°C发生液-气相变...",
  "boundary_conditions": ["标准大气压: 沸点=100°C", "海拔每升高305m: 沸点降约1°C"],
  "citation_quote": "all the pan heat at the boil goes into vaporizing the liquid water",
  "_thinking": "...",
  "_chunks_used": ["chunk_1327", "chunk_545", "chunk_1346"],
  "_tokens": {"in": 1401, "out": 552}
}
```

---

## 核心 Prompt 模板

```
分析后抽取原理。

问题: {question_text}

参考文本:
[文本1] {chunk1_full_text[:2000]}
[文本2] {chunk2_full_text[:2000]}
[文本3] {chunk3_full_text[:2000]}

格式:
<thinking>分析三段文本，找出最相关的科学机制，确认数值</thinking>
<principle>
{"principle_name": "中文名", "mechanism": "physics/chemistry/biology/sensory",
 "scientific_statement": "含数值的可证伪陈述", "boundary_conditions": ["条件:数值"],
 "citation_quote": "原文<30词"}
</principle>
```

**质量要求：**
- `scientific_statement` 必须含具体数值
- `boundary_conditions` 每条格式为 `"条件: 数值"`
- `citation_quote` 直接引用原文，<30词

---

## API 配置注意事项

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",   # ✅ 用 Bearer，不是 x-api-key
    "anthropic-version": "2023-06-01",
}
```

- model 名用点号：`claude-opus-4.6`（非连字符 `claude-opus-4-6`）
- 403 Forbidden 通常是 model 名错误或 header 格式错误，不是 key 失效

---

## 脚本使用

```bash
# 1. 先看数据结构（必做）
python3 scripts/stage3_debug.py

# 2. Dry-run：检查 prompt 拼接，不消耗 token
python3 scripts/stage3_distill.py --dry-run

# 3. 试跑 5 题验证质量
python3 scripts/stage3_distill.py --reset --limit 5

# 4. 质量 OK → 全量（自动断点续跑）
python3 scripts/stage3_distill.py

# 5. 重置重跑
python3 scripts/stage3_distill.py --reset --limit N
```

---

## 容错设计

| 机制 | 实现 |
|------|------|
| 断点续跑 | `progress.json` 记录已完成 question_id，重跑自动跳过 |
| 超时 | 90 秒/请求 |
| 重试 | 失败后等待 5s/10s/15s，共 3 次 |
| 失败记录 | 3 次失败后写 `failed.json`，继续下一题 |
| 定期保存 | 每 20 条刷写 progress |

---

## 调试检查清单

拿到新数据集时，**先跑 debug 脚本**确认：

1. `chunks_smart.json` 的顶层结构：`{"chunks": [...]}` 还是直接 `[...]`？
2. chunk 的文本字段名：`full_text` / `text` / `content`？
3. chunk 的 ID 方式：有显式 `chunk_id` 字段，还是按下标隐式 `chunk_N`？
4. `question_chunk_matches.json` 的 chunk 引用字段：`top_chunks` / `matches` / `chunks`？
5. API endpoint 类型：官方 Anthropic（用 `x-api-key`）还是代理（用 `Authorization: Bearer`）？

---

## 成本估算

按 claude-opus-4.6，top-3 chunks（各 2000 字上限）：

| 规模 | input tokens | output tokens | 估算成本 |
|------|-------------|---------------|---------|
| 5 题（试跑） | ~7K | ~4K | ~$0.41 |
| 306 题（全量） | ~425K | ~250K | ~$13 |

定价参考：input $15/M，output $75/M（claude-opus 系列）
