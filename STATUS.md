# L0 知识引擎 — 项目状态

> 每次新对话：复制此文件内容粘贴到对话开头

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| OFC Stage 1: 文本切分+标注 | ✅ 完成 | 1427 chunks |
| OFC Stage 1: 问题母表 | ✅ 完成 | 306 题，14 个领域 |
| OFC Stage 2: 问题-Chunk 匹配 | ✅ 完成 | 96.1% matched，avg cosine ~0.83 |
| OFC Stage 3: Claude 蒸馏 | ✅ 完成 | 303条有效原理，$25.6 |
| OFC Stage 3: 质量检查 | ✅ 完成 | 38条quote过长，50条无数值待处理 |
| MC Vol2 Stage 1: PDF提取+切分+标注 | 🔄 进行中 | 子对话4运行中 |
| MC Vol3/4/1 Stage 1 | ⏳ 待做 | 串行，Vol2完成后开始 |

## 技术栈

| 组件 | 选型 |
|------|------|
| 蒸馏模型 | Claude Opus 4.6 |
| API 端点 | http://1.95.142.151:3000 |
| 切分模型 | qwen3.5:2b（3.7s/chunk） |
| 标注模型 | qwen3.5:9b（4.5s/chunk） |
| Embedding | qwen3-embedding:8b（本地 Ollama） |
| PDF文字提取 | MinerU |
| PDF视觉提取 | qwen3-vl-plus / 本地qwen3-vl-32b |
| RAG 平台 | Dify + Weaviate |

## 关键文件路径（本地）
```
/Users/jeff/l0-knowledge-engine/
├── data/
│   └── l0_question_master.json                    ✅ 306 题
├── output/
│   ├── stage1/chunks_smart.json                   ✅ 1427 chunks (OFC)
│   ├── stage2/question_chunk_matches.json         ✅ 匹配结果
│   ├── stage3/l0_principles_fixed.jsonl           ✅ 303条原理
│   ├── stage3/damaged_questions.jsonl             ⚠️ 3条损坏备份
│   └── mc/vol2/stage1/chunks_smart.json           🔄 生成中
├── merge_mineru_qwen.py                           ✅ 合并脚本
├── mineru_api.py                                  ✅ MinerU脚本
└── qwen_vision_compare.py                         ✅ 视觉脚本
```

## 书目状态

| 书目 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| On Food and Cooking (OFC) | ✅ | ✅ | ✅ 303条 |
| MC Vol 2 - Techniques and Equipment | 🔄 | ⏳ | ⏳ |
| MC Vol 3 - Animals and Plants | ⏳ | ⏳ | ⏳ |
| MC Vol 4 - Ingredients and Preparations | ⏳ | ⏳ | ⏳ |
| MC Vol 1 - History and Fundamentals (epub) | ⏳ | ⏳ | ⏳ |

## OFC 原理质量遗留问题

| 问题 | 数量 | 处理方式 |
|------|------|---------|
| citation_quote 过长 | 38条 | 待处理 |
| scientific_statement 无数值 | 50条 | 人工判断 |
| 损坏记录 | 3条 | 可选重跑 |

## Skills

| Skill | 说明 |
|-------|------|
| skills/SKILL_question_chunk_matching.md | Stage 2 语义匹配 |
| skills/SKILL_llm_distill_pipeline.md | Stage 3 蒸馏pipeline |
| skills/SKILL_pdf_vision_extraction.md | MC PDF视觉提取 |

## 下一步

1. 🔄 等MC Vol2 Stage 1完成
2. 🔜 MC Vol3/4/1 串行
3. 🔜 OFC L1 pipeline设计
4. 🔜 多书合并去重
