# L0 知识引擎 — 项目状态

> 每次新对话：复制此文件内容粘贴到对话开头

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| OFC Stage 1: 文本切分+标注 | ✅ 完成 | 1427 chunks |
| OFC Stage 1: 问题母表 | ✅ 完成 | 306 题，14 个领域 |
| OFC Stage 2: 问题-Chunk 匹配 | ✅ 完成 | 96.1% matched，avg cosine ~0.83 |
| OFC Stage 3: Claude 蒸馏 | 🔄 进行中 | l0_principles.jsonl 生成中 |
| OFC Stage 3: 质量检查 | ⏳ 待做 | 等 Stage 3 完成后开 |
| MC PDF视觉方案 | ✅ 测试完成 | MinerU+qwen3-vl-plus 合并方案 |
| MC Stage 1: 文本切分 | ⏳ 待做 | 等 OFC 蒸馏完成后并行 |

## 技术栈

| 组件 | 选型 |
|------|------|
| 蒸馏模型 | Claude Opus 4.6 |
| API 端点 | http://1.95.142.151:3000 |
| 切分模型 | qwen3.5:2b |
| 标注模型 | qwen3.5:9b |
| Embedding | qwen3-embedding:8b（本地 Ollama） |
| PDF文字提取 | MinerU |
| PDF视觉提取 | qwen3-vl-plus / 本地qwen3-vl-32b |
| RAG 平台 | Dify + Weaviate |

## 关键文件路径（本地）
```
/Users/jeff/l0-knowledge-engine/
├── data/
│   └── l0_question_master.json              ✅ 306 题
├── output/
│   ├── stage1/chunks_smart.json             ✅ 1427 chunks (OFC)
│   ├── stage2/question_chunk_matches.json   ✅ 匹配结果
│   ├── stage3/l0_principles.jsonl           🔄 蒸馏中
│   └── comparison/ch13_merged.md            ✅ MC视觉测试
├── merge_mineru_qwen.py                     ✅ 合并脚本
└── books/ofc/chapters/                      OFC章节文本
```

## MC 视觉方案结论

| 内容类型 | 方案 |
|---------|------|
| 文字段落 | MinerU（准确、免费） |
| 图片化表格 | qwen3-vl-plus（MinerU无法识别） |
| 合并后 | 词数31,799 / 表格54个 / 残留占位符3个 |

本地部署：M4 Max 可跑 qwen3-vl-32b（llama.cpp，约20GB Q4量化）

## 下一步

1. ⏳ OFC Stage 3 蒸馏完成 → 质量检查
2. 🔜 MC Stage 1：对全书跑 MinerU+qwen 合并流程
3. 🔜 本地部署 qwen3-vl-32b 降低成本
4. 🔜 两书合并去重 + 交叉验证
