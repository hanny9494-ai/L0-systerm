# L0 知识引擎 — 项目状态

> 每次新对话：fetch此文件获取最新状态
> https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/STATUS.md

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| OFC Stage 1 | ✅ 完成 | 1427 chunks |
| OFC Stage 2 | ✅ 完成 | 96.1% matched |
| OFC Stage 3 | ✅ 完成 | 303条有效原理，$25.6 |
| MC Vol2 Stage 1 | 🔄 进行中 | 子对话4运行中 |
| MC Vol3/4/1 Stage 1 | ⏳ 待做 | 串行 |
| 因果链实验 | 🔄 进行中 | 新子对话验证GraphRAG方向 |

## 架构方向（2026-03-12 确认）

### 最终目标
三类用户：专业厨师 / 餐饮老板 / 研发团队
核心能力：因果链推理（不是检索）
L0定位：解码食谱的钥匙

### 技术方向
传统RAG + GraphRAG 双引擎
- RAG：回答"这个原理是什么"
- GraphRAG：回答"这些原理怎么联动"
图谱存储：Neo4j（待搭建）
图谱构建：场景驱动因果链提取（验证中）

### 待确认
因果链提取实验结果 → 决定是否推倒现有pipeline

## 技术栈

| 组件 | 选型 |
|------|------|
| 蒸馏模型 | Claude Opus 4.6 |
| API | http://1.95.142.151:3000，Bearer认证 |
| 切分 | qwen3.5:2b（Ollama本地） |
| 标注 | qwen3.5:9b（Ollama本地） |
| Embedding | qwen3-embedding:8b |
| PDF提取 | MinerU + qwen3-vl-plus |
| 向量库 | Weaviate |
| 图谱库 | Neo4j（待搭建） |
| RAG平台 | Dify |

## 关键文件
```
/Users/jeff/l0-knowledge-engine/
├── output/stage3/l0_principles_fixed.jsonl  ✅ 303条L0原理
├── output/mc/vol2/stage1/                   🔄 生成中
└── data/l0_question_master.json             ✅ 306题
```

## 书目状态

| 书目 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| On Food and Cooking | ✅ | ✅ | ✅ 303条 |
| MC Vol 2 | 🔄 | ⏳ | ⏳ |
| MC Vol 3 | ⏳ | ⏳ | ⏳ |
| MC Vol 4 | ⏳ | ⏳ | ⏳ |
| MC Vol 1 (epub) | ⏳ | ⏳ | ⏳ |

## Scripts（GitHub备份）
https://github.com/hanny9494-ai/L0-systerm/tree/main/scripts
