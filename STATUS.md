# L0 餐饮研发引擎 — 项目状态

> 每次新对话fetch：
> https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/STATUS.md

## 系统定位
```
最终形态：餐饮研发引擎
目标用户：专业厨师 / 餐饮老板 / 研发团队
核心能力：因果链推理（不是检索）
```

---

## 整体架构
```
┌─────────────────────────────────────────────┐
│              餐饮研发引擎                      │
│                                             │
│  输入：审美意图 / 问题 / 食谱                  │
│         ↓                                   │
│  ┌──────────────────────────────┐           │
│  │     双RAG引擎                 │           │
│  │  传统RAG（Weaviate）          │           │
│  │  → 回答"这个原理是什么"        │           │
│  │  GraphRAG（Neo4j）           │           │
│  │  → 回答"这些原理怎么联动"      │           │
│  └──────────────────────────────┘           │
│         ↓                                   │
│  L0因果链推理                                │
│         ↓                                   │
│  输出：配方 / 解释 / 优化建议                  │
└─────────────────────────────────────────────┘
```

---

## 知识层次架构
```
L0 科学原理图谱（因果链网络）
    ↕ enables / conflicts / requires / amplifies
    节点 = 科学原理
    边 = 因果关系（带条件+强度）
    来源：食品科学书蒸馏

L6 风味审美层
    ↕ 映射关系
    粤菜审美文字 → L0解码
    "清而不淡，鲜而不腥" →
      清 = 低温长煮+撇浮沫
      不淡 = 足够氨基酸萃取
      鲜 = 食材新鲜度控制
      不腥 = 挥发性胺去除
    来源：Flavor Bible / Flavor Thesaurus / 粤菜文字

配方参数库
    传统食谱结构化提取
    来源：Cookbook系列
```

---

## 三类问题能力
```
问题A：解释层
"为什么我的鸡汤不够鲜？"
→ L0原理库推理

问题B：优化层
"胶质感不够，参数怎么调？"
→ L0 + 边界数据

问题C：设计层
"给我一个粤菜风格、满足健康+鲜味目标的鸡汤配方"
→ L0 + 边界 + L6感官trade-off
```

---

## 合成数据策略
```
L0原理 × L6审美 → 生成配方推理链（海量）

用途：
→ 扩充参数空间覆盖
→ 训练推理路径
→ 后期fine-tune专属烹饪推理模型

前提：L0质量要高，L6要来自真实文字
不用于：扩充L0原理本身（防止幻觉）
```

---

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| OFC Stage 1-3 | ✅ 完成 | 303条L0原理 |
| MC Vol2 Stage 1 | 🔄 进行中 | 子对话4 |
| MC Vol3/4/1 Stage 1 | 🔄 进行中 | batch_mc_stage1.py，Codex本地跑 |
| MC Stage 2（向量匹配） | ⏳ 待做 | Gemini Embedding 2 |
| MC Stage 3（蒸馏） | ⏳ 待做 | Claude Opus 4.6 |
| Neo4j图谱搭建 | ⏳ 待做 | MC完成后 |
| 因果链实验 | ⏳ 待做 | 验证GraphRAG方向 |
| 第二批科学书 | ⏳ 待做 | Neurogastronomy等 |
| L6风味层 | ⏳ 待做 | Flavor Bible结构化 |
| 粤菜审美层 | ⏳ 待做 | 文字待下载 |
| Cookbook结构化 | ⏳ 待做 | 配方参数库 |
| 合成数据生成 | ⏳ 待做 | L0+L6就绪后 |

---

## Pipeline流程（每本书）
```
PDF/epub
  ↓ Step 0: epub→PDF（calibre）
  ↓ Step 1: MinerU API（文字提取，超200页自动切分）
  ↓ Step 2: qwen3-vl-plus（图片/表格识别）
  ↓ Step 3: 合并 → raw_merged.md
  ↓ Step 4: qwen3.5:2b（按章节切分，每章300-500字）
  ↓ Step 5: qwen3.5:9b（标注summary+topics）
  → chunks_smart.json

Stage 2：Gemini Embedding 2
  chunks_smart.json → gemini-embedding-2-preview向量化
  → 存入Weaviate
  → 306题 × chunks cosine匹配
  → question_chunk_matches.json
  （用Batch API，50%折扣，~$0.10/M tokens）

Stage 3：Claude蒸馏
  question_chunk_matches.json → Claude Opus 4.6
  → l0_principles.jsonl
```

---

## 技术栈

| 组件 | 选型 | 状态 |
|------|------|------|
| L0蒸馏 | Claude Opus 4.6，代理API | ✅ |
| 代理API | http://1.95.142.151:3000 Bearer | ✅ |
| PDF提取 | MinerU API（MINERU_API_KEY） | ✅ |
| 图片/表格 | qwen3-vl-plus（DASHSCOPE_API_KEY） | ✅ |
| 切分 | qwen3.5:2b Ollama本地 | ✅ |
| 标注 | qwen3.5:9b Ollama本地 | ✅ |
| Embedding | gemini-embedding-2-preview（GEMINI_API_KEY） | 🆕 |
| 向量库 | Weaviate | ✅ |
| 图谱库 | Neo4j Docker | ⏳ |
| 增量学习 | Cognee | ⏳ |
| RAG平台 | Dify | ✅ |
| epub转换 | calibre（ebook-convert） | ✅ |

---

## 环境变量
```bash
export MINERU_API_KEY=""
export DASHSCOPE_API_KEY=""
export GEMINI_API_KEY=""        # Google AI Studio
# Claude代理不用key，用Bearer在脚本里配置
```

---

## 关键文件路径
```
/Users/jeff/l0-knowledge-engine/
├── data/l0_question_master.json              ✅ 306题
├── output/
│   ├── stage3/l0_principles_fixed.jsonl     ✅ 303条OFC原理
│   └── mc/
│       ├── batch_progress.json              🔄 进度追踪
│       ├── vol2/stage1/chunks_smart.json    🔄 生成中
│       ├── vol3/stage1/chunks_smart.json    🔄 生成中
│       ├── vol4/stage1/chunks_smart.json    ⏳
│       └── vol1/stage1/chunks_smart.json    ⏳
└── scripts/
    ├── batch_mc_stage1.py                   ✅ Codex已推
    └── mineru_api.py                        ✅

/Users/jeff/Documents/厨书数据库/            📚 书库
```

---

## 书库规划

### 第一批：科学书（蒸馏L0）
| 书名 | 状态 |
|------|------|
| On Food and Cooking | ✅ 完成 |
| Modernist Cuisine Vol 1-4 | 🔄 进行中 |
| The Professional Chef | ⏳ |
| The Science of Good Cooking | ⏳ |
| Neurogastronomy | ⏳ |
| Molecular Gastronomy (Hervé This) | ⏳ |
| Mouthfeel | ⏳ |
| The Art of Fermentation | ⏳ |
| Koji Alchemy | ⏳ |
| The Science of Spice | ⏳ |
| Flavorama | ⏳ |
| Professional Baking 7th Ed | ⏳ |
| 冰淇淋风味学 | ⏳ |

### 第二批：风味图谱（直接作为L6节点）
```
Flavor Bible → 风味搭配关系
Flavor Thesaurus → 风味关联
```

### 第三批：粤菜（待下载）
```
传统粤菜审美文字 → L6语言层（护城河）
粤菜食谱 → 配方参数库
```

### 第四批：Cookbook（结构化提取配方参数）
```
French Laundry, Alinea, Noma, Momofuku,
Sous Vide (Keller), Charcuterie, Ratio...
```

---

## 下一步优先级
```
P0：MC Vol2/3/4/1 Stage1全部完成
P0：Stage2 Gemini Embedding 2版本（Codex写）
P1：MC Stage3蒸馏
P2：Neo4j + Cognee搭建
P3：第二批科学书
P4：粤菜书下载+处理
P5：L6风味层建立
P6：合成数据生成
```

---

## GraphRAG图谱设计（待实施）
```
节点类型：
→ Principle（L0原理）
→ Ingredient（食材）
→ Technique（技法）
→ FlavorProfile（风味特征）
→ AestheticGoal（审美目标，L6）

边类型：
→ enables（A使B成为可能）
→ conflicts（A与B冲突）
→ requires（A需要B作前提）
→ amplifies（A增强B的效果）
→ maps_to（L6审美 → L0原理）

边属性：
→ condition（在什么条件下）
→ strength（0-1强度）
→ source_book（来源）
```
