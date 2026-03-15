# 餐饮研发引擎 — 项目状态 v2

> 每次新对话前fetch此文件：
> https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/STATUS.md

---

## 系统定位
- **目标用户**：专业厨师 / 餐饮老板 / 研发团队
- **核心能力**：因果链科学推理 + 粤菜审美转化（不是配方检索）
- **最终形态**：餐饮研发引擎

---

## 当前进度总览

| 阶段 | 状态 | 说明 |
|------|------|------|
| OFC Stage 1-3 | ✅ 完成 | 303条L0原理（l0_principles_fixed.jsonl）|
| MC Vol2 Stage1 | 🔄 进行中 | Step3 merge完成，Step4/5待跑 |
| MC Vol3/4 Stage1 | 🔄 进行中 | Step3 merge完成，Step4/5待跑 |
| MC Vol1 Stage1 | ⏳ 待做 | epub需先转PDF（calibre）|
| Stage2 Embedding | ⏳ 待做 | 切换Gemini Embedding 2 |
| Stage3B 因果链 | ⏳ 待做 | 脚本已就绪，~$6，现在可跑 |
| Stage3.5 粤菜映射 | ⏳ 待做 | Stage3B完成后 |
| Neo4j 图谱 | ⏳ 待做 | Schema v2设计完成 |
| Weaviate 向量库 | ⏳ 待做 | Schema设计完成 |
| L6 风味层 | ⏳ 待做 | Flavor Bible待处理 |
| Station层（粤菜） | ⏳ 待做 | 5个station定义完成 |

---

## 最优先执行（按顺序）

```
① MC Vol2/3/4 跑 Step4（qwen3.5:2b切分）+ Step5（9b标注）
   python batch_mc_stage1.py --vol 2 --start-step 4
   python batch_mc_stage1.py --vol 3 --start-step 4
   python batch_mc_stage1.py --vol 4 --start-step 4

② Stage3B 因果链补充（脚本已在 scripts/stage3b_distill.py）
   python scripts/stage3b_distill.py \
     --input output/stage3/l0_principles_fixed.jsonl \
     --matches output/stage2/question_chunk_matches.json \
     --output output/stage3/l0_principles_v2.jsonl

③ 切换 Gemini Embedding 2，重跑 Stage2（OFC+MC合并）

④ Neo4j Docker 搭建 + Schema v2 导入

⑤ 20道粤菜验证集（白切鸡等基准测试）
```

---

## 技术栈

| 组件 | 选型 | 状态 |
|------|------|------|
| PDF提取 | MinerU API + qwen3-vl-plus | ✅ |
| 文本切分 | qwen3.5:2b Ollama | ✅ |
| Topic标注 | qwen3.5:9b Ollama (think:False) | ✅ |
| 原理蒸馏 | Claude Opus 4.6，代理API | ✅ |
| 因果链蒸馏 | Claude Sonnet 4.6 | ⏳ |
| Embedding | gemini-embedding-2-preview | ⏳ 切换中 |
| 向量库 | Weaviate | ⏳ |
| 图谱库 | Neo4j Docker | ⏳ |
| 增量学习 | Cognee | ⏳ |

---

## 关键文件路径

```
/Users/jeff/l0-knowledge-engine/
├── data/l0_question_master.json              ✅ 306题
├── output/
│   ├── stage2/question_chunk_matches.json    ✅ OFC匹配结果
│   ├── stage3/l0_principles_fixed.jsonl      ✅ 303条原理
│   └── mc/
│       ├── vol2/raw_merged.md                ✅ merge完成
│       ├── vol3/raw_merged.md                ✅ merge完成
│       └── vol4/raw_merged.md                ✅ merge完成
└── scripts/
    ├── batch_mc_stage1.py                    ✅ GitHub已有
    ├── merge_mineru_qwen.py                  ✅ GitHub已有
    ├── stage3b_distill.py                    ✅ 新增
    ├── scan_low_hit_chunks.py                ✅ 新增
    └── review_new_questions.html             ✅ 新增
```

---

## L0 原理库设计（v2）

### 原子命题类型（4种）
- `fact_atom` — 单一数值事实（~35%，106条）
- `causal_chain` — 因果序列A→B→C（~45%，136条）
- `compound_condition` — n元同时条件（~15%，45条，超边候选）
- `mathematical_law` — 定量数学关系（~5%，15条）

### Domain分类（16个，v2）
保留8个不变：protein_science / carbohydrate / lipid_science /
fermentation / food_safety / water_activity / enzyme / color_pigment / equipment_physics

新增/拆分7个：
- chemical_reaction → maillard_caramelization / oxidation_reduction / salt_acid_chemistry
- flavor_sensory → taste_perception / aroma_volatiles
- heat_transfer + physical_change → thermal_dynamics / mass_transfer
- 新增：texture_rheology

---

## 子对话启动模板

```
请先fetch以下文件获取项目最新状态：
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/STATUS.md

需要用到的脚本（按需fetch）：
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/batch_mc_stage1.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/merge_mineru_qwen.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/stage3b_distill.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/scan_low_hit_chunks.py
```

---

## 环境变量
```bash
export MINERU_API_KEY=""
export DASHSCOPE_API_KEY=""
export GEMINI_API_KEY=""
export ANTHROPIC_BASE_URL="http://1.95.142.151:3000"
export ANTHROPIC_API_KEY="Bearer"
```

## GitHub
- Repo: https://github.com/hanny9494-ai/L0-systerm
- 本地同步: ~/L0-docs/
