# 子对话启动模板

## 通用前置步骤（每个子对话开头都加）

```
请先fetch以下文件获取项目状态和脚本：

状态：https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/STATUS.md

脚本（按需fetch）：
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/mineru_api.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/merge_mineru_qwen.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/qwen_vision_compare.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/stage3_distill.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/stage3_debug.py
```

---

## 子对话4：MC Vol2 Stage 1（🔄 进行中）

```
请先fetch：
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/STATUS.md
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/mineru_api.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/merge_mineru_qwen.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/qwen_vision_compare.py

任务：MC Stage 1 - Vol 2 文本提取+切分+标注

PDF: /Users/jeff/Documents/厨书数据库/工具科学书/Volume 2 - Techniques and Equipment.pdf

Step 1: MinerU文字提取 + qwen-vl图片表格识别 → 合并为 raw_merged.md
Step 2: qwen3.5:2b 切分（Ollama, options={"think": False}）
Step 3: qwen3.5:9b 标注（topics/summary/chapter_num）

合法topics: heat_transfer, chemical_reaction, physical_change, water_activity,
            protein_science, lipid_science, carbohydrate, enzyme, flavor_sensory,
            fermentation, food_safety, emulsion_colloid, color_pigment, equipment_physics

输出: /Users/jeff/l0-knowledge-engine/output/mc/vol2/stage1/chunks_smart.json
完成后汇报：总chunk数、topics分布、各步骤耗时
```

---

## 子对话5：MC Vol3 Stage 1（⏳ 待开）

```
同子对话4，替换：
PDF: /Users/jeff/Documents/厨书数据库/工具科学书/Volume 3 - Animals and Plants.pdf
输出: /Users/jeff/l0-knowledge-engine/output/mc/vol3/stage1/chunks_smart.json
```

---

## 子对话6：MC Vol4 Stage 1（⏳ 待开）

```
同子对话4，替换：
PDF: /Users/jeff/Documents/厨书数据库/工具科学书/Volume 4 - Ingredients and Preparations.pdf
输出: /Users/jeff/l0-knowledge-engine/output/mc/vol4/stage1/chunks_smart.json
```

---

## 子对话7：MC Vol1 Stage 1（⏳ 待开，需先转PDF）

```
先转换：
ebook-convert "/Users/jeff/Documents/厨书数据库/工具科学书/volume-1-History+and+Fundamentals.epub" \
              "/Users/jeff/Documents/厨书数据库/工具科学书/Volume 1 - History and Fundamentals.pdf"

然后同子对话4流程。
输出: /Users/jeff/l0-knowledge-engine/output/mc/vol1/stage1/chunks_smart.json
```

---

## 子对话8：MC Stage 2+3 蒸馏（⏳ 待开）

```
请先fetch：
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/STATUS.md
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/stage3_distill.py
https://raw.githubusercontent.com/hanny9494-ai/L0-systerm/main/scripts/stage3_debug.py

输入: /Users/jeff/l0-knowledge-engine/output/mc/vol*/stage1/chunks_smart.json
问题母表: /Users/jeff/l0-knowledge-engine/data/l0_question_master.json
API: http://1.95.142.151:3000，Authorization: Bearer，model: claude-opus-4.6
输出: /Users/jeff/l0-knowledge-engine/output/mc/stage3/l0_principles.jsonl
```

---

## 子对话9：OFC L1 Pipeline（⏳ 待开）

```
输入: /Users/jeff/l0-knowledge-engine/output/stage3/l0_principles_fixed.jsonl（303条L0原理）
任务: 为每条L0原理生成3-5条L1实践知识
API: http://1.95.142.151:3000，Authorization: Bearer，model: claude-opus-4.6

L1 Schema:
{
  "l1_id": "L1-HT-001-01",
  "l0_id": "L0-HT-001",
  "practice": "实践描述",
  "mechanism_link": "与L0的关联",
  "application": "适用场景",
  "boundary": "边界条件"
}

输出: /Users/jeff/l0-knowledge-engine/output/l1/l1_practices.jsonl
```
