# 子对话启动模板

## 子对话 3：OFC 质量检查（⏳ 待开）
```
任务：L0 原理质量检查

输入: /Users/jeff/l0-knowledge-engine/output/stage3/l0_principles.jsonl

检查项:
1. principle_name 有中文名
2. scientific_statement 含数值
3. boundary_conditions 含具体数值（格式：条件:数值）
4. citation_quote 来自原文
5. _thinking 有实质推理过程
6. mechanism 在允许值内（physics/chemistry/biology/sensory）

输出: 质量报告 + 问题原理列表 + 各domain优秀样例各1条
```

## 子对话 4：MC Stage 1（⏳ 待开）
```
任务：Modernist Cuisine Stage 1 文本切分+标注

已验证方案:
- 文字：MinerU → /Users/jeff/l0-knowledge-engine/output/comparison/mineru/
- 视觉：qwen3-vl-plus 或本地 qwen3-vl-32b（llama.cpp）
- 合并脚本：/Users/jeff/l0-knowledge-engine/merge_mineru_qwen.py

目标输出格式:
{
  "chunk_idx": 0,
  "full_text": "...",
  "summary": "中文摘要",
  "chapter_num": 1,
  "topics": ["heat_transfer"],
  "source_book": "modernist_cuisine"
}

输出文件: /Users/jeff/l0-knowledge-engine/output/mc/stage1/chunks_smart.json

完成后汇报: 总chunk数、视觉内容占比、与OFC格式差异
```
