# SKILL: LLM 知识蒸馏 Pipeline

## API 注意事项
- 代理endpoint用 `Authorization: Bearer {KEY}`，不是 `x-api-key`
- model名用点号：`claude-opus-4.6`（非连字符）
- chunk_id：`chunk_N` → `chunks[N]`（按下标）

## Prompt 模板
```
分析后抽取原理。
问题: {question_text}
参考文本:
[文本1] {chunk1[:2000]}
[文本2] {chunk2[:2000]}
[文本3] {chunk3[:2000]}

<thinking>分析三段文本，确认数值</thinking>
<principle>
{"principle_name":"中文名","mechanism":"physics/chemistry/biology/sensory",
"scientific_statement":"含数值","boundary_conditions":["条件:数值"],"citation_quote":"<30词"}
</principle>
```

## 容错
断点续跑(progress.json) + 90s超时 + 重试3次 + 每20条保存

## 成本（claude-opus-4.6，top-3）
5题~$0.41，306题全量~$25

## 运行
```bash
python3 scripts/stage3_debug.py          # 诊断
python3 scripts/stage3_distill.py --dry-run   # 验证
python3 scripts/stage3_distill.py --reset --limit 5  # 试跑
python3 scripts/stage3_distill.py        # 全量
```
