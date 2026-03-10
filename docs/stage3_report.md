# Stage 3 质量检查与修复报告

## 最终状态

| 项目 | 结果 |
|------|------|
| 原始记录 | 306 条 |
| 修复后记录 | **303 条** |
| 输出文件 | `l0_principles_fixed.jsonl` |

---

## 问题一：principle_id Domain 漂移 ✅ 已修复

**问题**：principle_id 使用了非标准前缀（GEN/CH/SEN/EM/BIO/OPT），与合法14个 domain 代码不符。

**根因**：Stage 3 生成时 principle_id 的 domain 前缀未从 question_id 提取，而是独立编号。

**修复**：从每条记录的 `question_id`（如 `L0-Q-CR-004`）提取 domain 代码，重新生成 `principle_id`（如 `L0-CR-001`）。

- 重新映射记录数：**284 条**
- 合法 domain 分布见质量报告

---

## 问题二：缺失题目 ✅ 无缺失

**问题**：原以为有5题未生成原理。

**根因**：初版检查脚本用数字编号（1-306）匹配，但实际 `question_id` 为字符串格式（`L0-Q-CR-004`），导致误报。

**结论**：与 `question_chunk_matches.json` 的306题逐一比对，**覆盖率100%，无缺失**。

---

## 问题三：3条损坏记录 ✅ 已移除（忽略处理）

**问题**：以下3条 `scientific_statement`、`boundary_conditions`、`_thinking` 均为空，且互相重复。

| principle_id | question_id |
|---|---|
| L0-FM-024 | L0-Q-FM-022 |
| L0-GEN-143 | L0-Q-SF-015 |
| L0-HT-063 | L0-Q-EP-012 |

**处理**：直接从 fixed 文件中移除，不重跑。303条为最终有效记录。

---

## 问题四：常规字段质量

| 字段 | 通过率 | 状态 |
|------|--------|------|
| `_thinking` 非空 | 99.7% | ✅ |
| `citation_quote` ≤30词 | 86.6% | ⚠️ 38条过长 |
| `scientific_statement` 含数值 | 83.7% | ⚠️ 50条无量化数值 |
| `boundary_conditions` 有冒号结构 | 92.1% | ⚠️ 24条无结构（已忽略） |

**待处理**：38条 citation_quote 过长、50条 scientific_statement 无数值，需人工判断是否为合理的定性描述。

---

## 输出文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 修复后原理库 | `output/stage3/l0_principles_fixed.jsonl` | 303条，可进入下游 |
| 损坏记录备份 | `output/stage3/damaged_questions.jsonl` | 3条，含原始 chunks |
| 详细质量报告 | `output/stage3/quality_report.txt` | 完整字段统计 |
