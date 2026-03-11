#!/usr/bin/env python3
"""
Stage 3 Quality Check Script
Usage: python3 stage3_qc.py /path/to/l0_principles.jsonl
"""

import json
import re
import sys
from collections import defaultdict, Counter

# ── 配置 ──────────────────────────────────────────────────────────────────────
VALID_DOMAINS = {"HT", "CR", "PC", "WA", "PS", "LS", "CB", "EN", "FS", "FM", "SF", "EC", "CP", "EP"}
TOTAL_QUESTIONS = 306
CITATION_MAX_WORDS = 30
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append((i, str(e)))
    return records, errors

def extract_domain(principle_id):
    """Extract domain prefix from principle_id like HT-001, CR-002-a, etc."""
    if not principle_id:
        return "MISSING"
    m = re.match(r'^([A-Z]+)', str(principle_id))
    return m.group(1) if m else "UNKNOWN"

def extract_question_id(principle_id):
    """Try to extract the numeric question part from principle_id."""
    if not principle_id:
        return None
    m = re.search(r'(\d+)', str(principle_id))
    return int(m.group(1)) if m else None

def count_words(text):
    if not text:
        return 0
    return len(str(text).split())

def has_number(text):
    """Check if text contains any numeric value (integer, float, percentage, fraction)."""
    if not text:
        return False
    return bool(re.search(r'\d+\.?\d*\s*%?|\d+/\d+', str(text)))

def check_boundary_format(bc):
    """Check boundary_conditions format: each item should contain colon-separated condition:value."""
    if not bc:
        return False, "empty"
    if isinstance(bc, list):
        items = bc
    elif isinstance(bc, str):
        items = [bc]
    else:
        return False, f"wrong type: {type(bc)}"
    
    passed = []
    failed = []
    for item in items:
        item_str = str(item)
        # Accept formats like "条件:数值", "condition: value", or "key:value"
        if re.search(r'[：:].+', item_str):
            passed.append(item_str)
        else:
            failed.append(item_str)
    
    if failed:
        return False, f"{len(failed)}/{len(items)} items missing colon separator"
    return True, "ok"

def main(filepath):
    print(f"\n{'='*70}")
    print(f"  Stage 3 Quality Check Report")
    print(f"  File: {filepath}")
    print(f"{'='*70}\n")

    # ── 1. Load ────────────────────────────────────────────────────────────────
    records, parse_errors = load_jsonl(filepath)
    print(f"📂 加载结果: {len(records)} 条记录", end="")
    if parse_errors:
        print(f"  ⚠️  {len(parse_errors)} 行解析失败: {parse_errors[:3]}")
    else:
        print("  ✅ 无解析错误")

    # ── 2. Domain 分布 ────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("📊 [1] Domain 分布统计")
    print(f"{'─'*70}")
    
    domain_counter = Counter()
    invalid_domain_records = []
    
    for r in records:
        pid = r.get("principle_id", "")
        domain = extract_domain(pid)
        domain_counter[domain] += 1
        if domain not in VALID_DOMAINS:
            invalid_domain_records.append(pid)
    
    # Print valid domains
    print(f"\n  {'Domain':<8} {'Count':>6}  {'Bar'}")
    print(f"  {'──────':<8} {'─────':>6}  {'───'}")
    for domain in sorted(VALID_DOMAINS):
        count = domain_counter.get(domain, 0)
        bar = "█" * count if count <= 50 else "█" * 50 + f"…+{count-50}"
        print(f"  {domain:<8} {count:>6}  {bar}")
    
    # Print invalid domains
    invalid_domains = {d: c for d, c in domain_counter.items() if d not in VALID_DOMAINS}
    if invalid_domains:
        print(f"\n  ⚠️  非标准 Domain（需审查）:")
        for d, c in sorted(invalid_domains.items(), key=lambda x: -x[1]):
            print(f"     {d:<8} {c:>4} 条")
        print(f"\n  涉及的 principle_id 列表（前50条）:")
        for pid in invalid_domain_records[:50]:
            print(f"     - {pid}")
        if len(invalid_domain_records) > 50:
            print(f"     ... 还有 {len(invalid_domain_records)-50} 条")
    else:
        print(f"\n  ✅ 所有记录 domain 均在合法范围内")

    # ── 3. 缺失题目 ───────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("🔍 [2] 缺失题目检查（问题覆盖率）")
    print(f"{'─'*70}")
    
    # Try to find question IDs — look for source_question_id, question_id, or extract from principle_id
    covered_qids = set()
    for r in records:
        qid = r.get("source_question_id") or r.get("question_id") or r.get("question_number")
        if qid:
            try:
                covered_qids.add(int(qid))
            except (ValueError, TypeError):
                covered_qids.add(str(qid))
        else:
            # Try to extract from principle_id
            extracted = extract_question_id(r.get("principle_id", ""))
            if extracted:
                covered_qids.add(extracted)
    
    all_qids_int = {q for q in covered_qids if isinstance(q, int)}
    
    if all_qids_int:
        expected = set(range(1, TOTAL_QUESTIONS + 1))
        missing = sorted(expected - all_qids_int)
        print(f"\n  期望题数: {TOTAL_QUESTIONS}")
        print(f"  已覆盖题数: {len(all_qids_int)}")
        print(f"  缺失题数: {len(missing)}")
        if missing:
            print(f"\n  ❌ 缺失的题目编号:")
            for qid in missing:
                print(f"     - Q{qid:03d}")
        else:
            print(f"  ✅ 所有题目均已覆盖")
    else:
        # Fallback: count unique records per some key
        print(f"\n  ⚠️  未找到标准 question_id 字段，尝试从 principle_id 提取...")
        print(f"  当前 records 数量: {len(records)}")
        print(f"  期望: {TOTAL_QUESTIONS}")
        diff = TOTAL_QUESTIONS - len(records)
        if diff > 0:
            print(f"  ⚠️  记录数比期望少 {diff} 条（但无法精确定位缺失题号）")
        
        # Show unique question-like IDs found
        unique_ids = sorted(all_qids_int)[:10] if all_qids_int else []
        print(f"  找到的字段示例（抽样）:")
        for r in records[:3]:
            print(f"     {list(r.keys())}")

    # ── 4. 常规质量检查 ───────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("🔬 [3] 字段质量检查")
    print(f"{'─'*70}")
    
    total = len(records)
    
    checks = {
        "scientific_statement 含数值": [],
        "boundary_conditions 格式正确": [],
        "citation_quote < 30词": [],
        "_thinking 非空": [],
    }
    
    flagged_records = []  # Records needing manual review
    
    for r in records:
        pid = r.get("principle_id", "?")
        flags = []
        
        # Check 1: scientific_statement contains number
        ss = r.get("scientific_statement", "")
        if has_number(ss):
            checks["scientific_statement 含数值"].append(True)
        else:
            checks["scientific_statement 含数值"].append(False)
            flags.append("scientific_statement 无数值")
        
        # Check 2: boundary_conditions format
        bc = r.get("boundary_conditions", [])
        ok, reason = check_boundary_format(bc)
        checks["boundary_conditions 格式正确"].append(ok)
        if not ok:
            flags.append(f"boundary_conditions 格式问题({reason})")
        
        # Check 3: citation_quote word count
        cq = r.get("citation_quote", "")
        wc = count_words(cq)
        if wc <= CITATION_MAX_WORDS and wc > 0:
            checks["citation_quote < 30词"].append(True)
        else:
            checks["citation_quote < 30词"].append(False)
            if wc == 0:
                flags.append("citation_quote 为空")
            else:
                flags.append(f"citation_quote 过长({wc}词)")
        
        # Check 4: _thinking not empty
        thinking = r.get("_thinking", "")
        if thinking and str(thinking).strip():
            checks["_thinking 非空"].append(True)
        else:
            checks["_thinking 非空"].append(False)
            flags.append("_thinking 为空")
        
        if flags:
            flagged_records.append((pid, flags))
    
    # Print check results
    print(f"\n  {'检查项':<30} {'通过':>6} {'失败':>6} {'通过率':>8}")
    print(f"  {'──────':<30} {'──':>6} {'──':>6} {'──':>8}")
    for check_name, results in checks.items():
        passed = sum(results)
        failed = len(results) - passed
        rate = passed / len(results) * 100 if results else 0
        status = "✅" if rate >= 90 else ("⚠️ " if rate >= 70 else "❌")
        print(f"  {status} {check_name:<28} {passed:>6} {failed:>6} {rate:>7.1f}%")
    
    # ── 5. 需人工复查清单 ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("📋 [4] 需人工复查的原理列表")
    print(f"{'─'*70}")
    print(f"\n  共 {len(flagged_records)} / {total} 条需要复查\n")
    
    if flagged_records:
        # Group by issue type
        issue_groups = defaultdict(list)
        for pid, flags in flagged_records:
            for flag in flags:
                # Normalize flag to category
                cat = flag.split("(")[0].strip()
                issue_groups[cat].append(pid)
        
        print("  按问题类型汇总:")
        for issue, pids in sorted(issue_groups.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{len(pids)}条] {issue}")
            for pid in pids[:20]:
                print(f"     - {pid}")
            if len(pids) > 20:
                print(f"     ... 还有 {len(pids)-20} 条")
    
    # ── 6. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("📊 总结")
    print(f"{'='*70}")
    overall_pass = sum(1 for r in flagged_records if not r[1]) 
    print(f"  总记录数: {total}")
    print(f"  无问题记录: {total - len(flagged_records)}")
    print(f"  需复查记录: {len(flagged_records)}")
    print(f"  漂移 domain 数: {len(invalid_domains)}")
    print(f"\n  完成时间: " + __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 stage3_qc.py /path/to/l0_principles.jsonl")
        sys.exit(1)
    main(sys.argv[1])
