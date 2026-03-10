#!/usr/bin/env python3
"""
公平对比：MinerU vs qwen3-vl-plus
两边跑同一份数据：ch13_thickeners.pdf 的 54 张页面截图

MinerU  → 已有 full.md，按标题块分析
qwen-vl → 直接识别每页 PNG，输出 Markdown
"""

import os
import re
import json
import time
import base64
from pathlib import Path
from openai import OpenAI

# ──────────────────────────────────────────────
MODEL      = 'qwen3-vl-plus'
PAGES_DIR  = Path('/Users/jeff/l0-knowledge-engine/output/comparison/pages')
MINERU_MD  = Path('/Users/jeff/l0-knowledge-engine/output/comparison/mineru/full.md')
OUTPUT_DIR = Path('/Users/jeff/l0-knowledge-engine/output/comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = """请完整提取这一页的所有内容，输出 Markdown 格式：

- 正文：直接输出文字，保留段落结构
- 表格：用标准 Markdown 表格（| 列1 | 列2 |），保留所有数值和单位
- 图表/示意图：用1-2句话描述类型和关键信息
- 操作步骤图/照片：简述内容

不要加页码或额外说明，直接输出内容。"""

# ──────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)


def analyze_block(text: str) -> dict:
    tables = re.findall(r'(?:^\|.+\n)+', text, re.MULTILINE)
    imgs   = re.findall(r'!\[.*?\]\(.*?\)', text)
    words  = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b', text.lower())
    temps  = re.findall(r'\d+\.?\d*\s*°\s*[FC]', text, re.I)
    times  = re.findall(r'\d+\.?\d*\s*(?:second|minute|hour|min)s?', text, re.I)
    return {
        'table_count':  len(tables),
        'img_refs':     len(imgs),
        'word_count':   len(words),
        'unique_words': len(set(words)),
        'temps':        temps,
        'times':        times,
        'has_table':    len(tables) > 0,
    }


def split_mineru(md_path: Path) -> list[str]:
    text  = md_path.read_text(encoding='utf-8')
    parts = re.split(r'\n(?=#{1,2} )', text)
    parts = [p.strip() for p in parts if p.strip()]
    print(f'  MinerU: {len(parts)} 个内容块')
    return parts


def recognize_page(img_path: Path) -> tuple[str, float]:
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                {'type': 'text', 'text': PROMPT}
            ]
        }],
        max_tokens=2048
    )
    return resp.choices[0].message.content, round(time.time() - t0, 2)


def main():
    pages = sorted(PAGES_DIR.glob('p*.png'))
    total = len(pages)
    print(f'{"="*60}')
    print(f'公平对比：MinerU vs qwen3-vl-plus')
    print(f'PDF 页数: {total}  模型: {MODEL}')
    print(f'{"="*60}\n')

    # ── MinerU 侧 ──
    print('▶ 解析 MinerU full.md...')
    mineru_blocks = split_mineru(MINERU_MD)
    mineru_stats  = [analyze_block(b) for b in mineru_blocks]

    # ── qwen-vl 侧 ──
    print(f'\n▶ qwen-vl 逐页识别 ({total} 页)...\n')
    qwen_results = []
    for i, img in enumerate(pages):
        print(f'  [{i+1:02d}/{total}] {img.name}', end=' ', flush=True)
        try:
            text, elapsed = recognize_page(img)
            stats = analyze_block(text)
            print(f'{elapsed:.1f}s  表格:{stats["table_count"]}  词:{stats["word_count"]}')
            qwen_results.append({
                'page': i, 'file': img.name,
                'elapsed_sec': elapsed, 'text': text, 'stats': stats,
            })
        except Exception as e:
            print(f'❌ {e}')
            qwen_results.append({'page': i, 'file': img.name, 'error': str(e)})
        time.sleep(0.3)

    # ── 保存原始数据 ──
    raw = {
        'mineru': [{'block': i, 'text': b, 'stats': s}
                   for i, (b, s) in enumerate(zip(mineru_blocks, mineru_stats))],
        'qwen': qwen_results,
    }
    (OUTPUT_DIR / 'raw_results.json').write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding='utf-8')

    # ── 汇总统计 ──
    qwen_ok  = [r for r in qwen_results if 'error' not in r]
    m_tables = sum(s['table_count'] for s in mineru_stats)
    q_tables = sum(r['stats']['table_count'] for r in qwen_ok)
    m_imgs   = sum(s['img_refs'] for s in mineru_stats)
    m_words  = sum(s['word_count'] for s in mineru_stats)
    q_words  = sum(r['stats']['word_count'] for r in qwen_ok)
    m_temps  = sum(len(s['temps']) for s in mineru_stats)
    q_temps  = sum(len(r['stats']['temps']) for r in qwen_ok)
    m_times  = sum(len(s['times']) for s in mineru_stats)
    q_times  = sum(len(r['stats']['times']) for r in qwen_ok)
    avg_time = sum(r['elapsed_sec'] for r in qwen_ok) / len(qwen_ok) if qwen_ok else 0
    table_pages = [r for r in qwen_ok if r['stats']['table_count'] > 0]

    # ── 报告 ──
    report = f"""# Chapter 13 Thickeners — 公平对比报告
> 输入：同一份 ch13_thickeners.pdf（{total} 页）
> MinerU：PDF 直接解析 → full.md
> qwen-vl：PDF 页面截图 → 逐页识别

## 量化对比

| 指标 | MinerU | qwen3-vl-plus |
|------|--------|---------------|
| 处理单元 | {len(mineru_blocks)} 个内容块 | {total} 页 |
| **表格识别** | **{m_tables}** | **{q_tables}** |
| 图片占位符 | {m_imgs} | 0（直接识别） |
| 总词数 | {m_words:,} | {q_words:,} |
| 温度数值 | {m_temps} | {q_temps} |
| 时间数值 | {m_times} | {q_times} |
| 平均耗时/页 | <1s | {avg_time:.1f}s |

## 各维度胜者

| 维度 | 胜者 |
|------|------|
| 表格识别 | {'qwen ✅' if q_tables > m_tables else 'MinerU ✅' if m_tables > q_tables else '平手'} |
| 数值提取（温度+时间） | {'qwen ✅' if q_temps+q_times > m_temps+m_times else 'MinerU ✅'} |
| 文字总量 | {'qwen ✅' if q_words > m_words else 'MinerU ✅'} |
| 速度 | MinerU ✅ |
| 成本 | MinerU ✅ |

---

## qwen-vl 含表格的页面（共 {len(table_pages)} 页）

"""
    for r in table_pages:
        report += f"### {r['file']}\n\n{r['text']}\n\n---\n\n"

    report += f"\n## MinerU 含表格内容块（共 {m_tables} 个）\n\n"
    mineru_table_blocks = [(i, b) for i, (b, s) in enumerate(zip(mineru_blocks, mineru_stats)) if s['table_count'] > 0]
    if mineru_table_blocks:
        for i, b in mineru_table_blocks:
            report += f"### 块 {i}\n\n{b[:2000]}\n\n---\n\n"
    else:
        report += "_MinerU 未识别出任何 Markdown 表格（表格以图片形式嵌入 PDF）_\n"

    out_md = OUTPUT_DIR / 'fair_comparison_report.md'
    out_md.write_text(report, encoding='utf-8')

    print(f"""
{'='*60}
结果汇总

             MinerU    qwen3-vl-plus
表格识别:    {m_tables:<10}{q_tables}
温度数值:    {m_temps:<10}{q_temps}
时间数值:    {m_times:<10}{q_times}
总词数:      {m_words:<10}{q_words}
平均耗时:    <1s       {avg_time:.1f}s

报告: {out_md}
{'='*60}
""")


if __name__ == '__main__':
    main()
