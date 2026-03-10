#!/usr/bin/env python3
"""
合并脚本：MinerU 文字 + qwen-vl 表格/图片识别 → 完整 MD
策略：
  - MinerU 提供文字段落（准确、完整）
  - qwen-vl 提供表格和图片描述（MinerU 无法识别的部分）
  - 按页顺序合并，qwen 内容插入对应位置
"""

import re
import json
from pathlib import Path

MINERU_MD  = Path('/Users/jeff/l0-knowledge-engine/output/comparison/mineru/full.md')
RAW_JSON   = Path('/Users/jeff/l0-knowledge-engine/output/comparison/raw_results.json')
OUTPUT_MD  = Path('/Users/jeff/l0-knowledge-engine/output/comparison/ch13_merged.md')


def load_qwen_results(raw_json: Path) -> list[dict]:
    data = json.loads(raw_json.read_text(encoding='utf-8'))
    return [r for r in data['qwen'] if 'error' not in r]


def extract_tables_from_qwen(text: str) -> list[str]:
    """从 qwen 输出中提取所有 Markdown 表格"""
    tables = re.findall(r'(?:^\|.+\n)+', text, re.MULTILINE)
    return [t.strip() for t in tables if len(t.strip().split('\n')) >= 2]


def extract_image_descriptions(text: str) -> list[str]:
    """提取图片描述段落（非表格、非正文标题的段落）"""
    # 去掉表格
    text_no_tables = re.sub(r'(?:^\|.+\n)+', '', text, flags=re.MULTILINE)
    # 找图片描述（通常包含photo/图/figure等关键词）
    descs = []
    for para in re.split(r'\n{2,}', text_no_tables):
        para = para.strip()
        if not para:
            continue
        if re.search(r'photo|图|figure|照片|image|shows?|depicts?|illustrat', para, re.I):
            descs.append(para)
    return descs


def build_mineru_sections(md_path: Path) -> list[dict]:
    """把 MinerU MD 切成段落块，保留原始顺序"""
    text  = md_path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)

    sections = []
    current  = []
    current_heading = ''

    for line in lines:
        if re.match(r'^#{1,2} ', line):
            if current:
                sections.append({
                    'heading': current_heading,
                    'text': ''.join(current).strip(),
                    'has_img_placeholder': '![](' in ''.join(current),
                })
            current_heading = line.strip()
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append({
            'heading': current_heading,
            'text': ''.join(current).strip(),
            'has_img_placeholder': '![](' in ''.join(current),
        })

    return sections


def merge(mineru_sections: list[dict], qwen_results: list[dict]) -> str:
    """
    合并策略：
    1. 遍历 MinerU 段落
    2. 遇到图片占位符 (![](...))，用 qwen 识别的对应内容替换
    3. qwen 识别的表格单独追加到对应段落后面
    """

    # 建立 qwen 图片哈希 → 内容的映射（从 MinerU MD 提取哈希）
    mineru_full = MINERU_MD.read_text(encoding='utf-8')
    img_hashes  = re.findall(r'images/([a-f0-9]+)\.jpg', mineru_full)

    # qwen 按页顺序，图片按 MinerU 顺序对应
    # 每页 qwen 结果可能对应多个 MinerU 图片占位符
    # 简化策略：把 qwen 所有表格和图片描述按顺序池化，按需取用
    qwen_tables = []
    qwen_descs  = []
    for r in qwen_results:
        tbls = extract_tables_from_qwen(r['text'])
        dscs = extract_image_descriptions(r['text'])
        for t in tbls:
            qwen_tables.append({'page': r['file'], 'content': t})
        for d in dscs:
            qwen_descs.append({'page': r['file'], 'content': d})

    print(f'  qwen 表格池: {len(qwen_tables)} 个')
    print(f'  qwen 图片描述池: {len(qwen_descs)} 个')

    # ── 逐段处理 MinerU ──
    output_parts = []
    table_idx = 0
    desc_idx  = 0

    for sec in mineru_sections:
        text = sec['text']

        if sec['has_img_placeholder']:
            # 统计这个段落有多少图片占位符
            placeholders = re.findall(r'!\[.*?\]\([^)]+\)', text)
            n_imgs = len(placeholders)

            # 替换占位符：先插入 qwen 识别内容
            new_text = text

            # 替换每个占位符
            for ph in placeholders:
                replacement_parts = []

                # 如果有可用的表格，插入表格
                if table_idx < len(qwen_tables):
                    replacement_parts.append(
                        f"\n<!-- qwen-vl 识别表格 (来自 {qwen_tables[table_idx]['page']}) -->\n"
                        + qwen_tables[table_idx]['content']
                    )
                    table_idx += 1
                # 如果有图片描述，插入描述
                elif desc_idx < len(qwen_descs):
                    replacement_parts.append(
                        f"\n> 📷 {qwen_descs[desc_idx]['content']}\n"
                    )
                    desc_idx += 1
                else:
                    # 保留原始占位符
                    replacement_parts.append(ph)

                replacement = '\n'.join(replacement_parts)
                new_text = new_text.replace(ph, replacement, 1)

            output_parts.append(new_text)
        else:
            # 无图片占位符，直接保留 MinerU 文字
            output_parts.append(text)

    # ── 追加未消耗的 qwen 表格 ──
    remaining_tables = qwen_tables[table_idx:]
    if remaining_tables:
        output_parts.append('\n\n---\n\n## 附录：qwen-vl 识别的额外表格\n')
        for t in remaining_tables:
            output_parts.append(f"\n<!-- {t['page']} -->\n{t['content']}\n")

    return '\n\n'.join(output_parts)


def main():
    print('=' * 60)
    print('合并 MinerU + qwen-vl → 完整 MD')
    print('=' * 60)

    print('\n▶ 加载数据...')
    mineru_sections = build_mineru_sections(MINERU_MD)
    qwen_results    = load_qwen_results(RAW_JSON)
    print(f'  MinerU 段落块: {len(mineru_sections)}')
    print(f'  qwen 页面结果: {len(qwen_results)}')

    img_sections = [s for s in mineru_sections if s['has_img_placeholder']]
    print(f'  含图片占位符的段落: {len(img_sections)}')

    print('\n▶ 合并中...')
    merged = merge(mineru_sections, qwen_results)

    # 清理多余空行
    merged = re.sub(r'\n{4,}', '\n\n\n', merged)

    OUTPUT_MD.write_text(merged, encoding='utf-8')

    # 统计合并后质量
    tables_in_merged = re.findall(r'(?:^\|.+\n)+', merged, re.MULTILINE)
    words_in_merged  = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b', merged.lower())
    imgs_remaining   = re.findall(r'!\[.*?\]\([^)]+\)', merged)

    print(f"""
▶ 合并结果
  文件大小:       {len(merged):,} 字符
  表格数:         {len(tables_in_merged)}
  总词数:         {len(words_in_merged):,}
  残留图片占位符: {len(imgs_remaining)}（无法被 qwen 覆盖的）

✅ 输出: {OUTPUT_MD}
""")


if __name__ == '__main__':
    main()
