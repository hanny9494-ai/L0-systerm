#!/usr/bin/env python3
"""
EPUB 处理脚本 - On Food and Cooking
从 EPUB 提取章节文本

Usage:
    pip install ebooklib beautifulsoup4
    python process_epub.py
"""

import os
import sys
import json
import re
from pathlib import Path

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError:
    print("请先安装依赖:")
    print("  pip install ebooklib beautifulsoup4")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================

EPUB_PATH = Path("/Users/jeff/Documents/厨书（待转换）/_OceanofPDF.com_On_Food_and_Cooking_The_Science_and_Lore_of_the_Kitchen_-_Harold_McGee.epub")

OUTPUT_BASE = Path("/Users/jeff/l0-knowledge-engine")

# ============================================================
# EPUB Processing
# ============================================================

def extract_text_from_html(html_content: bytes) -> str:
    """从 HTML 内容中提取纯文本"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 移除脚本和样式
    for tag in soup(['script', 'style']):
        tag.decompose()
    
    # 获取文本
    text = soup.get_text(separator='\n')
    
    # 清理空行
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    return text


def extract_chapters(epub_path: Path) -> list[dict]:
    """提取所有章节"""
    book = epub.read_epub(str(epub_path))
    
    chapters = []
    chapter_num = 0
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content()
            text = extract_text_from_html(content)
            
            if len(text) > 500:  # 忽略太短的内容（目录、版权页等）
                chapter_num += 1
                
                # 尝试从内容中提取标题
                lines = text.split('\n')
                title = lines[0][:100] if lines else f"Chapter {chapter_num}"
                
                chapters.append({
                    "chapter_num": chapter_num,
                    "title": title,
                    "item_name": item.get_name(),
                    "text": text,
                    "char_count": len(text),
                })
                
                print(f"  [{chapter_num}] {title[:60]}... ({len(text):,} chars)")
    
    return chapters


def save_chapters(chapters: list[dict], output_dir: Path):
    """保存章节到单独文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个章节为单独文件
    for ch in chapters:
        filename = f"ch{ch['chapter_num']:03d}.txt"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {ch['title']}\n\n")
            f.write(ch['text'])
    
    # 保存索引
    index = {
        "source": str(EPUB_PATH),
        "total_chapters": len(chapters),
        "chapters": [
            {
                "num": ch['chapter_num'],
                "title": ch['title'],
                "file": f"ch{ch['chapter_num']:03d}.txt",
                "char_count": ch['char_count'],
            }
            for ch in chapters
        ]
    }
    
    with open(output_dir / "index.json", 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 保存 {len(chapters)} 个章节到 {output_dir}")
    print(f"✓ 索引文件: {output_dir / 'index.json'}")


def create_pilot_chunks(chapters: list[dict], output_dir: Path, chunk_size: int = 3000):
    """创建试跑用的文本块"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 关键词匹配 - 用于找到与试跑问题相关的章节
    keywords = {
        "maillard": ["maillard", "browning", "amino acid", "reducing sugar"],
        "caramelization": ["carameliz", "sugar", "pyrolysis"],
        "protein": ["protein", "denatur", "coagulat", "collagen", "gelatin"],
        "starch": ["starch", "gelatiniz", "retrogradation", "amylose", "amylopectin"],
        "emulsion": ["emulsion", "emulsif", "lecithin", "mayonnaise"],
        "water_activity": ["water activity", "aw", "moisture"],
        "fat": ["fat", "oil", "saturated", "unsaturated", "rancid", "oxidation"],
        "heat": ["heat", "temperature", "cooking", "boiling", "frying"],
        "flavor": ["flavor", "taste", "aroma", "capsaicin", "spicy"],
    }
    
    pilot_chunks = []
    chunk_id = 0
    
    for ch in chapters:
        text_lower = ch['text'].lower()
        
        # 检查是否包含关键词
        matched_topics = []
        for topic, kws in keywords.items():
            if any(kw in text_lower for kw in kws):
                matched_topics.append(topic)
        
        if matched_topics:
            # 将章节切成小块
            text = ch['text']
            for i in range(0, len(text), chunk_size):
                chunk_id += 1
                chunk_text = text[i:i+chunk_size]
                
                # 确保不在单词中间切断
                if i + chunk_size < len(text):
                    last_space = chunk_text.rfind(' ')
                    if last_space > chunk_size * 0.8:
                        chunk_text = chunk_text[:last_space]
                
                pilot_chunks.append({
                    "chunk_id": chunk_id,
                    "chapter_num": ch['chapter_num'],
                    "chapter_title": ch['title'],
                    "topics": matched_topics,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                })
    
    # 保存
    with open(output_dir / "pilot_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(pilot_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 创建 {len(pilot_chunks)} 个文本块用于试跑")
    print(f"✓ 文件: {output_dir / 'pilot_chunks.json'}")
    
    return pilot_chunks


def main():
    print("="*60)
    print("On Food and Cooking - EPUB 处理工具")
    print("="*60)
    
    if not EPUB_PATH.exists():
        print(f"错误: 找不到文件 {EPUB_PATH}")
        sys.exit(1)
    
    print(f"EPUB: {EPUB_PATH}")
    print(f"输出: {OUTPUT_BASE}")
    
    # 创建目录
    books_dir = OUTPUT_BASE / "books" / "ofc"
    books_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- 提取章节 ---")
    chapters = extract_chapters(EPUB_PATH)
    
    print(f"\n共提取 {len(chapters)} 个章节")
    
    # 保存章节
    save_chapters(chapters, books_dir / "chapters")
    
    # 创建试跑文本块
    print("\n--- 创建试跑文本块 ---")
    create_pilot_chunks(chapters, books_dir / "pilot_chunks")
    
    print("\n" + "="*60)
    print("完成！下一步:")
    print("="*60)
    print(f"""
1. 检查提取结果:
   ls {books_dir}/chapters/
   cat {books_dir}/pilot_chunks/pilot_chunks.json | head -100

2. 运行抽取:
   cd {OUTPUT_BASE}
   python l0_extract.py pilot
""")


if __name__ == "__main__":
    main()
