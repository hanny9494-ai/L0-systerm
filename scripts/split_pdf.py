#!/usr/bin/env python3
"""
PDF 切分脚本 - On Food and Cooking
将 PDF 按页切分为 PNG 图片

Usage:
    pip install pdf2image Pillow
    # macOS 需要安装 poppler: brew install poppler
    
    python split_pdf.py
"""

import os
import sys
from pathlib import Path

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("请先安装依赖:")
    print("  pip install pdf2image Pillow")
    print("  brew install poppler  # macOS")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================

# PDF 路径
PDF_PATH = Path("/Users/jeff/Documents/厨书（待转换）/_OceanofPDF.com_On_Food_and_Cooking_The_Science_and_Lore_of_the_Kitchen_-_Harold_McGee")

# 输出目录
OUTPUT_BASE = Path("/Users/jeff/l0-knowledge-engine")

# 章节页码范围 (On Food and Cooking 2nd Edition)
CHAPTERS = {
    "ch14_food_molecules": (843, 882),   # Ch14 - 食品分子基础 (试跑优先)
    "ch13_cooking_methods": (777, 842),  # Ch13 - 烹饪方法
    "ch03_meat": (118, 178),             # Ch3 - 肉类
    "ch02_eggs": (68, 108),              # Ch2 - 蛋类
    "ch09_bread": (521, 579),            # Ch9 - 面包
    "ch12_fats_oils": (707, 776),        # Ch12 - 油脂
}

# 试跑页面 (Ch14 全部 + Ch13 部分)
PILOT_PAGES = list(range(843, 883)) + list(range(777, 787))  # 50 页


def find_pdf(base_path: Path) -> Path:
    """找到 PDF 文件"""
    base_path = Path(base_path)
    
    # 如果直接是 PDF 文件
    if base_path.suffix.lower() == '.pdf' and base_path.exists():
        return base_path
    
    # 如果是目录，找里面的 PDF
    if base_path.is_dir():
        pdfs = list(base_path.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
    
    # 尝试加 .pdf 后缀
    pdf_path = Path(str(base_path) + ".pdf")
    if pdf_path.exists():
        return pdf_path
    
    raise FileNotFoundError(f"找不到 PDF 文件: {base_path}")


def split_pages(pdf_path: Path, pages: list[int], output_dir: Path, dpi: int = 150):
    """切分指定页面"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"PDF: {pdf_path}")
    print(f"输出: {output_dir}")
    print(f"页面: {len(pages)} 页")
    print(f"DPI: {dpi}")
    print()
    
    for i, page_num in enumerate(pages):
        output_file = output_dir / f"p{page_num:04d}.png"
        
        if output_file.exists():
            print(f"  [{i+1}/{len(pages)}] p{page_num} - 已存在，跳过")
            continue
        
        print(f"  [{i+1}/{len(pages)}] p{page_num} - 转换中...", end="", flush=True)
        
        try:
            # 只转换指定页
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
            )
            
            if images:
                images[0].save(output_file, "PNG")
                print(f" ✓ {output_file.name}")
            else:
                print(" ✗ 无内容")
                
        except Exception as e:
            print(f" ✗ 错误: {e}")
    
    print(f"\n完成！输出目录: {output_dir}")


def main():
    print("="*60)
    print("On Food and Cooking - PDF 切分工具")
    print("="*60)
    
    # 找到 PDF
    try:
        pdf_path = find_pdf(PDF_PATH)
        print(f"找到 PDF: {pdf_path}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请检查路径是否正确，或手动指定 PDF 文件路径")
        sys.exit(1)
    
    # 创建项目目录
    project_dir = OUTPUT_BASE
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)
    (project_dir / "books" / "ofc").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "-"*60)
    print("选择操作:")
    print("  1. 切分试跑页面 (50页, Ch14+Ch13部分)")
    print("  2. 切分 Ch14 全章 (40页)")
    print("  3. 切分全部章节")
    print("  4. 自定义页码范围")
    print("-"*60)
    
    choice = input("输入选项 [1]: ").strip() or "1"
    
    if choice == "1":
        output_dir = project_dir / "books" / "ofc" / "pilot_pages"
        split_pages(pdf_path, PILOT_PAGES, output_dir)
        
    elif choice == "2":
        start, end = CHAPTERS["ch14_food_molecules"]
        pages = list(range(start, end + 1))
        output_dir = project_dir / "books" / "ofc" / "ch14"
        split_pages(pdf_path, pages, output_dir)
        
    elif choice == "3":
        for chapter_name, (start, end) in CHAPTERS.items():
            pages = list(range(start, end + 1))
            output_dir = project_dir / "books" / "ofc" / chapter_name
            print(f"\n--- {chapter_name} ---")
            split_pages(pdf_path, pages, output_dir)
            
    elif choice == "4":
        start = int(input("起始页码: "))
        end = int(input("结束页码: "))
        pages = list(range(start, end + 1))
        output_dir = project_dir / "books" / "ofc" / f"pages_{start}_{end}"
        split_pages(pdf_path, pages, output_dir)
    
    else:
        print("无效选项")
        sys.exit(1)
    
    # 输出下一步指引
    print("\n" + "="*60)
    print("下一步:")
    print("="*60)
    print(f"""
1. 复制脚本和数据到项目目录:
   cp l0_extract.py {project_dir}/
   cp pilot_20_questions.json {project_dir}/data/

2. 安装 Python 依赖:
   pip install anthropic

3. 运行试跑:
   cd {project_dir}
   python l0_extract.py pilot

4. 查看统计:
   python l0_extract.py stats
""")


if __name__ == "__main__":
    main()
