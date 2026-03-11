#!/usr/bin/env python3
"""
Stage 1: 文本标注 - 使用本地 qwen3:14b
读取章节文本，切分 chunks，标注科学内容

Usage:
    python stage1_text.py --test     # 测试 2 个章节
    python stage1_text.py            # 全部章节
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    import ollama
except ImportError:
    print("请安装 ollama: pip install ollama")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    "model": "qwen3:14b",
    "chapters_dir": Path("./books/ofc/chapters"),
    "output_dir": Path("./output/stage1"),
    "annotations_file": Path("./output/stage1/chunk_annotations.json"),
    
    # Chunk 设置
    "chunk_size": 2000,  # 字符数
    "chunk_overlap": 200,
    
    # 跳过的章节（版权页、目录等）
    "skip_chapters": [1, 2, 30],  # 根据实际内容调整
    
    # 14 个领域
    "domains": [
        "heat_transfer", "chemical_reaction", "physical_change", 
        "water_activity", "protein_science", "lipid_science",
        "carbohydrate", "enzyme", "flavor_sensory", "fermentation",
        "food_safety", "emulsion_colloid", "color_pigment", "equipment_physics"
    ],
}

# ============================================================
# Prompts
# ============================================================

ANNOTATION_PROMPT = """请分析这段食品科学书籍内容，提取以下信息。

## 文本内容
{text}

## 任务
1. 判断是否有食品科学内容
2. 标记相关领域
3. 提取关键概念

## 领域列表 (只能从以下选择)
- heat_transfer: 热传递/温度控制（烹饪温度、导热、辐射热等）
- chemical_reaction: 化学反应（美拉德反应、焦糖化、氧化等）
- physical_change: 物理变化/相变（冰晶、蒸发、溶解等）
- water_activity: 水分活度（Aw、水分含量、干燥等）
- protein_science: 蛋白质科学（变性、凝固、胶原蛋白等）
- lipid_science: 油脂科学（脂肪酸、酸败、乳化等）
- carbohydrate: 糖类/淀粉（糊化、老化、结晶等）
- enzyme: 酶反应（酶促褐变、蛋白酶等）
- flavor_sensory: 风味/感官（香气、味道、口感等）
- fermentation: 发酵（酵母、乳酸菌、发酵产物等）
- food_safety: 食品安全（细菌、温度危险区、保存等）
- emulsion_colloid: 乳化/胶体（乳化剂、凝胶、泡沫等）
- color_pigment: 色素变化（叶绿素、肌红蛋白、褐变等）
- equipment_physics: 设备原理（微波、压力锅、烤箱等）

## 输出格式 (纯 JSON，无其他文字)
```json
{{
  "has_scientific_content": true或false,
  "topics": ["domain1", "domain2"],
  "key_concepts": ["concept1", "concept2", "concept3"],
  "scientific_facts": [
    "具体的科学事实1（包含数值或可验证条件）",
    "具体的科学事实2"
  ],
  "brief_summary": "一句话描述这段内容"
}}
```

只输出 JSON，不要其他文字。"""

# ============================================================
# Text Processor
# ============================================================

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list:
    """将文本切分为 chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 尝试在句号处切分
        if end < len(text):
            # 找最近的句号
            for sep in ['. ', '。', '.\n', '\n\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size * 0.7:
                    end = start + last_sep + len(sep)
                    break
        
        chunk_text = text[start:end].strip()
        if len(chunk_text) > 100:  # 忽略太短的
            chunks.append({
                "start": start,
                "end": end,
                "text": chunk_text,
            })
        
        start = end - overlap
    
    return chunks


class TextAnnotator:
    def __init__(self):
        self.annotations = []
        self.stats = {
            "total_chunks": 0,
            "annotated": 0,
            "errors": 0,
            "start_time": datetime.now().isoformat(),
        }
        self.load_existing()
    
    def load_existing(self):
        """加载已有标注"""
        if CONFIG["annotations_file"].exists():
            with open(CONFIG["annotations_file"], "r", encoding="utf-8") as f:
                data = json.load(f)
                self.annotations = data.get("chunks", [])
                self.stats = data.get("stats", self.stats)
            print(f"📂 加载已有标注: {len(self.annotations)} chunks")
    
    def save(self):
        """保存标注"""
        CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
        
        output = {
            "stats": self.stats,
            "chunks": self.annotations,
        }
        
        with open(CONFIG["annotations_file"], "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    
    def annotate_chunk(self, chunk: dict, chapter_num: int, chunk_idx: int) -> dict:
        """标注单个 chunk"""
        try:
            response = ollama.chat(
                model=CONFIG["model"],
                messages=[
                    {
                        "role": "user",
                        "content": ANNOTATION_PROMPT.format(text=chunk["text"][:3000]),
                    }
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 1024,
                }
            )
            
            content = response["message"]["content"]
            
            # 解析 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                # 尝试找 { } 之间的内容
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                else:
                    json_str = content.strip()
            
            annotation = json.loads(json_str)
            annotation["chapter_num"] = chapter_num
            annotation["chunk_idx"] = chunk_idx
            annotation["char_start"] = chunk["start"]
            annotation["char_end"] = chunk["end"]
            annotation["text_preview"] = chunk["text"][:200]
            annotation["annotated_at"] = datetime.now().isoformat()
            
            # 验证 topics
            if "topics" in annotation:
                annotation["topics"] = [
                    t for t in annotation["topics"] 
                    if t in CONFIG["domains"]
                ]
            
            return annotation
            
        except json.JSONDecodeError as e:
            return {
                "chapter_num": chapter_num,
                "chunk_idx": chunk_idx,
                "error": f"JSON parse error: {e}",
                "raw_response": content[:300] if 'content' in dir() else None,
            }
        except Exception as e:
            return {
                "chapter_num": chapter_num,
                "chunk_idx": chunk_idx,
                "error": str(e),
            }
    
    def process_chapter(self, chapter_path: Path, chapter_num: int):
        """处理单个章节"""
        with open(chapter_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # 切分 chunks
        chunks = split_into_chunks(
            text, 
            CONFIG["chunk_size"], 
            CONFIG["chunk_overlap"]
        )
        
        print(f"\n  Chapter {chapter_num}: {len(chunks)} chunks, {len(text):,} chars")
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"ch{chapter_num:02d}-{i:03d}"
            
            # 检查是否已处理
            if any(a.get("chapter_num") == chapter_num and a.get("chunk_idx") == i 
                   for a in self.annotations):
                print(f"    [{i+1}/{len(chunks)}] {chunk_id} - 已存在，跳过")
                continue
            
            print(f"    [{i+1}/{len(chunks)}] {chunk_id}...", end="", flush=True)
            
            annotation = self.annotate_chunk(chunk, chapter_num, i)
            annotation["chunk_id"] = chunk_id
            
            if "error" not in annotation:
                self.annotations.append(annotation)
                self.stats["annotated"] += 1
                
                has_sci = "✓" if annotation.get("has_scientific_content") else "✗"
                topics = annotation.get("topics", [])[:2]
                print(f" sci:{has_sci} | {topics}")
            else:
                self.stats["errors"] += 1
                print(f" ❌ {annotation.get('error', '')[:40]}")
            
            self.stats["total_chunks"] += 1
            
            # 每 5 个 chunk 保存一次
            if (i + 1) % 5 == 0:
                self.save()
            
            time.sleep(0.1)
        
        self.save()
    
    def run(self, test_mode: bool = False):
        """运行标注"""
        print(f"\n{'='*60}")
        print(f"🏷️  Stage 1: 文本标注 (qwen3:14b)")
        print(f"{'='*60}")
        print(f"模型: {CONFIG['model']}")
        print(f"Chunk 大小: {CONFIG['chunk_size']} 字符")
        
        # 加载章节索引
        index_path = CONFIG["chapters_dir"] / "index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
                chapters = index.get("chapters", [])
        else:
            # 自动发现章节文件
            chapters = [
                {"num": int(p.stem.replace("ch", "")), "file": p.name}
                for p in sorted(CONFIG["chapters_dir"].glob("ch*.txt"))
            ]
        
        print(f"章节数: {len(chapters)}")
        
        # 过滤要处理的章节
        if test_mode:
            # 测试模式：只跑 2 个章节
            chapters = [c for c in chapters if c.get("num", 0) in [7, 14]][:2]
            print(f"测试模式: 只处理 {[c.get('num') for c in chapters]}")
        else:
            chapters = [c for c in chapters if c.get("num", 0) not in CONFIG["skip_chapters"]]
        
        # 处理每个章节
        for chapter in chapters:
            chapter_num = chapter.get("num", 0)
            chapter_file = chapter.get("file", f"ch{chapter_num:03d}.txt")
            chapter_path = CONFIG["chapters_dir"] / chapter_file
            
            if not chapter_path.exists():
                print(f"\n  Chapter {chapter_num}: ⚠️ 文件不存在")
                continue
            
            self.process_chapter(chapter_path, chapter_num)
        
        # 最终保存
        self.stats["end_time"] = datetime.now().isoformat()
        self.save()
        
        # 打印统计
        self.print_summary()
    
    def print_summary(self):
        """打印统计摘要"""
        print(f"\n{'='*60}")
        print(f"📊 标注完成")
        print(f"{'='*60}")
        print(f"总 Chunks: {len(self.annotations)}")
        print(f"错误数: {self.stats['errors']}")
        
        # 有科学内容的 chunks
        sci_count = sum(1 for a in self.annotations if a.get("has_scientific_content"))
        print(f"有科学内容: {sci_count} chunks ({sci_count*100/max(1,len(self.annotations)):.1f}%)")
        
        # 领域分布
        topic_counts = {}
        for a in self.annotations:
            for t in a.get("topics", []):
                topic_counts[t] = topic_counts.get(t, 0) + 1
        
        print(f"\n领域分布:")
        for t, c in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {t}: {c}")
        
        print(f"\n输出文件: {CONFIG['annotations_file']}")
        print(f"\n下一步: python qg1_check.py")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: 文本标注")
    parser.add_argument("--test", action="store_true", help="测试模式 (2 章节)")
    args = parser.parse_args()
    
    annotator = TextAnnotator()
    annotator.run(test_mode=args.test)


if __name__ == "__main__":
    main()
