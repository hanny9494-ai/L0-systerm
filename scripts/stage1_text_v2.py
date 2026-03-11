#!/usr/bin/env python3
"""
Stage 1: 文本标注 - 使用本地 qwen3:14b
直接用 requests 调用，绕过代理问题
"""

import json
import requests
import time
import argparse
from pathlib import Path
from datetime import datetime

CONFIG = {
    "model": "qwen3:14b",
    "ollama_url": "http://127.0.0.1:11434/api/chat",
    "chapters_dir": Path("./books/ofc/chapters"),
    "output_dir": Path("./output/stage1"),
    "annotations_file": Path("./output/stage1/chunk_annotations.json"),
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "skip_chapters": [1, 2, 3, 4, 5, 6, 30],
    "domains": [
        "heat_transfer", "chemical_reaction", "physical_change", 
        "water_activity", "protein_science", "lipid_science",
        "carbohydrate", "enzyme", "flavor_sensory", "fermentation",
        "food_safety", "emulsion_colloid", "color_pigment", "equipment_physics"
    ],
}

PROMPT = """分析这段食品科学内容，输出 JSON：

{text}

只输出 JSON，格式：
{{"has_scientific_content": true/false, "topics": ["domain1"], "key_concepts": ["concept1"], "brief_summary": "一句话"}}

topics 只能从这些选：heat_transfer, chemical_reaction, physical_change, water_activity, protein_science, lipid_science, carbohydrate, enzyme, flavor_sensory, fermentation, food_safety, emulsion_colloid, color_pigment, equipment_physics"""

def call_ollama(prompt):
    resp = requests.post(
        CONFIG["ollama_url"],
        json={
            "model": CONFIG["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3}
        },
        proxies={"http": None, "https": None},
        timeout=120
    )
    if resp.status_code == 200:
        return resp.json()["message"]["content"]
    else:
        raise Exception(f"Status {resp.status_code}")

def split_chunks(text, size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append({"start": start, "end": end, "text": chunk})
        start = end - overlap
    return chunks

def parse_json(content):
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except:
        pass
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Stage 1: 文本标注 ({CONFIG['model']})")
    print(f"{'='*60}")
    
    # 加载章节
    chapters_dir = CONFIG["chapters_dir"]
    index_file = chapters_dir / "index.json"
    
    with open(index_file) as f:
        index = json.load(f)
    
    chapters = [c for c in index["chapters"] if c["num"] not in CONFIG["skip_chapters"]]
    
    if args.test:
        chapters = [c for c in chapters if c["num"] in [7, 8]][:2]
        print(f"测试模式: {[c['num'] for c in chapters]}")
    
    all_annotations = []
    
    for ch in chapters:
        ch_file = chapters_dir / ch["file"]
        with open(ch_file) as f:
            text = f.read()
        
        chunks = split_chunks(text)
        print(f"\nChapter {ch['num']}: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:5] if args.test else chunks):  # 测试只跑5个
            print(f"  [{i+1}] ch{ch['num']:02d}-{i:03d}...", end="", flush=True)
            
            try:
                prompt = PROMPT.format(text=chunk["text"][:1500])
                response = call_ollama(prompt)
                result = parse_json(response)
                
                if result:
                    result["chapter_num"] = ch["num"]
                    result["chunk_idx"] = i
                    result["text_preview"] = chunk["text"][:150]
                    all_annotations.append(result)
                    
                    sci = "✓" if result.get("has_scientific_content") else "✗"
                    topics = result.get("topics", [])[:2]
                    print(f" {sci} {topics}")
                else:
                    print(f" ⚠️ JSON解析失败")
                    
            except Exception as e:
                print(f" ❌ {e}")
            
            time.sleep(0.5)
    
    # 保存
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    with open(CONFIG["annotations_file"], "w") as f:
        json.dump({"chunks": all_annotations}, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 保存 {len(all_annotations)} 条标注")
    print(f"文件: {CONFIG['annotations_file']}")

if __name__ == "__main__":
    main()
