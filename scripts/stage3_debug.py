#!/usr/bin/env python3
"""
调试：打印 stage2 三个文件 + stage1 的实际结构
"""
import json

FILES = {
    "stage2/question_chunk_matches.json": "/Users/jeff/l0-knowledge-engine/output/stage2/question_chunk_matches.json",
    "stage2/chunk_meta.json":             "/Users/jeff/l0-knowledge-engine/output/stage2/chunk_meta.json",
    "stage1/chunks_smart.json":           "/Users/jeff/l0-knowledge-engine/output/stage1/chunks_smart.json",
}

def peek(obj, depth=0, max_depth=2, label=""):
    pad = "  " * depth
    if depth > max_depth:
        print(f"{pad}  ...")
        return
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{pad}{label}dict  keys={keys[:12]}")
        for k in keys[:3]:
            peek(obj[k], depth+1, max_depth, label=f"['{k}'] → ")
    elif isinstance(obj, list):
        print(f"{pad}{label}list  len={len(obj)}")
        if obj:
            peek(obj[0], depth+1, max_depth, label="[0] → ")
    else:
        print(f"{pad}{label}{type(obj).__name__} = {str(obj)[:120]}")

for label, path in FILES.items():
    print("\n" + "=" * 64)
    print(f"📄 {label}")
    print("=" * 64)
    try:
        with open(path) as f:
            data = json.load(f)
        peek(data)

        print("\n── 第一个元素完整内容 ──")
        if isinstance(data, list):
            first = data[0]
        elif isinstance(data, dict):
            first_key = list(data.keys())[0]
            print(f"  (key = '{first_key}')")
            first = data[first_key]
        else:
            first = data
        print(json.dumps(first, ensure_ascii=False, indent=2)[:1200])

    except FileNotFoundError:
        print(f"  ❌ 文件不存在: {path}")
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
