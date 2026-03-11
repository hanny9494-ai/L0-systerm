#!/usr/bin/env python3
"""
PDF 视觉方案测试 - qwen3.5 多模态识别
测试目标: 评估 2b/9b 模型对 PDF 页面的表格识别能力
"""

import requests
import base64
import json
import time
import os
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
PAGES_DIR = Path("/Users/jeff/l0-knowledge-engine/books/ofc/pages")
OUTPUT_DIR = Path("/Users/jeff/l0-knowledge-engine/output/vision_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "2b": "qwen2.5vl:2b",   # 多模态视觉模型
    "9b": "qwen2.5vl:7b",   # 注意: ollama 上是 qwen2.5vl 系列
}

# 如果你用的是 qwen3 系列（需确认 ollama 中的确切名称）
# MODELS = {
#     "2b": "qwen3.5:2b",
#     "9b": "qwen3.5:9b",
# }

PROMPT_TABLE = """/no_think 请分析这页书籍内容，输出严格的JSON格式：
{
  "page_topic": "本页主题（一句话）",
  "has_table": true/false,
  "tables": [
    {
      "title": "表格标题",
      "headers": ["列名1", "列名2", ...],
      "rows": [["值1", "值2", ...], ...]
    }
  ],
  "key_values": {
    "temperatures": ["提取所有温度数值，如 165°F"],
    "times": ["提取所有时间数值，如 30 minutes"],
    "other_numbers": ["其他重要数值"]
  },
  "summary": "本页核心内容摘要（50字以内）"
}
只输出JSON，不要任何解释。"""


# ─────────────────────────────────────────────
# 步骤 1: PDF → 图片
# ─────────────────────────────────────────────
def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 150):
    """将 PDF 转换为页面图片"""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("❌ 需要安装 pdf2image: pip install pdf2image")
        print("   还需要 poppler: brew install poppler")
        return False

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"📄 正在转换 PDF: {pdf_path}")
    print(f"   DPI={dpi}, 输出目录: {output_dir}")

    images = convert_from_path(pdf_path, dpi=dpi)
    total = len(images)
    print(f"   共 {total} 页")

    for i, img in enumerate(images):
        out_path = out / f"p{i:04d}.png"
        img.save(str(out_path), "PNG")
        if i % 50 == 0:
            print(f"   已保存 {i+1}/{total} 页...")

    print(f"✅ 转换完成，共 {total} 张图片")
    return True


# ─────────────────────────────────────────────
# 步骤 2: 单页视觉识别
# ─────────────────────────────────────────────
def recognize_page(image_path: str, model: str, timeout: int = 120) -> dict:
    """调用 Ollama 对单页图片进行识别"""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": PROMPT_TABLE,
            "images": [img_b64]
        }],
        "stream": False,
        "options": {
            "temperature": 0.1,  # 低温度，提高结构化输出稳定性
            "num_predict": 1024,
        }
    }

    start = time.time()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            proxies={"http": None, "https": None},
            timeout=timeout
        )
        resp.raise_for_status()
        elapsed = time.time() - start
        raw = resp.json()["message"]["content"]

        # 尝试解析 JSON
        parsed = safe_parse_json(raw)
        return {
            "success": True,
            "model": model,
            "image": image_path,
            "elapsed_sec": round(elapsed, 2),
            "raw": raw,
            "parsed": parsed,
        }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "timeout", "model": model, "image": image_path}
    except Exception as e:
        return {"success": False, "error": str(e), "model": model, "image": image_path}


def safe_parse_json(text: str) -> Optional[dict]:
    """安全解析 JSON，处理模型可能输出的 markdown 代码块"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试找到第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except:
                pass
    return None


# ─────────────────────────────────────────────
# 步骤 3: 批量测试 + 对比
# ─────────────────────────────────────────────
def run_comparison_test(
    pages_dir: str,
    sample_pages: list[int],
    models: list[str],
    output_dir: str
):
    """
    对指定页面列表，用多个模型分别测试，输出对比报告。
    sample_pages: 页码索引列表，如 [0, 10, 50, 100]
    """
    pages_path = Path(pages_dir)
    out_path = Path(output_dir)

    results = {}
    summary = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_tested": models,
        "pages_tested": sample_pages,
        "model_stats": {}
    }

    for model in models:
        print(f"\n{'='*50}")
        print(f"🤖 测试模型: {model}")
        print(f"{'='*50}")

        model_results = []
        total_time = 0
        table_found = 0
        parse_success = 0

        for page_idx in sample_pages:
            img_file = pages_path / f"p{page_idx:04d}.png"
            if not img_file.exists():
                print(f"  ⚠️  页面不存在: {img_file}")
                continue

            print(f"  📖 识别第 {page_idx} 页...", end=" ", flush=True)
            result = recognize_page(str(img_file), model)

            if result["success"]:
                elapsed = result["elapsed_sec"]
                total_time += elapsed
                print(f"✅ {elapsed:.1f}s")

                if result["parsed"]:
                    parse_success += 1
                    p = result["parsed"]
                    if p.get("has_table"):
                        table_found += 1
                        tables = p.get("tables", [])
                        print(f"     📊 发现 {len(tables)} 个表格: {p.get('page_topic', '')}")
                    else:
                        print(f"     📝 无表格: {p.get('page_topic', '')}")
                else:
                    print(f"     ⚠️  JSON 解析失败")
            else:
                print(f"❌ 失败: {result.get('error')}")

            model_results.append(result)

        results[model] = model_results

        # 模型统计
        n = len([r for r in model_results if r["success"]])
        avg_time = total_time / n if n > 0 else 0
        summary["model_stats"][model] = {
            "pages_ok": n,
            "pages_failed": len(sample_pages) - n,
            "avg_time_sec": round(avg_time, 2),
            "table_detected": table_found,
            "json_parse_rate": f"{parse_success}/{n}" if n > 0 else "0/0",
        }

    # 保存详细结果
    detail_file = out_path / "vision_results_detail.json"
    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存汇总
    summary_file = out_path / "vision_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印对比报告
    print_comparison_report(summary)

    return summary, results


def print_comparison_report(summary: dict):
    """打印人类可读的对比报告"""
    print(f"\n{'='*60}")
    print("📊 模型对比报告")
    print(f"{'='*60}")
    print(f"测试时间: {summary['test_time']}")
    print(f"测试页数: {len(summary['pages_tested'])} 页")
    print()

    stats = summary["model_stats"]
    header = f"{'模型':<20} {'成功页':<8} {'平均耗时':<10} {'表格检出':<10} {'JSON解析率'}"
    print(header)
    print("-" * 60)

    for model, s in stats.items():
        row = (
            f"{model:<20} "
            f"{s['pages_ok']:<8} "
            f"{s['avg_time_sec']:.1f}s{'':<6} "
            f"{s['table_detected']:<10} "
            f"{s['json_parse_rate']}"
        )
        print(row)

    print()
    # 速度对比
    models = list(stats.keys())
    if len(models) >= 2:
        t1 = stats[models[0]]["avg_time_sec"]
        t2 = stats[models[1]]["avg_time_sec"]
        if t1 > 0 and t2 > 0:
            ratio = max(t1, t2) / min(t1, t2)
            faster = models[0] if t1 < t2 else models[1]
            print(f"⚡ 速度差异: {faster} 快 {ratio:.1f}x")


# ─────────────────────────────────────────────
# 步骤 4: 提取样例输出
# ─────────────────────────────────────────────
def extract_sample_outputs(results: dict, output_dir: str, n_samples: int = 3):
    """提取有表格的页面作为质量样例"""
    out_path = Path(output_dir)
    samples = []

    for model, model_results in results.items():
        for r in model_results:
            if not r.get("success") or not r.get("parsed"):
                continue
            p = r["parsed"]
            if p.get("has_table") and p.get("tables"):
                samples.append({
                    "model": model,
                    "image": r["image"],
                    "elapsed_sec": r["elapsed_sec"],
                    "topic": p.get("page_topic"),
                    "tables": p.get("tables"),
                    "key_values": p.get("key_values"),
                    "summary": p.get("summary"),
                })

    samples = samples[:n_samples]
    sample_file = out_path / "quality_samples.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n📋 已提取 {len(samples)} 个表格样例 → {sample_file}")
    for s in samples:
        print(f"  [{s['model']}] {s['topic']} - {len(s['tables'])} 个表格")

    return samples


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PDF 视觉方案测试")
    parser.add_argument(
        "--mode",
        choices=["convert", "test", "full"],
        default="test",
        help="convert=仅转PDF, test=仅测视觉, full=全流程"
    )
    parser.add_argument(
        "--pdf",
        default="/Users/jeff/l0-knowledge-engine/books/on_food_and_cooking.pdf"
    )
    parser.add_argument(
        "--pages-dir",
        default=str(PAGES_DIR)
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR)
    )
    parser.add_argument(
        "--sample-pages",
        default="0,10,30,50,100,200",
        help="逗号分隔的页码，如 0,10,50,100"
    )
    parser.add_argument(
        "--models",
        default="2b",
        help="逗号分隔的模型 key，如 2b,9b"
    )
    args = parser.parse_args()

    sample_pages = [int(x) for x in args.sample_pages.split(",")]
    model_keys = args.models.split(",")
    models_to_test = [MODELS[k] for k in model_keys if k in MODELS]

    print("🚀 PDF 视觉方案测试启动")
    print(f"   模式: {args.mode}")
    print(f"   模型: {models_to_test}")
    print(f"   测试页: {sample_pages}")
    print()

    if args.mode in ("convert", "full"):
        print("📄 步骤 1: 转换 PDF → 图片")
        ok = convert_pdf_to_images(args.pdf, args.pages_dir, dpi=150)
        if not ok and args.mode == "full":
            print("❌ PDF 转换失败，终止")
            return

    if args.mode in ("test", "full"):
        print("\n🔍 步骤 2: 视觉识别测试")
        summary, results = run_comparison_test(
            pages_dir=args.pages_dir,
            sample_pages=sample_pages,
            models=models_to_test,
            output_dir=args.output_dir,
        )

        print("\n🎯 步骤 3: 提取质量样例")
        extract_sample_outputs(results, args.output_dir)

        print(f"\n✅ 完成！结果保存在: {args.output_dir}")
        print("   - vision_results_detail.json  # 完整识别结果")
        print("   - vision_summary.json          # 模型对比汇总")
        print("   - quality_samples.json         # 表格质量样例")


if __name__ == "__main__":
    main()
