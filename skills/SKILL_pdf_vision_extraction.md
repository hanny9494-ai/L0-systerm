# SKILL: PDF 视觉提取（Modernist Cuisine 类排版）

## 问题
Modernist Cuisine 表格是图片化排版，MinerU VLM 无法识别，只生成占位符。

## 解决方案：MinerU + qwen3-vl-plus 合并

| 内容 | 方案 |
|------|------|
| 文字段落 | MinerU（准确、免费） |
| 图片化表格 | qwen3-vl-plus |
| 照片/图表 | qwen3-vl-plus |

## 合并效果（ch13 Thickeners，54页）
- 词数：31,799（MinerU 24,778 / qwen 19,432）
- 表格：54个（MinerU 0个）
- 残留占位符：3个（纯装饰图）

## 本地部署
M4 Max 128GB 可跑 qwen3-vl-32b：
```bash
brew install llama.cpp
# 下载 Qwen3-VL-32B-Instruct-Q4_K_M.gguf + mmproj
# 注意：Ollama 暂不支持 qwen3-vl 视觉，需用 llama.cpp 或 vllm
```

## 相关脚本
- 合并：/Users/jeff/l0-knowledge-engine/merge_mineru_qwen.py
- 视觉对比：/Users/jeff/l0-knowledge-engine/qwen_vision_compare.py
- MinerU API：/Users/jeff/l0-knowledge-engine/mineru_api.py
