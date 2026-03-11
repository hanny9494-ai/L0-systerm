#!/usr/bin/env python3
"""
L0 Knowledge Engine - Extraction Pipeline
Using Claude Opus 4.6 for maximum quality

Usage:
    python l0_extract.py pilot      # 20题试跑
    python l0_extract.py chapter 14 # 单章抽取
    python l0_extract.py book ofc   # 全书抽取
"""

import anthropic
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    # API Configuration
    "api_base_url": "http://1.95.142.151:3000",
    "api_key": "sk-7g6RsJ5li3UUa3JXfHTzpvdsPimJPOre3S5eyN7WWdrrt33E",
    "model": "claude-opus-4-20250514",  # Claude Opus 4.6
    "max_tokens": 4096,
    "temperature": 0.3,
    
    # Token tracking
    "pricing": {
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},  # per 1M tokens
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    },
    
    # Paths
    "data_dir": Path("./data"),
    "output_dir": Path("./output"),
    "books_dir": Path("./books"),
    "stats_file": Path("./output/extraction_stats.json"),
}

# ============================================================
# Data Classes
# ============================================================

@dataclass
class TokenStats:
    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    total_principles: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def cost(self, model: str) -> float:
        pricing = CONFIG["pricing"].get(model, {"input": 15.0, "output": 75.0})
        return (self.input_tokens / 1_000_000) * pricing["input"] + \
               (self.output_tokens / 1_000_000) * pricing["output"]
    
    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_calls += 1
    
    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_calls": self.total_calls,
            "total_principles": self.total_principles,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost(CONFIG["model"]),
        }


@dataclass
class L0Principle:
    principle_id: str
    principle_name: str
    mechanism: str  # physics | chemistry | biology | sensory
    scientific_statement: str
    control_variables: list[str]
    expected_effects: list[str]
    boundary_conditions: list[str]
    counter_examples: list[str]
    evidence_level: str
    citation: dict
    source_question_id: str
    extraction_confidence: float
    extracted_at: str = ""
    
    def __post_init__(self):
        if not self.extracted_at:
            self.extracted_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def validate(self) -> tuple[bool, list[str]]:
        errors = []
        
        # Required fields
        if not self.principle_id:
            errors.append("missing_principle_id")
        if not self.mechanism or self.mechanism not in ["physics", "chemistry", "biology", "sensory"]:
            errors.append("invalid_mechanism")
        if not self.scientific_statement:
            errors.append("missing_scientific_statement")
        if not self.boundary_conditions:
            errors.append("missing_boundary_conditions")
        if not self.citation or not self.citation.get("source"):
            errors.append("missing_citation")
        
        # Quality checks
        vague_words = ["可能", "大概", "一般", "通常", "也许", "某些", "probably", "maybe", "usually"]
        if any(w in self.scientific_statement for w in vague_words):
            errors.append("vague_statement")
        
        return len(errors) == 0, errors


# ============================================================
# Claude API Client
# ============================================================

class ClaudeExtractor:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=CONFIG["api_key"],
            base_url=CONFIG["api_base_url"],
        )
        self.stats = TokenStats()
        self.load_stats()
    
    def load_stats(self):
        if CONFIG["stats_file"].exists():
            with open(CONFIG["stats_file"], "r") as f:
                data = json.load(f)
                self.stats.input_tokens = data.get("input_tokens", 0)
                self.stats.output_tokens = data.get("output_tokens", 0)
                self.stats.total_calls = data.get("total_calls", 0)
                self.stats.total_principles = data.get("total_principles", 0)
    
    def save_stats(self):
        CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
        with open(CONFIG["stats_file"], "w") as f:
            json.dump(self.stats.to_dict(), f, indent=2)
    
    def encode_image(self, image_path: Path) -> tuple[str, str]:
        """Encode image to base64 for Claude API"""
        suffix = image_path.suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        return data, media_type
    
    def call_claude(self, messages: list, system: str = "") -> tuple[str, int, int]:
        """Call Claude API and track tokens"""
        response = self.client.messages.create(
            model=CONFIG["model"],
            max_tokens=CONFIG["max_tokens"],
            temperature=CONFIG["temperature"],
            system=system,
            messages=messages,
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self.stats.add(input_tokens, output_tokens)
        self.save_stats()
        
        content = response.content[0].text
        return content, input_tokens, output_tokens
    
    def scan_page(self, image_path: Path, questions: list[dict]) -> list[dict]:
        """Phase 2: Scan a book page and find relevant passages for questions"""
        
        image_data, media_type = self.encode_image(image_path)
        
        questions_text = "\n".join([
            f"- {q['question_id']}: {q['question_text']}"
            for q in questions
        ])
        
        system = """你是食品科学专家。你的任务是阅读书籍页面，识别哪些段落可以回答给定的问题。

规则：
1. 只输出有明确关联的匹配（confidence >= 0.7）
2. relevant_text 必须是页面上的原文，不要改写
3. 如果一个段落能回答多个问题，分别列出
4. 输出 JSON 格式"""

        prompt = f"""请阅读这张书籍页面，找出可以回答以下问题的段落。

## 问题列表
{questions_text}

## 输出格式
请输出 JSON：
```json
{{
  "page_content_summary": "页面主要内容的简短摘要",
  "matches": [
    {{
      "question_id": "L0-Q-XXX-001",
      "relevant_text": "页面上的原文段落",
      "confidence": 0.85
    }}
  ]
}}
```

如果页面没有任何匹配，输出空 matches 数组。"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]
        
        response, input_tokens, output_tokens = self.call_claude(messages, system)
        
        # Parse JSON from response
        try:
            # Extract JSON from markdown code block if present
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            matches = result.get("matches", [])
            
            print(f"  📄 扫描完成: {len(matches)} 个匹配, {input_tokens}+{output_tokens} tokens")
            return matches
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON 解析失败: {e}")
            return []
    
    def extract_principle(
        self, 
        question: dict, 
        relevant_text: str, 
        source: str, 
        page: int
    ) -> Optional[L0Principle]:
        """Phase 3: Extract L0 principle from matched text"""
        
        system = """你是食品科学原理抽取专家。你的任务是从书籍原文中抽取可证伪的 L0 科学原理。

核心要求：
1. scientific_statement 必须是可证伪的陈述，包含具体数值或可验证的条件
2. boundary_conditions 必须包含具体数值（温度、时间、浓度等）
3. 不要使用模糊词汇（可能、大概、一般、通常）
4. 如果原文信息不足，输出 skip

输出必须是纯 JSON，不要任何其他文字。"""

        prompt = f"""## 目标问题
问题 ID: {question['question_id']}
问题类型: {question['question_type']}
领域: {question['domain']}
问题: {question['question_text']}

## 原文
来源: {source}, 第 {page} 页
原文: "{relevant_text}"

## 输出格式
```json
{{
  "principle_id": "L0-{question['domain'].upper()[:2]}-KEYWORD-001",
  "principle_name": "简短的中文原理名称",
  "mechanism": "physics 或 chemistry 或 biology 或 sensory",
  "scientific_statement": "一句可证伪的科学陈述",
  "control_variables": ["变量1", "变量2"],
  "expected_effects": ["效果1", "效果2"],
  "boundary_conditions": ["条件: 具体数值范围"],
  "counter_examples": ["反例或例外"],
  "evidence_level": "textbook",
  "citation": {{
    "source": "{source}",
    "page": {page},
    "quote": "原文短引（英文，<50词）"
  }},
  "extraction_confidence": 0.0到1.0
}}
```

如果信息不足：
```json
{{"skip": true, "reason": "原因说明"}}
```"""

        messages = [{"role": "user", "content": prompt}]
        
        response, input_tokens, output_tokens = self.call_claude(messages, system)
        
        try:
            # Parse JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            
            if result.get("skip"):
                print(f"  ⏭️ 跳过: {result.get('reason', 'unknown')}")
                return None
            
            principle = L0Principle(
                principle_id=result["principle_id"],
                principle_name=result["principle_name"],
                mechanism=result["mechanism"],
                scientific_statement=result["scientific_statement"],
                control_variables=result.get("control_variables", []),
                expected_effects=result.get("expected_effects", []),
                boundary_conditions=result.get("boundary_conditions", []),
                counter_examples=result.get("counter_examples", []),
                evidence_level=result.get("evidence_level", "textbook"),
                citation=result.get("citation", {}),
                source_question_id=question["question_id"],
                extraction_confidence=result.get("extraction_confidence", 0.8),
            )
            
            is_valid, errors = principle.validate()
            if not is_valid:
                print(f"  ⚠️ 验证警告: {errors}")
            
            self.stats.total_principles += 1
            self.save_stats()
            
            print(f"  ✅ 抽取成功: {principle.principle_id}, {input_tokens}+{output_tokens} tokens")
            return principle
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ❌ 解析失败: {e}")
            return None


# ============================================================
# Main Pipeline
# ============================================================

def load_questions(path: Path) -> list[dict]:
    """Load questions from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", data)


def run_pilot(extractor: ClaudeExtractor, questions_path: Path, pages_dir: Path):
    """Run 20-question pilot test"""
    print("\n" + "="*60)
    print("🚀 L0 抽取试跑 - 20 题")
    print("="*60)
    
    questions = load_questions(questions_path)[:20]
    print(f"📋 加载 {len(questions)} 个问题")
    
    # Get page images
    pages = sorted(pages_dir.glob("*.png")) + sorted(pages_dir.glob("*.jpg"))
    print(f"📄 找到 {len(pages)} 个页面")
    
    all_principles = []
    all_matches = []
    
    # Phase 2: Scan pages
    print("\n--- Phase 2: 扫描页面 ---")
    for page_path in pages[:10]:  # Limit for pilot
        print(f"\n处理: {page_path.name}")
        matches = extractor.scan_page(page_path, questions)
        for m in matches:
            m["page_path"] = str(page_path)
            m["page_number"] = int(page_path.stem.replace("p", "").replace("page_", ""))
        all_matches.extend(matches)
    
    print(f"\n📊 Phase 2 完成: {len(all_matches)} 个候选匹配")
    
    # Phase 3: Extract principles
    print("\n--- Phase 3: 抽取原理 ---")
    for match in all_matches:
        question = next((q for q in questions if q["question_id"] == match["question_id"]), None)
        if not question:
            continue
        
        print(f"\n问题: {match['question_id']}")
        principle = extractor.extract_principle(
            question=question,
            relevant_text=match["relevant_text"],
            source="On Food and Cooking",
            page=match.get("page_number", 0),
        )
        
        if principle:
            all_principles.append(principle.to_dict())
    
    # Save results
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "pilot_principles.jsonl", "w", encoding="utf-8") as f:
        for p in all_principles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    with open(output_dir / "pilot_matches.json", "w", encoding="utf-8") as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 试跑完成")
    print("="*60)
    print(f"总 API 调用: {extractor.stats.total_calls}")
    print(f"总 Tokens: {extractor.stats.total_tokens:,}")
    print(f"  - Input: {extractor.stats.input_tokens:,}")
    print(f"  - Output: {extractor.stats.output_tokens:,}")
    print(f"总成本: ${extractor.stats.cost(CONFIG['model']):.2f}")
    print(f"抽取原理数: {len(all_principles)}")
    print(f"\n输出文件:")
    print(f"  - {output_dir / 'pilot_principles.jsonl'}")
    print(f"  - {output_dir / 'pilot_matches.json'}")


def print_stats(extractor: ClaudeExtractor):
    """Print current token stats"""
    print("\n" + "="*60)
    print("📊 Token 统计")
    print("="*60)
    print(f"模型: {CONFIG['model']}")
    print(f"总调用: {extractor.stats.total_calls}")
    print(f"总 Tokens: {extractor.stats.total_tokens:,}")
    print(f"  - Input: {extractor.stats.input_tokens:,}")
    print(f"  - Output: {extractor.stats.output_tokens:,}")
    print(f"总成本: ${extractor.stats.cost(CONFIG['model']):.2f}")
    print(f"已抽取原理: {extractor.stats.total_principles}")


# ============================================================
# Entry Point
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("""
L0 Knowledge Engine - Extraction Pipeline

Usage:
    python l0_extract.py pilot              # 20题试跑
    python l0_extract.py stats              # 查看 token 统计
    python l0_extract.py reset              # 重置统计
    
Examples:
    export ANTHROPIC_API_KEY=sk-ant-xxx
    python l0_extract.py pilot
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    extractor = ClaudeExtractor()
    
    if command == "pilot":
        questions_path = CONFIG["data_dir"] / "pilot_20_questions.json"
        pages_dir = CONFIG["books_dir"] / "ofc" / "pilot_pages"
        
        if not questions_path.exists():
            print(f"❌ 找不到问题文件: {questions_path}")
            print("请先创建 pilot_20_questions.json")
            sys.exit(1)
        
        if not pages_dir.exists():
            print(f"❌ 找不到页面目录: {pages_dir}")
            print("请先准备书籍页面图片")
            sys.exit(1)
        
        run_pilot(extractor, questions_path, pages_dir)
    
    elif command == "stats":
        print_stats(extractor)
    
    elif command == "reset":
        if CONFIG["stats_file"].exists():
            CONFIG["stats_file"].unlink()
        print("✅ 统计已重置")
    
    else:
        print(f"❌ 未知命令: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
