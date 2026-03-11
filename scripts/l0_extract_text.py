#!/usr/bin/env python3
"""
L0 Knowledge Engine - Extraction Pipeline (Text Version)
Using Claude Opus 4.6 for maximum quality

Supports EPUB text extraction (no images needed)

Usage:
    python l0_extract.py pilot      # 20题试跑
    python l0_extract.py stats      # 查看统计
"""

import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

try:
    import anthropic
except ImportError:
    print("请安装 anthropic: pip install anthropic")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    # API Configuration
    "api_base_url": "http://1.95.142.151:3000",
    "api_key": "sk-7g6RsJ5li3UUa3JXfHTzpvdsPimJPOre3S5eyN7WWdrrt33E",
    "model": "claude-opus-4.6",
    "max_tokens": 4096,
    "temperature": 0.3,
    
    # Token tracking (per 1M tokens)
    "pricing": {
        "claude-opus-4.6": {"input": 15.0, "output": 75.0},
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
    mechanism: str
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
        if not self.principle_id:
            errors.append("missing_principle_id")
        if self.mechanism not in ["physics", "chemistry", "biology", "sensory"]:
            errors.append("invalid_mechanism")
        if not self.scientific_statement:
            errors.append("missing_scientific_statement")
        if not self.boundary_conditions:
            errors.append("missing_boundary_conditions")
        
        vague_words = ["可能", "大概", "一般", "通常", "也许"]
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
    
    def call_claude(self, messages: list, system: str = "") -> tuple[str, int, int]:
        """Call Claude API and track tokens"""
        try:
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
            
        except Exception as e:
            print(f"  ❌ API 错误: {e}")
            raise
    
    def scan_text_chunk(self, text: str, questions: list[dict], chunk_info: str = "") -> list[dict]:
        """Phase 2: Scan text chunk and find relevant passages"""
        
        questions_text = "\n".join([
            f"- {q['question_id']}: {q['question_text']}"
            for q in questions
        ])
        
        system = """你是食品科学专家。你的任务是阅读文本内容，识别哪些段落可以回答给定的问题。

规则：
1. 只输出有明确关联的匹配（confidence >= 0.7）
2. relevant_text 必须是原文，不要改写
3. 一个段落能回答多个问题时，分别列出
4. 输出纯 JSON，不要其他文字"""

        prompt = f"""请阅读以下书籍内容，找出可以回答问题的段落。

## 来源信息
{chunk_info}

## 书籍内容
{text[:8000]}  

## 问题列表
{questions_text}

## 输出格式
```json
{{
  "matches": [
    {{
      "question_id": "L0-Q-XXX-001",
      "relevant_text": "原文相关段落",
      "confidence": 0.85
    }}
  ]
}}
```"""

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response, input_tokens, output_tokens = self.call_claude(messages, system)
            
            # Parse JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            matches = result.get("matches", [])
            
            print(f"  📄 扫描完成: {len(matches)} 个匹配, {input_tokens}+{output_tokens} tokens, ${self.stats.cost(CONFIG['model']):.3f}")
            return matches
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON 解析失败: {e}")
            return []
    
    def extract_principle(
        self, 
        question: dict, 
        relevant_text: str, 
        source: str = "On Food and Cooking",
        chapter: str = ""
    ) -> Optional[L0Principle]:
        """Phase 3: Extract L0 principle from matched text"""
        
        domain_code = question['domain'][:2].upper()
        
        system = """你是食品科学原理抽取专家。从书籍原文中抽取可证伪的 L0 科学原理。

核心要求：
1. scientific_statement 必须可证伪，包含具体数值或可验证条件
2. boundary_conditions 必须有具体数值（温度、时间、浓度等）
3. 不要模糊词（可能、大概、一般、通常）
4. 信息不足则输出 skip

输出纯 JSON。"""

        prompt = f"""## 目标问题
问题 ID: {question['question_id']}
问题类型: {question['question_type']}
领域: {question['domain']}
问题: {question['question_text']}

## 原文
来源: {source}, {chapter}
原文: "{relevant_text}"

## 输出格式
```json
{{
  "principle_id": "L0-{domain_code}-KEYWORD-001",
  "principle_name": "简短中文原理名称",
  "mechanism": "physics 或 chemistry 或 biology 或 sensory",
  "scientific_statement": "可证伪的科学陈述",
  "control_variables": ["变量1", "变量2"],
  "expected_effects": ["效果1"],
  "boundary_conditions": ["条件: 具体数值"],
  "counter_examples": ["反例"],
  "evidence_level": "textbook",
  "citation": {{
    "source": "{source}",
    "chapter": "{chapter}",
    "quote": "原文短引(<50词)"
  }},
  "extraction_confidence": 0.85
}}
```

信息不足时:
```json
{{"skip": true, "reason": "原因"}}
```"""

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response, input_tokens, output_tokens = self.call_claude(messages, system)
            
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
            
            print(f"  ✅ 抽取成功: {principle.principle_id}, ${self.stats.cost(CONFIG['model']):.3f}")
            return principle
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ❌ 解析失败: {e}")
            return None


# ============================================================
# Main Pipeline
# ============================================================

def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", data)


def load_text_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pilot(extractor: ClaudeExtractor):
    """Run 20-question pilot test with text chunks"""
    print("\n" + "="*60)
    print("🚀 L0 抽取试跑 - 20 题 (文本模式)")
    print("="*60)
    
    # Load questions
    questions_path = CONFIG["data_dir"] / "pilot_20_questions.json"
    if not questions_path.exists():
        print(f"❌ 找不到问题文件: {questions_path}")
        return
    
    questions = load_questions(questions_path)
    print(f"📋 加载 {len(questions)} 个问题")
    
    # Load text chunks
    chunks_path = CONFIG["books_dir"] / "ofc" / "pilot_chunks" / "pilot_chunks.json"
    if not chunks_path.exists():
        print(f"❌ 找不到文本块: {chunks_path}")
        print("请先运行: python process_epub.py")
        return
    
    chunks = load_text_chunks(chunks_path)
    print(f"📄 加载 {len(chunks)} 个文本块")
    
    all_principles = []
    all_matches = []
    
    # Phase 2: Scan chunks
    print("\n--- Phase 2: 扫描文本块 ---")
    for i, chunk in enumerate(chunks[:15]):  # Limit for pilot
        chunk_info = f"Chapter {chunk['chapter_num']}: {chunk['chapter_title'][:50]}"
        print(f"\n[{i+1}/{min(15, len(chunks))}] {chunk_info}")
        
        matches = extractor.scan_text_chunk(
            text=chunk["text"],
            questions=questions,
            chunk_info=chunk_info
        )
        
        for m in matches:
            m["chunk_id"] = chunk["chunk_id"]
            m["chapter_num"] = chunk["chapter_num"]
            m["chapter_title"] = chunk["chapter_title"]
        
        all_matches.extend(matches)
        
        # Rate limit
        time.sleep(0.5)
    
    print(f"\n📊 Phase 2 完成: {len(all_matches)} 个候选匹配")
    
    # Phase 3: Extract principles
    print("\n--- Phase 3: 抽取原理 ---")
    for i, match in enumerate(all_matches):
        question = next((q for q in questions if q["question_id"] == match["question_id"]), None)
        if not question:
            continue
        
        print(f"\n[{i+1}/{len(all_matches)}] {match['question_id']}")
        
        principle = extractor.extract_principle(
            question=question,
            relevant_text=match["relevant_text"],
            source="On Food and Cooking",
            chapter=f"Chapter {match.get('chapter_num', '?')}"
        )
        
        if principle:
            all_principles.append(principle.to_dict())
        
        time.sleep(0.5)
    
    # Save results
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "pilot_principles.jsonl", "w", encoding="utf-8") as f:
        for p in all_principles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    with open(output_dir / "pilot_matches.json", "w", encoding="utf-8") as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=2)
    
    # Summary
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
    print("\n" + "="*60)
    print("📊 Token 统计")
    print("="*60)
    print(f"模型: {CONFIG['model']}")
    print(f"API: {CONFIG['api_base_url']}")
    print(f"总调用: {extractor.stats.total_calls}")
    print(f"总 Tokens: {extractor.stats.total_tokens:,}")
    print(f"  - Input: {extractor.stats.input_tokens:,}")
    print(f"  - Output: {extractor.stats.output_tokens:,}")
    print(f"总成本: ${extractor.stats.cost(CONFIG['model']):.2f}")
    print(f"已抽取原理: {extractor.stats.total_principles}")


def main():
    if len(sys.argv) < 2:
        print("""
L0 Knowledge Engine - Extraction Pipeline (Text Version)

Usage:
    python l0_extract.py pilot    # 20题试跑
    python l0_extract.py stats    # 查看统计
    python l0_extract.py reset    # 重置统计
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    extractor = ClaudeExtractor()
    
    if command == "pilot":
        run_pilot(extractor)
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
