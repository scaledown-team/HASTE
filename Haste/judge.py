#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Judge functionality for HasteContext.

This module provides functions for evaluating the quality of code extraction,
including the ability to call external LLM APIs for code quality assessment.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import tiktoken

from .api import build_structural_context_from_source


def _strip_code_fences(text: str) -> str:
    """Remove common markdown code fences like ```json ... ```."""
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        # remove first line fence and trailing fence
        lines = s.splitlines()
        if len(lines) >= 2:
            inner = "\n".join(lines[1:-1])
            return inner.strip()
    return s


def _parse_json_dict_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a JSON object embedded in text.

    Tries direct loads, code-fence stripping, then balanced-brace scanning.
    Returns dict or None.
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    stripped = _strip_code_fences(text)
    if stripped != text:
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # Balanced brace scan
    s = text
    starts = [i for i, ch in enumerate(s) if ch == '{']
    for si in starts:
        depth = 0
        for j in range(si, len(s)):
            ch = s[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    cand = s[si:j + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break
    return None


def call_api_with_context(system_context: str, prompt: str, model: Optional[str] = None, api_provider: str = "openai") -> Dict[str, Any]:
    """
    Call an LLM API with the given system context and prompt.
    
    Args:
        system_context: The system context to send to the API
        prompt: The prompt to send to the API
        model: The model to use (defaults to GPT-4 for OpenAI)
        api_provider: The API provider to use ("openai" or "custom")
        
    Returns:
        dict: The API response as a dictionary
    """
    if api_provider == "openai":
        try:
            import openai
            client = openai.OpenAI()
            model = model or "gpt-4o"
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_context},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                "full_response": response.choices[0].message.content,
                "model": model,
                "provider": "openai"
            }
        except Exception as e:
            return {"error": str(e), "provider": "openai"}
    
    # Default implementation for custom providers would go here
    # For now, just return an error
    return {"error": "Unsupported API provider", "provider": api_provider}


def calculate_compression_metrics(original_code: str, compressed_code: str) -> Dict[str, float]:
    """
    Calculate compression metrics between original and compressed code.
    
    Args:
        original_code: The original source code
        compressed_code: The compressed code
        
    Returns:
        dict: Metrics including tokens, reduction percentage, and compression ratio
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    orig_tokens = len(encoding.encode(original_code))
    comp_tokens = len(encoding.encode(compressed_code))
    
    # Calculate reduction percentage and compression ratio
    reduction_pct = (1 - (comp_tokens / orig_tokens)) * 100 if orig_tokens > 0 else 0.0
    compression_ratio = (orig_tokens / max(1, comp_tokens)) if orig_tokens > 0 else 0.0
    
    return {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "reduction_percentage": reduction_pct,
        "compression_ratio": compression_ratio
    }


def simple_judge(compressed_code: str, query: str, original_code: str = None) -> Dict[str, Union[float, str]]:
    """
    Simple judge function that evaluates code quality without external API calls.
    
    Args:
        compressed_code: The code extracted by HasteContext
        query: The original query used
        original_code: The original source code (optional)
        
    Returns:
        dict: Score and justification
    """
    if not compressed_code or len(compressed_code.strip()) < 10:
        return {"score": 0, "justification": "Empty or very short code snippet."}
    
    # Extract structural context using HasteContext's built-in function
    structure = build_structural_context_from_source(compressed_code)
    
    # Basic scoring based on structure
    score = 0
    
    # Check if the compressed code contains expected elements
    has_functions = "Functions:" in structure
    has_classes = "Classes:" in structure
    has_imports = "Imports:" in structure
    has_docstring = "Docstring:" in structure
    
    # Award points for structural elements
    if has_functions:
        score += 30
    if has_classes:
        score += 20
    if has_imports:
        score += 10
    if has_docstring:
        score += 10
    
    # Check for query terms in the code
    query_terms = set(term.lower() for term in query.split() if len(term) > 3)
    content_lower = compressed_code.lower()
    matched_terms = sum(1 for term in query_terms if term in content_lower)
    
    # Award points for query term matches
    term_score = min(30, matched_terms * 10)
    score += term_score
    
    # Cap the score at 100
    score = min(100, score)
    
    # Generate justification
    justification_parts = []
    if has_functions:
        justification_parts.append("contains functions")
    if has_classes:
        justification_parts.append("contains classes")
    if matched_terms > 0:
        justification_parts.append(f"matches {matched_terms} query terms")
    
    # Add compression metrics if original code is provided
    if original_code:
        metrics = calculate_compression_metrics(original_code, compressed_code)
        justification_parts.append(f"achieves {metrics['reduction_percentage']:.1f}% reduction")
    
    justification = "Code " + ", ".join(justification_parts) + "."
    
    return {
        "score": score,
        "justification": justification
    }


def run_judge_with_retries(
    file_content: str,
    compressed_context: str,
    suggestion: str,
    model_output: str,
    reduction_pct: float,
    *,
    lang_name: str = "python",
    ast_summary: Optional[str] = None,
    max_attempts: Optional[int] = None,
    delay_s: Optional[float] = None,
    model: Optional[str] = None,
    api_provider: str = "openai",
    use_external_api: bool = True
) -> tuple[float, str, str]:
    """Call the judge with retries until it returns score and justification.

    Args:
        file_content: Original file content
        compressed_context: Compressed code context
        suggestion: Suggestion for code change
        model_output: Output from the model
        reduction_pct: Reduction percentage
        lang_name: Language name (default: "python")
        ast_summary: Optional AST summary
        max_attempts: Maximum number of retry attempts
        delay_s: Delay between retries in seconds
        model: Model to use for API calls
        api_provider: API provider to use
        use_external_api: Whether to use external API or simple judge

    Returns:
        tuple: (score, justification, raw_response_text)
    """
    if not use_external_api:
        # Use simple judge when external API is not requested
        result = simple_judge(compressed_context, suggestion, file_content)
        return (result["score"], result["justification"], json.dumps(result))
        
    attempts = 0
    max_attempts = max_attempts or int(os.getenv("JUDGE_MAX_ATTEMPTS", "5"))
    delay_s = delay_s or float(os.getenv("JUDGE_RETRY_DELAY_S", "2"))

    last_raw = ""
    score_val: Optional[float] = None
    justification: str = ""

    while attempts < max_attempts:
        attempts += 1
        try:
            judge_system = (
                "You are a strict static PATCH JUDGE for localized code edits produced "
                "from a COMPRESSED context. Score how well the edit satisfies the suggestion "
                "while staying a minimal, local change that can be dropped into the original file. "
                "Return ONLY JSON with keys 'score' (0-100) and 'justification' (<100 words)."
            )

            judge_prompt = f"""
Return ONLY a JSON object with keys "score" (0-100) and "justification" (<100 words). No preamble, no code fences.

Language: {lang_name}

Suggestion:
{suggestion.strip()}

Compressed context (this is the ONLY code the editor saw when producing the edit):
<compressed>
{compressed_context}
</compressed>

Optional structural summary of the original file (for orientation; do not assume access to code outside the compressed region):
{(ast_summary or "[none]")}

Original file (reference only; do NOT require edits outside the compressed region unless the suggestion explicitly demands it):
<original>
{file_content}
</original>

Proposed EDIT to apply (model output):
<edit>
{model_output}
</edit>

SCORING RUBRIC (sum to 100):
- Suggestion compliance (0-50): Implements exactly what the suggestion asks; correct target (function/class/lines); correct behavior; no hallucinated features.
- Locality & compatibility (0-30): Can be applied within the COMPRESSED context without touching unrelated code; uses only in-scope names; no new global deps unless explicitly required by the suggestion.
- Syntax & consistency (0-10): Valid {lang_name} syntax; identifiers/signatures match surrounding context implied by the compressed region/summary.
- Minimality (0-10): No unnecessary edits, boilerplate, comments, or refactors beyond the suggestion.

RULES:
- Ignore formatting/whitespace and unrelated style.
- Do NOT judge effects on parts of the original file not mentioned in the suggestion.
- If the edit contains explanations or code fences, ignore non-code and grade the code content.
- If the edit obviously cannot work at all, score near 0.
- If the suggestion was something which is already in the file, and model output is also similar then it's valid case i.e model didn't hallucinate anything.
- Unless and until model gives something hallucinatory content or something which breaks the code, then it's not a valid case.
- Good practice added by the model is valid case. 
- Don't judge the model's output for json parsing
"""

            judge = call_api_with_context(
                system_context=judge_system,
                prompt=judge_prompt,
                model=model,
                api_provider=api_provider
            )
            last_raw = judge.get("full_response", "")
        except Exception:
            last_raw = ""

        parsed = _parse_json_dict_from_text(last_raw) if isinstance(last_raw, str) else None
        if isinstance(parsed, dict):
            if "score" in parsed:
                try:
                    score_val = float(parsed["score"])
                except Exception:
                    score_val = None
            if "justification" in parsed and isinstance(parsed["justification"], str):
                justification = parsed["justification"].strip()

        if score_val is not None and justification:
            break

        if attempts < max_attempts:
            time.sleep(delay_s)

    if score_val is None:
        score_val = max(0.0, min(100.0, float(f"{reduction_pct:.2f}")))
    if not justification:
        justification = "Fallback to reduction percentage or parsing failure"

    return score_val, justification, last_raw
