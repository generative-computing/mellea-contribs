"""
Author: IBM Research – Mellea Agent Team
Maintainer: Mellea Agent - IBM Research

Purpose: Generic Double Round Robin (DRR) engine for Mellea agents.DRR performs pairwise comparisons between multiple items, using LLM judgment.

What DRR Does:
- Takes N items to compare.
- Runs every possible pair (A vs B, B vs A).
- The model decides winner ("A" or "B").
- Each winner gets a point.
- Returns all items with accumulated scores.

When should an agent use it?
- When multiple candidate judgments exist for an entity.
- When the agent wants to perform robust evaluation using pairwise reasoning.
- Especially useful for cases where scores or confidence values alone may be ambiguous.
"""

import json
from typing import Any, Dict, List, Tuple, Optional
from mellea.backends.types import ModelOption
from mellea.stdlib.requirement import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.helpers.fancy_logger import FancyLogger

logger = FancyLogger.get_logger()

DRR_CACHE: Dict[str, Any] = {}

def cache_key(items: List[Any], context: Any, prompt: str) -> str:
    return f"{hash(str(items))}::{hash(str(context))}::{hash(prompt)}"

def cached(fn):
    def wrapper(*args, **kwargs):
        key = cache_key(
            kwargs.get("items") or args[0],
            kwargs.get("context") or args[3] if len(args) > 3 else None,
            kwargs.get("comparison_prompt") or args[1]
        )
        if key in DRR_CACHE:
            return DRR_CACHE[key]
        result = fn(*args, **kwargs)
        DRR_CACHE[key] = result
        return result
    return wrapper


def extract_choice(raw: str) -> str:
    cleaned = raw.strip().strip('\'" .!?\n\r\t').upper()
    if cleaned.startswith("A"):
        return "A"
    if cleaned.startswith("B"):
        return "B"
    return None

def compare_pair(item_a: Any,
                 item_b: Any,
                 comparison_prompt: str,
                 m,
                 context: Optional[Any] = None) -> Optional[str]:

    try:
        prompt = f"""
        Option A:
        {json.dumps(item_a, indent=2, default=str) if not isinstance(item_a, str) else item_a}

        Option B:
        {json.dumps(item_b, indent=2, default=str) if not isinstance(item_b, str) else item_b}

        Task:
        {comparison_prompt}

        Respond ONLY with a single token: A or B
        """
    except Exception as e:
        logger.warning(f"Failed to construct prompt: {e}")
        return None

    system_prompt = "You are a terse pairwise selector. Output exactly one token: A or B."

    validator = lambda s: extract_choice(s) in ["A", "B"]

    response = m.instruct(
        prompt,
        grounding_context=context if context else None,
        model_options={ModelOption.SYSTEM_PROMPT: system_prompt},
        requirements=[req("Response must be 'A' or 'B'", validation_fn=simple_validate(validator))],
        strategy=RejectionSamplingStrategy(loop_budget=2),
    )

    raw = getattr(response, "value", "").strip()
    winner = extract_choice(raw)
    return winner  # can be None if model output is invalid

@cached
def double_round_robin(items: List[Any],
                       comparison_prompt: str,
                       m,
                       context: Optional[Any] = None) -> List[Tuple[Any, int]]:

    n = len(items)
    scores = {i: 0 for i in range(n)}  # index → score

    for i in range(n):
        for j in range(i + 1, n):
            winner1 = compare_pair(items[i], items[j], comparison_prompt, m, context)
            winner2 = compare_pair(items[j], items[i], comparison_prompt, m, context)

            if winner1 == "A":
                scores[i] += 1
            elif winner1 == "B":
                scores[j] += 1

            if winner2 == "A":
                scores[j] += 1
            elif winner2 == "B":
                scores[i] += 1

    results = [(items[i], scores[i]) for i in range(n)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results