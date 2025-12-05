"""
Author: IBM Research – Mellea Agent Team
Maintainer: Mellea Agent - IBM Research

Purpose: Generic Top-K selection engine using Mellea agents.
- Takes N items and asks the model to pick the top K based on a comparison prompt.
- Returns items sorted by score (frequency of selection).

When should an agent use it?
- Use when you have multiple candidate items and need the model to rank or pick the best ones.
- Suitable for:
    * Tool selection
    * Candidate recommendations
    * Ranking options in a tournament or comparison setup
- Avoid using for:
    * Single-item deterministic selection
    * Non-comparable items (items must have attributes or context the model can reason over)

"""

import json
from typing import Any, Dict, List, Optional, Tuple
from mellea.backends.types import ModelOption
from mellea.stdlib.requirement import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

TOP_K_CACHE: Dict[str, Any] = {}


def cache_key(items: List[Any], context: Any, prompt: str, k: int) -> str:
    return f"{hash(str(items))}::{hash(str(context))}::{hash(prompt)}::{k}"


def cached(fn):
    def wrapper(*args, **kwargs):
        key = cache_key(
            kwargs.get("items") or args[0],
            kwargs.get("context") or (args[3] if len(args) > 3 else None),
            kwargs.get("comparison_prompt") or args[1],
            kwargs.get("k") or args[2]
        )
        if key in TOP_K_CACHE:
            return TOP_K_CACHE[key]
        result = fn(*args, **kwargs)
        TOP_K_CACHE[key] = result
        return result

    return wrapper


def extract_top_k(raw: str, items: List[Any]) -> List[Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [item for item in parsed if item in [i["name"] if isinstance(i, dict) else i for i in items]]
    except Exception:
        pass
    return []


@cached
def top_k(items: List[Any],
          comparison_prompt: str,
          m,
          k: int = 1,
          context: Optional[Any] = None) -> List[Tuple[Any, int]]:
    system_prompt = f"You are a terse Top-{k} selector. Output exactly {k} item names as a valid JSON array, no additional text."

    try:
        user_prompt = f"""
        TASK: Select the top {k} items based on the following criteria:

        {comparison_prompt}

        Items:
        {json.dumps(items, indent=2, default=str)}

        Respond ONLY with a JSON array of exactly {k} items.
        """
    except Exception as e:
        print(f"Error constructing prompt: {e}")
        user_prompt = comparison_prompt

    response = m.instruct(
        user_prompt,
        grounding_context=context if context else None,
        model_options={ModelOption.SYSTEM_PROMPT: system_prompt},
        requirements=[
            req(f"Response must be a valid JSON array of up to {k} items",
                validation_fn=simple_validate(lambda s: bool(extract_top_k(s, items))))
        ],
        strategy=RejectionSamplingStrategy(loop_budget=2),
    )

    selected = extract_top_k(getattr(response, "value", ""), items)

    scores = {i["name"] if isinstance(i, dict) else i: 1 if (i["name"] if isinstance(i, dict) else i) in selected else 0
              for i in items}

    results = [(i, scores[i["name"] if isinstance(i, dict) else i]) for i in items]
    results.sort(key=lambda x: x[1], reverse=True)
    return results
