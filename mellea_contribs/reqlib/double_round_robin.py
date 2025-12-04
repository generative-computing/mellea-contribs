"""
double_round_robin.py

Author:
Maintainer: Mellea Agent - IBM Research

Purpose:Standalone library for Double Round Robin (DRR) evaluation of root cause judgments for Mellea agents.

What is Double Round Robin (DRR)?
DRR is a pairwise comparison method to determine the most appropriate judgment label for a given entity by evaluating all possible pairs of candidate labels twice
(A vs B and B vs A). The label that wins the most pairwise comparisons is selected as the final judgment.

When should an agent use it?
- When multiple candidate judgments exist for an entity.
- When the agent wants to perform robust evaluation using pairwise reasoning.
- Especially useful for cases where scores or confidence values alone may be ambiguous.
"""

import json
from mellea.backends.types import ModelOption
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.requirement import req, simple_validate
from typing import Dict, Any, List

JUDGMENT_CACHE: Dict[str, Any] = {}

def cache_key(entity_id: str, stage: str, payload: Any) -> str:
    return f"{entity_id}::{stage}::{hash(str(payload))}"

def cached(stage_name):
    def decorator(fn):
        def wrapper(entity_id, *args, **kwargs):
            payload = (args, kwargs)
            key = cache_key(entity_id, stage_name, payload)
            if key in JUDGMENT_CACHE:
                return JUDGMENT_CACHE[key]
            result = fn(entity_id, *args, **kwargs)
            JUDGMENT_CACHE[key] = result
            return result
        return wrapper
    return decorator

LABELS = ["PRIMARY_FAILURE", "EXONERATE", "INSUFFICIENT_EVIDENCE", "DEFER", "SYMPTOMS_ONLY"]

def extract_label(raw_output: str, valid_labels: List[str]) -> str:
    lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        if line in valid_labels:
            return line
    return "INSUFFICIENT_EVIDENCE"  # fallback if nothing matches

@cached("pairwise_eval")
def evaluate_pair(entity_id: str, a: str, b: str, entity_context: Dict[str, Any], m) -> str:
    prompt = f"""
        Compare two judgments for the entity {entity_id} based on its observability context.
        Option A: {a}
        Option B: {b}
        Which judgment is more appropriate? Respond ONLY with either '{a}' or '{b}'.
        Context (JSON): {json.dumps(entity_context, indent=2)}
        """
    response = m.instruct(
        prompt,
        model_options={ModelOption.SYSTEM_PROMPT: prompt},
        grounding_context=entity_context,
        requirements=[
            req(
                f"Response must be either '{a}' or '{b}'",
                validation_fn=simple_validate(lambda s: s.strip() == a or s.strip() == b),
            )
        ],
        strategy=RejectionSamplingStrategy(loop_budget=2),
    )

    raw = getattr(response, "value", a).strip()
    return extract_label(raw, LABELS)

@cached("double_round_robin")
def run_double_round_robin(entity_id: str, entity_context: Dict[str, Any], m, labels: List[str] = None) -> str:
    if labels is None:
        labels = LABELS
    scores = {lbl: 0 for lbl in labels}

    for i in range(len(LABELS)):
        for j in range(i + 1, len(LABELS)):
            A = LABELS[i]
            B = LABELS[j]
            winner1 = evaluate_pair(entity_id, A, B, entity_context, m)
            winner2 = evaluate_pair(entity_id, B, A, entity_context, m)

            winner1 = extract_label(winner1, LABELS)
            winner2 = extract_label(winner2, LABELS)

            scores[winner1] += 1
            scores[winner2] += 1

    final = max(scores.items(), key=lambda x: x[1])[0]
    return final
