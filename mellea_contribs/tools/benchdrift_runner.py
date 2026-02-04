"""BenchDrift-Mellea integration for robustness testing of Mellea m-programs.

Generates semantic variations of a problem using BenchDrift (demo-ui branch),
tests each through the m-program, and checks correctness. Per-variation streaming:
each variation is generated, tested, and reported before moving to the next.

All models run via Ollama.
"""

import logging
import os
import re
import time
from typing import List, Dict, Any, Callable, Optional

from mellea import MelleaSession

from benchdrift.pipeline.feature_relevance import (
    get_problem_features,
    enrich_features_with_llm,
    rank_transformations_two_level,
    parse_axes,
    _get_valid_axes,
    _rank_axes_by_features,
    TRANSFORMATION_TO_AXIS,
)
from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine
from benchdrift.pipeline.comprehensive_variation_engine_v2 import (
    clean_model_response,
    is_valid_question,
)
from benchdrift.models.model_client import ModelClientFactory

logger = logging.getLogger(__name__)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# --- Helpers ---

def _generate_one_variation(gen_client, problem: str, trans_name: str,
                            config: dict, max_retries: int = 2) -> str:
    system_prompt = (
        f"You are an expert at creating intent-preserving question variations.\n\n"
        f"TASK: Create a {trans_name} variation of the given problem.\n\n"
        f"TRANSFORMATION GOAL: {config['prompt']}\n\n"
        f"UNIVERSAL RULES:\n"
        f"1. PRESERVE the exact answer\n"
        f"2. MAINTAIN all mathematical/logical relationships\n"
        f"3. Numbers: format can change (5 -> five), value CANNOT (5 -> 6)\n"
        f"4. Units: convert correctly or not at all\n"
        f"5. Use PLAIN TEXT only\n"
        f"6. Return ONLY the question inside <question> tags\n"
        f"7. Do NOT explain your reasoning\n\n"
        f"<question>Your transformed question here</question>"
    )
    user_prompt = f"Original: {problem}\n\nReturn only the <question>...</question>."

    for attempt in range(max_retries + 1):
        try:
            raw = gen_client.get_single_response(
                system_prompt=system_prompt, user_prompt=user_prompt,
                max_new_tokens=1024, temperature=0.5)
        except Exception:
            if attempt == max_retries:
                return ""
            continue
        cleaned = _clean_response(raw)
        if cleaned and cleaned.strip().upper() == "SKIP":
            return ""
        if cleaned and is_valid_question(cleaned):
            return cleaned
        user_prompt = (f"Original: {problem}\n\nReturn ONLY the question text "
                       f"inside <question> tags. No explanation, no analysis.")
    return ""


def _clean_response(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r'<question>(.*?)</question>', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return clean_model_response(text)


def _extract_final_answer(text: str) -> str:
    """Extract final numeric/dollar answer from verbose model text."""
    if not text:
        return ""
    for pattern in [
        r'(?:total\s+cost|total|answer|result)\s*(?:is|=|:)\s*\$?([\d,]+\.?\d*)',
        r'=\s*\$?([\d,]+\.?\d*)\s*$',
        r'\*\*\$?([\d,]+\.?\d*)\*\*',
    ]:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].replace(',', '')
    dollars = re.findall(r'\$([\d,]+\.?\d*)', text)
    if dollars:
        return dollars[-1].replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', re.sub(r'(\d),(\d)', r'\1\2', text))
    if nums:
        return nums[-1]
    return text.strip()


def _answers_match(predicted: str, truth: str) -> bool:
    def _normalize(s):
        if not s:
            return ""
        extracted = _extract_final_answer(s)
        if extracted:
            s = extracted
        s = s.lower().strip().lstrip('$')
        for suffix in (".", "!", "?"):
            if s.endswith(suffix):
                s = s[:-1].strip()
        frac = re.search(r'-?\d+\s*/\s*\d+', s)
        if frac:
            return re.sub(r'\s', '', frac.group())
        nums = re.findall(r'-?\d+\.?\d*', re.sub(r'(\d),(\d)', r'\1\2', s))
        return nums[-1] if nums else s

    a, b = _normalize(predicted), _normalize(truth)
    if a == b:
        return True
    try:
        def _f(x):
            return float(x.split('/')[0]) / float(x.split('/')[1]) if '/' in x else float(x)
        return abs(_f(a) - _f(b)) < 1e-6
    except (ValueError, ZeroDivisionError, IndexError):
        pass
    return a in b or b in a


def _default_answer_extractor(response: Any) -> str:
    if hasattr(response, 'value'):
        return str(response.value)
    return str(response)


def _get_answer(problem, m_program_callable, answer_extractor, target_client) -> str:
    try:
        if m_program_callable is not None:
            return answer_extractor(m_program_callable(problem))
        raw = target_client.get_single_response(
            system_prompt="Solve the problem. Return ONLY the final answer.",
            user_prompt=problem, max_new_tokens=256, temperature=0.1)
        return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
    except Exception as e:
        return f"ERROR: {e}"


# --- Core API ---

def run_benchdrift_pipeline(
    baseline_problem: str,
    ground_truth_answer: str,
    m_program_callable: Optional[Callable[[str], Any]] = None,
    mellea_session: Optional[MelleaSession] = None,
    answer_extractor: Optional[Callable[[Any], str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, str, Dict], None]] = None,
) -> List[Dict[str, Any]]:
    """Generate semantic variations and test m-program robustness.

    Args:
        baseline_problem: Problem text to generate variations for
        ground_truth_answer: Expected correct answer
        m_program_callable: M-program function (takes str, returns Any)
        mellea_session: Mellea session (required with m_program_callable)
        answer_extractor: Extract answer string from m-program response
        config_overrides: Pipeline config (gen_model, target_model, top_k, use_axes, etc.)
        progress_callback: Called after each variation: (current, total, status, entry)
            status is "baseline", "skip", "PASS", or "FAIL"

    Returns:
        List of probe dicts with: is_baseline, is_variant, variation_type,
        modified_problem, variant_answer, correct, ground_truth_answer
    """
    if (m_program_callable is None) != (mellea_session is None):
        raise ValueError("Both m_program_callable and mellea_session must be provided together.")
    if config_overrides is None:
        raise ValueError("config_overrides is required.")

    gen_model = config_overrides.get('gen_model', 'qwen3:8b')
    target_model = config_overrides.get('target_model', 'granite3.3:8b')
    top_k = config_overrides.get('top_k', 10)
    use_axes = config_overrides.get('use_axes',
                                     'linguistic,referential,pragmatic,structural,constraint_targeted')
    ollama_url = config_overrides.get('ollama_url', OLLAMA_BASE_URL)
    timeout = config_overrides.get('timeout', 120)
    no_enrich = config_overrides.get('no_enrich', False)

    if answer_extractor is None:
        answer_extractor = _default_answer_extractor

    gen_client = ModelClientFactory.create_client('ollama', gen_model)
    target_client = None if m_program_callable else ModelClientFactory.create_client('ollama', target_model)

    # Feature analysis
    features = get_problem_features(baseline_problem)
    if not no_enrich:
        try:
            features.update(enrich_features_with_llm(
                baseline_problem, ollama_url, gen_model, timeout=timeout))
        except Exception as e:
            logger.debug(f"Feature enrichment skipped: {e}")

    # Relevance ranking
    enabled_axes = parse_axes(use_axes)
    all_types = {k: v for k, v in UnifiedVariationEngine.get_all_transformation_types().items()
                 if TRANSFORMATION_TO_AXIS.get(k) in enabled_axes}
    valid_axes = _get_valid_axes(features, enabled_axes=enabled_axes)
    ranked_axes = _rank_axes_by_features(features, valid_axes)
    ranked = rank_transformations_two_level(
        baseline_problem, features, all_types, top_k=top_k,
        pre_ranked_axes=ranked_axes, enabled_axes=enabled_axes)

    # Baseline
    baseline_answer = _get_answer(baseline_problem, m_program_callable, answer_extractor, target_client)
    baseline_correct = _answers_match(baseline_answer, ground_truth_answer)

    results = [{
        'is_baseline': True, 'is_variant': False,
        'variation_type': 'baseline',
        'modified_problem': baseline_problem,
        'ground_truth_answer': ground_truth_answer,
        'variant_answer': baseline_answer,
        'correct': baseline_correct,
    }]

    if progress_callback:
        progress_callback(0, len(ranked), "baseline", results[0])

    # Per-variation: generate → test → report
    for i, (trans_name, score, axis) in enumerate(ranked, 1):
        if trans_name not in all_types:
            continue
        t0 = time.time()
        variation = _generate_one_variation(gen_client, baseline_problem, trans_name, all_types[trans_name])
        gen_time = time.time() - t0

        if not variation:
            if progress_callback:
                progress_callback(i, len(ranked), "skip", {"variation_type": trans_name})
            continue

        answer = _get_answer(variation, m_program_callable, answer_extractor, target_client)
        correct = _answers_match(answer, ground_truth_answer)
        status = "PASS" if correct else "FAIL"

        entry = {
            'is_baseline': False, 'is_variant': True,
            'variation_type': trans_name,
            'transformation_axis': axis,
            'relevance_score': score,
            'modified_problem': variation,
            'ground_truth_answer': ground_truth_answer,
            'variant_answer': answer,
            'correct': correct,
            'generation_time_s': round(gen_time, 1),
        }
        results.append(entry)

        if progress_callback:
            progress_callback(i, len(ranked), status, entry)

    return results


# --- Result Analysis ---

def analyze_robustness(probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute robustness metrics: pass rate, per-type breakdown."""
    variants = [p for p in probes if p.get('is_variant')]
    if not variants:
        return {"error": "No variants", "pass_rate": 0.0, "total": 0}

    passed = sum(1 for p in variants if p.get('correct'))
    by_type = {}
    for p in variants:
        t = p.get('variation_type', 'unknown')
        if t not in by_type:
            by_type[t] = {'total': 0, 'passed': 0}
        by_type[t]['total'] += 1
        if p.get('correct'):
            by_type[t]['passed'] += 1

    baseline = next((p for p in probes if p.get('is_baseline')), None)
    return {
        "pass_rate": passed / len(variants),
        "total": len(variants),
        "passed": passed,
        "failed": len(variants) - passed,
        "baseline_correct": baseline.get('correct', False) if baseline else False,
        "by_variation_type": {t: s['passed'] / s['total'] for t, s in by_type.items()},
    }


# Keep old name as alias for backward compat
analyze_robustness_from_probes = analyze_robustness


def extract_passing_variants(probes: List[Dict[str, Any]]) -> List[str]:
    """Return modified_problem text for all variants that answered correctly."""
    return [p['modified_problem'] for p in probes
            if p.get('is_variant') and p.get('correct') and p.get('modified_problem')]


def extract_repair_candidates(probes: List[Dict[str, Any]],
                             baseline_problem: str) -> List[str]:
    """Find variants that pass when baseline fails (potential prompt repairs)."""
    baseline = next((p for p in probes if p.get('is_baseline')), None)
    if not baseline or baseline.get('correct'):
        return []
    return extract_passing_variants(probes)
