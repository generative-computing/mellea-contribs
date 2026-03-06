"""BenchDrift-Mellea integration for robustness testing of Mellea m-programs.

Generates semantic variations of a problem using BenchDrift (demo-ui branch),
validates them, tests each through the m-program, and evaluates correctness.

Per-variation streaming — for each ranked transformation:
  1. Generate variation (Ollama)
  2. Validate it preserves meaning (Ollama judge)
  3. Test it through m-program
  4. Evaluate answer correctness (string match or LLM judge)
  5. Report result

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
from benchdrift.pipeline.council_validator import (
    get_judge_validation_prompt,
    build_judge_user_prompt,
    parse_judge_response,
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


def _validate_variation(judge_client, original: str, variation: str,
                        ground_truth: str) -> bool:
    """Validate that a variation preserves the original answer.
    Uses BenchDrift's judge validation prompt from council_validator."""
    try:
        system_prompt = get_judge_validation_prompt()
        user_prompt = build_judge_user_prompt(original, variation, ground_truth)
        raw = judge_client.get_single_response(
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_new_tokens=32, temperature=0.0)
        verdict = parse_judge_response(raw)
        return verdict == "VALID"
    except Exception:
        return True  # on failure, don't filter out


def _llm_judge_answer(judge_client, problem: str, ground_truth: str,
                      predicted: str) -> bool:
    """Use LLM judge to evaluate if predicted answer matches ground truth."""
    try:
        system_prompt = (
            "You are an answer evaluation judge. Compare the predicted answer to the "
            "ground truth answer. They may be in different formats but represent the same value. "
            "Respond with ONLY 'CORRECT' or 'INCORRECT'."
        )
        user_prompt = (
            f"Problem: {problem}\n"
            f"Ground truth: {ground_truth}\n"
            f"Predicted: {predicted}\n\n"
            f"Is the predicted answer correct? Reply ONLY 'CORRECT' or 'INCORRECT'."
        )
        raw = judge_client.get_single_response(
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_new_tokens=32, temperature=0.0)
        return "CORRECT" in raw.upper() and "INCORRECT" not in raw.upper()
    except Exception:
        return _answers_match(predicted, ground_truth)  # fallback to string match


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

    For each ranked transformation:
      1. Generate a variation (gen_model via Ollama)
      2. Validate it preserves meaning (gen_model as judge)
      3. Test it through the m-program
      4. Evaluate correctness (string match, or LLM judge if use_llm_judge=True)

    Args:
        baseline_problem: Problem text to generate variations for
        ground_truth_answer: Expected correct answer
        m_program_callable: M-program function (takes str, returns Any)
        mellea_session: Mellea session (required with m_program_callable)
        answer_extractor: Extract answer string from m-program response
        config_overrides: Pipeline config:
            gen_model:       Ollama model for variation gen + validation (default: 'qwen3:8b')
            target_model:    Ollama model when no m-program (default: 'granite3.3:8b')
            top_k:           Number of ranked transformations (default: 10)
            use_axes:        Taxonomy axes (default: 5 core axes)
            no_enrich:       Skip LLM feature enrichment (default: False)
            skip_validation: Skip variation validation (default: False)
            use_llm_judge:   Use LLM for answer evaluation (default: False)
            ollama_url:      Ollama server URL
            timeout:         Ollama call timeout in seconds
        progress_callback: Called after each variation: (current, total, status, entry)
            status: "baseline", "skip", "invalid", "PASS", or "FAIL"

    Returns:
        List of probe dicts with: is_baseline, is_variant, variation_type,
        modified_problem, variant_answer, correct, valid, ground_truth_answer
    """
    if (m_program_callable is None) != (mellea_session is None):
        raise ValueError("Both m_program_callable and mellea_session must be provided together.")
    if config_overrides is None:
        raise ValueError("config_overrides is required.")

    gen_model = config_overrides.get('gen_model', 'qwen3:8b')
    judge_model = config_overrides.get('judge_model', gen_model)
    target_model = config_overrides.get('target_model', 'granite3.3:8b')
    top_k = config_overrides.get('top_k', 10)
    use_axes = config_overrides.get('use_axes',
                                     'linguistic,referential,pragmatic,structural,constraint_targeted')
    ollama_url = config_overrides.get('ollama_url', OLLAMA_BASE_URL)
    timeout = config_overrides.get('timeout', 120)
    no_enrich = config_overrides.get('no_enrich', False)
    skip_validation = config_overrides.get('skip_validation', False)
    use_llm_judge = config_overrides.get('use_llm_judge', False)

    if answer_extractor is None:
        answer_extractor = _default_answer_extractor

    gen_client = ModelClientFactory.create_client('ollama', gen_model)
    judge_client = gen_client if judge_model == gen_model else ModelClientFactory.create_client('ollama', judge_model)
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
    if use_llm_judge:
        baseline_correct = _llm_judge_answer(judge_client, baseline_problem, ground_truth_answer, baseline_answer)
    else:
        baseline_correct = _answers_match(baseline_answer, ground_truth_answer)

    results = [{
        'is_baseline': True, 'is_variant': False,
        'variation_type': 'baseline',
        'modified_problem': baseline_problem,
        'ground_truth_answer': ground_truth_answer,
        'variant_answer': baseline_answer,
        'correct': baseline_correct,
        'valid': True,
    }]

    if progress_callback:
        progress_callback(0, len(ranked), "baseline", results[0])

    # 3-stage pipeline: gen → validate → test, all overlapping
    # Stage 1 (gen) and Stage 2 (validate) run in background threads.
    # Stage 3 (test via m-program) runs on main thread as results arrive.
    import queue, threading

    valid_ranked = [(t, s, a) for t, s, a in ranked if t in all_types]
    total = len(valid_ranked)

    # Queue: validated variations ready for testing
    ready_q = queue.Queue()
    done_event = threading.Event()

    def _pipeline_worker():
        """Background: generate + validate each variation, push to ready_q."""
        for idx, (trans_name, score, axis) in enumerate(valid_ranked):
            variation = _generate_one_variation(
                gen_client, baseline_problem, trans_name, all_types[trans_name])
            if not variation:
                ready_q.put((idx, trans_name, score, axis, None, "skip"))
                continue
            if not skip_validation:
                if not _validate_variation(judge_client, baseline_problem, variation, ground_truth_answer):
                    ready_q.put((idx, trans_name, score, axis, None, "invalid"))
                    continue
            ready_q.put((idx, trans_name, score, axis, variation, "ready"))
        done_event.set()

    # Start background pipeline
    worker = threading.Thread(target=_pipeline_worker, daemon=True)
    worker.start()

    # Main thread: test each variation as soon as it arrives
    processed = 0
    while True:
        try:
            item = ready_q.get(timeout=0.3)
        except queue.Empty:
            if done_event.is_set() and ready_q.empty():
                break
            if progress_callback:
                progress_callback(processed, total, "waiting", {})
            continue

        idx, trans_name, score, axis, variation, gen_status = item
        processed += 1

        if gen_status == "skip":
            if progress_callback:
                progress_callback(processed, total, "skip", {"variation_type": trans_name})
            continue
        if gen_status == "invalid":
            if progress_callback:
                progress_callback(processed, total, "invalid", {"variation_type": trans_name})
            continue

        # Test immediately via m-program (different model, no contention)
        answer = _get_answer(variation, m_program_callable, answer_extractor, target_client)

        if use_llm_judge:
            correct = _llm_judge_answer(judge_client, variation, ground_truth_answer, answer)
        else:
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
            'valid': True,
        }
        results.append(entry)

        if progress_callback:
            progress_callback(processed, total, status, entry)

    worker.join(timeout=5)

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
