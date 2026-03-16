"""Evaluation and metrics utilities.

Provides functions for computing evaluation metrics and aggregating results.
"""

from typing import Any, Dict, List, Optional

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

from mellea_contribs.kg.qa_models import QAResult, QAStats
from mellea_contribs.kg.updater_models import UpdateResult, UpdateStats


def exact_match(predicted: str, expected: str) -> bool:
    """Check if predicted answer exactly matches expected answer (case-insensitive).

    Args:
        predicted: Predicted answer string.
        expected: Expected answer string.

    Returns:
        True if answers match exactly (case-insensitive), False otherwise.
    """
    return predicted.lower().strip() == expected.lower().strip()


def fuzzy_match(predicted: str, expected: str, threshold: float = 0.8) -> bool:
    """Check if predicted answer fuzzy-matches expected answer.

    Uses rapidfuzz token_set_ratio if available, otherwise falls back to exact match.

    Args:
        predicted: Predicted answer string.
        expected: Expected answer string.
        threshold: Similarity threshold (0-1, default: 0.8).

    Returns:
        True if similarity score >= threshold, False otherwise.
    """
    if fuzz is None:
        return exact_match(predicted, expected)

    score = fuzz.token_set_ratio(predicted.lower(), expected.lower()) / 100.0
    return score >= threshold


def mean_reciprocal_rank(results: List[Dict[str, Any]]) -> float:
    """Compute Mean Reciprocal Rank (MRR) for ranking results.

    Args:
        results: List of result dictionaries with 'answer', 'expected', and optional 'confidence'.

    Returns:
        MRR score (0-1).
    """
    if not results:
        return 0.0

    reciprocal_ranks = []

    for result in results:
        # Check for exact match first
        if exact_match(result.get("answer", ""), result.get("expected", "")):
            reciprocal_ranks.append(1.0)
        else:
            # For non-exact matches, use confidence as proxy
            confidence = result.get("confidence", 0.0)
            if confidence >= 0.9:
                reciprocal_ranks.append(1.0 / (1.0 + (1.0 - confidence)))
            else:
                reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def precision(predicted: List[str], expected: List[str]) -> float:
    """Compute precision metric (TP / (TP + FP)).

    Args:
        predicted: List of predicted items.
        expected: List of expected items.

    Returns:
        Precision score (0-1).
    """
    if not predicted:
        return 0.0

    tp = len(set(predicted) & set(expected))
    return tp / len(predicted)


def recall(predicted: List[str], expected: List[str]) -> float:
    """Compute recall metric (TP / (TP + FN)).

    Args:
        predicted: List of predicted items.
        expected: List of expected items.

    Returns:
        Recall score (0-1).
    """
    if not expected:
        return 0.0

    tp = len(set(predicted) & set(expected))
    return tp / len(expected)


def f1_score(prec: float, rec: float) -> float:
    """Compute F1 score (harmonic mean of precision and recall).

    Args:
        prec: Precision score.
        rec: Recall score.

    Returns:
        F1 score (0-1).
    """
    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def aggregate_qa_results(qa_results: List[QAResult]) -> QAStats:
    """Aggregate QA results into statistics.

    Args:
        qa_results: List of QAResult objects.

    Returns:
        QAStats object with aggregated statistics.
    """
    stats = QAStats()

    if not qa_results:
        return stats

    stats.total_questions = len(qa_results)

    # Count successful and failed
    successful = 0
    failed = 0
    times = []
    confidences = []

    for result in qa_results:
        if result.error:
            failed += 1
        else:
            successful += 1

        if result.processing_time_ms:
            times.append(result.processing_time_ms)

        if result.confidence:
            confidences.append(result.confidence)

    stats.successful_answers = successful
    stats.failed_answers = failed

    # Compute timing stats
    if times:
        stats.average_processing_time_ms = sum(times) / len(times)
        stats.min_processing_time_ms = min(times)
        stats.max_processing_time_ms = max(times)
        stats.total_time_ms = sum(times)

    # Compute confidence stats
    if confidences:
        stats.average_confidence = sum(confidences) / len(confidences)

    # Collect models used
    models = set(r.model_used for r in qa_results if r.model_used)
    stats.models_used = list(models)

    return stats


async def evaluate_predictions(
    session: Any,
    predictions: List[Dict[str, Any]],
    query_key: str = "query",
    answer_key: str = "answer",
    gold_key: str = "answer_aliases",
) -> List[Dict[str, Any]]:
    """Evaluate a list of QA predictions with LLM-based judgement.

    For each prediction the function checks whether the predicted answer
    matches the gold answer using a combination of fast heuristics (exact
    match, fuzzy match) and, for borderline cases, LLM judgement via a
    ``@generative`` function.

    Args:
        session: Mellea session used for LLM-based evaluation calls.
        predictions: List of dicts, each containing at least ``query_key``
            and ``answer_key`` fields plus an optional ``gold_key`` list of
            acceptable answers.
        query_key: Dict key for the question text (default: ``"query"``).
        answer_key: Dict key for the predicted answer (default: ``"answer"``).
        gold_key: Dict key for the list of gold answers
            (default: ``"answer_aliases"``).

    Returns:
        Same list with an added ``"correct"`` (bool) and
        ``"eval_method"`` (str) field on every item.
    """
    try:
        from mellea import generative
        from pydantic import BaseModel

        class _EvalResult(BaseModel):
            correct: bool
            reason: str

        @generative
        async def _llm_judge(
            query: str,
            predicted: str,
            gold_answers: str,
        ) -> _EvalResult:
            """Judge whether a predicted answer is correct.

            Question: {query}
            Predicted answer: {predicted}
            Acceptable answers: {gold_answers}

            Respond with a JSON object:
            {{"correct": true/false, "reason": "brief explanation"}}
            """
            pass

        _generative_available = True
    except Exception:
        _generative_available = False

    results = []
    for item in predictions:
        pred = str(item.get(answer_key, "")).strip()
        golds = item.get(gold_key, [])
        if isinstance(golds, str):
            golds = [golds]

        correct = False
        method = "none"

        # 1. Exact match
        for gold in golds:
            if exact_match(pred, str(gold)):
                correct = True
                method = "exact"
                break

        # 2. Fuzzy match
        if not correct:
            for gold in golds:
                if fuzzy_match(pred, str(gold)):
                    correct = True
                    method = "fuzzy"
                    break

        # 3. LLM judgement for uncertain cases
        if not correct and _generative_available and golds:
            try:
                gold_str = " | ".join(str(g) for g in golds)
                judge = await _llm_judge(
                    session,
                    query=str(item.get(query_key, "")),
                    predicted=pred,
                    gold_answers=gold_str,
                )
                correct = judge.correct
                method = "llm"
            except Exception:
                pass

        result = dict(item)
        result["correct"] = correct
        result["eval_method"] = method
        results.append(result)

    return results


def aggregate_update_results(update_results: List[UpdateResult]) -> UpdateStats:
    """Aggregate update results into statistics.

    Args:
        update_results: List of UpdateResult objects.

    Returns:
        UpdateStats object with aggregated statistics.
    """
    stats = UpdateStats()

    if not update_results:
        return stats

    stats.total_documents = len(update_results)

    # Count successful and failed
    successful = 0
    failed = 0
    times = []

    for result in update_results:
        if result.success:
            successful += 1
            stats.entities_extracted += result.entities_found
            stats.relations_extracted += result.relations_found
            stats.entities_new += result.entities_added
            stats.relations_new += result.relations_added
        else:
            failed += 1

        if result.processing_time_ms:
            times.append(result.processing_time_ms)

    stats.successful_documents = successful
    stats.failed_documents = failed

    # Compute timing stats
    if times:
        stats.total_processing_time_ms = sum(times)
        stats.average_processing_time_per_doc_ms = sum(times) / len(update_results)

    return stats
