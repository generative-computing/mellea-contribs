"""Tests for mellea_contribs.kg.utils.eval_utils module."""

import pytest

from mellea_contribs.kg.qa_models import QAResult, QAStats
from mellea_contribs.kg.updater_models import UpdateResult, UpdateStats
from mellea_contribs.kg.utils import (
    aggregate_qa_results,
    aggregate_update_results,
    exact_match,
    f1_score,
    fuzzy_match,
    mean_reciprocal_rank,
    precision,
    recall,
)


class TestExactMatch:
    """Tests for exact_match function."""

    def test_exact_match_identical(self):
        """Test exact match with identical strings."""
        assert exact_match("hello", "hello") is True

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        assert exact_match("Hello", "hello") is True
        assert exact_match("WORLD", "world") is True

    def test_exact_match_with_whitespace(self):
        """Test exact match handles whitespace."""
        assert exact_match("  hello  ", "hello") is True
        assert exact_match("hello", "  hello  ") is True

    def test_exact_match_different(self):
        """Test exact match with different strings."""
        assert exact_match("hello", "world") is False

    def test_exact_match_partial(self):
        """Test exact match with partial strings."""
        assert exact_match("hello", "hello world") is False

    def test_exact_match_empty_strings(self):
        """Test exact match with empty strings."""
        assert exact_match("", "") is True
        assert exact_match("", "test") is False


class TestFuzzyMatch:
    """Tests for fuzzy_match function."""

    def test_fuzzy_match_identical(self):
        """Test fuzzy match with identical strings."""
        assert fuzzy_match("hello", "hello") is True

    def test_fuzzy_match_similar(self):
        """Test fuzzy match with similar strings."""
        # Should match with high similarity
        assert fuzzy_match("hello", "helo", threshold=0.7) is True

    def test_fuzzy_match_different(self):
        """Test fuzzy match with different strings."""
        assert fuzzy_match("hello", "world", threshold=0.8) is False

    def test_fuzzy_match_threshold_sensitivity(self):
        """Test fuzzy match threshold sensitivity."""
        # Same pair, different thresholds
        pair1, pair2 = "testing", "test"
        assert fuzzy_match(pair1, pair2, threshold=0.5) is True
        assert fuzzy_match(pair1, pair2, threshold=0.99) is False

    def test_fuzzy_match_case_insensitive(self):
        """Test fuzzy match is case insensitive."""
        assert fuzzy_match("Hello", "hello") is True


class TestMeanReciprocalRank:
    """Tests for mean_reciprocal_rank function."""

    def test_mrr_empty_results(self):
        """Test MRR with empty results."""
        assert mean_reciprocal_rank([]) == 0.0

    def test_mrr_all_exact_matches(self):
        """Test MRR with all exact matches."""
        results = [
            {"answer": "correct", "expected": "correct"},
            {"answer": "right", "expected": "right"},
            {"answer": "yes", "expected": "yes"},
        ]
        mrr = mean_reciprocal_rank(results)
        assert mrr == 1.0

    def test_mrr_no_matches(self):
        """Test MRR with no matches."""
        results = [
            {"answer": "wrong", "expected": "correct"},
            {"answer": "bad", "expected": "good"},
        ]
        mrr = mean_reciprocal_rank(results)
        assert mrr == 0.0

    def test_mrr_partial_matches(self):
        """Test MRR with partial matches."""
        results = [
            {"answer": "correct", "expected": "correct"},  # Match
            {"answer": "wrong", "expected": "right"},  # No match
            {"answer": "okay", "expected": "okay"},  # Match
        ]
        mrr = mean_reciprocal_rank(results)
        # Average of [1.0, 0.0, 1.0] = 0.667
        assert 0.6 < mrr < 0.7

    def test_mrr_with_confidence(self):
        """Test MRR considering confidence."""
        results = [
            {"answer": "maybe", "expected": "correct", "confidence": 0.95},
            {"answer": "no", "expected": "yes", "confidence": 0.1},
        ]
        mrr = mean_reciprocal_rank(results)
        # First has high confidence despite mismatch
        assert 0.4 < mrr < 0.6


class TestPrecisionRecall:
    """Tests for precision and recall functions."""

    def test_precision_perfect(self):
        """Test precision with perfect predictions."""
        predicted = ["a", "b", "c"]
        expected = ["a", "b", "c"]
        assert precision(predicted, expected) == 1.0

    def test_precision_partial(self):
        """Test precision with partial overlap."""
        predicted = ["a", "b", "c"]
        expected = ["a", "b"]
        # TP = 2, TP+FP = 3, precision = 2/3
        assert abs(precision(predicted, expected) - 2/3) < 0.01

    def test_precision_empty_predicted(self):
        """Test precision with empty predictions."""
        predicted = []
        expected = ["a", "b"]
        assert precision(predicted, expected) == 0.0

    def test_recall_perfect(self):
        """Test recall with perfect predictions."""
        predicted = ["a", "b", "c"]
        expected = ["a", "b", "c"]
        assert recall(predicted, expected) == 1.0

    def test_recall_partial(self):
        """Test recall with partial overlap."""
        predicted = ["a", "b"]
        expected = ["a", "b", "c"]
        # TP = 2, TP+FN = 3, recall = 2/3
        assert abs(recall(predicted, expected) - 2/3) < 0.01

    def test_recall_empty_expected(self):
        """Test recall with empty expected."""
        predicted = ["a", "b"]
        expected = []
        assert recall(predicted, expected) == 0.0


class TestF1Score:
    """Tests for f1_score function."""

    def test_f1_perfect(self):
        """Test F1 with perfect precision and recall."""
        f1 = f1_score(1.0, 1.0)
        assert f1 == 1.0

    def test_f1_zero(self):
        """Test F1 with zero precision and recall."""
        f1 = f1_score(0.0, 0.0)
        assert f1 == 0.0

    def test_f1_balanced(self):
        """Test F1 with balanced precision and recall."""
        f1 = f1_score(0.8, 0.8)
        assert abs(f1 - 0.8) < 0.01

    def test_f1_imbalanced(self):
        """Test F1 with imbalanced precision and recall."""
        prec = 0.9
        rec = 0.7
        expected = 2 * (prec * rec) / (prec + rec)
        f1 = f1_score(prec, rec)
        assert abs(f1 - expected) < 0.01

    def test_f1_partial(self):
        """Test F1 with partial scores."""
        f1 = f1_score(0.5, 0.5)
        assert f1 == 0.5


class TestAggregateQaResults:
    """Tests for aggregate_qa_results function."""

    def test_aggregate_empty_results(self):
        """Test aggregating empty results."""
        results = []
        stats = aggregate_qa_results(results)
        assert stats.total_questions == 0
        assert stats.successful_answers == 0

    def test_aggregate_all_successful(self):
        """Test aggregating all successful results."""
        results = [
            QAResult(
                question="q1",
                answer="a1",
                confidence=0.9,
                processing_time_ms=100.0,
                model_used="gpt-4o-mini",
            ),
            QAResult(
                question="q2",
                answer="a2",
                confidence=0.8,
                processing_time_ms=120.0,
                model_used="gpt-4o-mini",
            ),
        ]
        stats = aggregate_qa_results(results)
        assert stats.total_questions == 2
        assert stats.successful_answers == 2
        assert stats.failed_answers == 0

    def test_aggregate_with_errors(self):
        """Test aggregating results with errors."""
        results = [
            QAResult(
                question="q1",
                answer="a1",
                confidence=0.9,
                processing_time_ms=100.0,
            ),
            QAResult(
                question="q2",
                answer="",
                error="Timeout",
                confidence=0.0,
            ),
        ]
        stats = aggregate_qa_results(results)
        assert stats.total_questions == 2
        assert stats.successful_answers == 1
        assert stats.failed_answers == 1

    def test_aggregate_confidence_stats(self):
        """Test confidence aggregation."""
        results = [
            QAResult(
                question="q1",
                answer="a1",
                confidence=0.9,
                processing_time_ms=100.0,
            ),
            QAResult(
                question="q2",
                answer="a2",
                confidence=0.7,
                processing_time_ms=120.0,
            ),
        ]
        stats = aggregate_qa_results(results)
        # Average confidence = (0.9 + 0.7) / 2 = 0.8
        assert abs(stats.average_confidence - 0.8) < 0.01


class TestAggregateUpdateResults:
    """Tests for aggregate_update_results function."""

    def test_aggregate_empty_results(self):
        """Test aggregating empty update results."""
        results = []
        stats = aggregate_update_results(results)
        assert stats.total_documents == 0
        assert stats.successful_documents == 0

    def test_aggregate_all_successful(self):
        """Test aggregating all successful updates."""
        results = [
            UpdateResult(
                document_id="doc1",
                success=True,
                entities_found=5,
                relations_found=3,
                entities_added=5,
                relations_added=3,
                processing_time_ms=100.0,
            ),
            UpdateResult(
                document_id="doc2",
                success=True,
                entities_found=7,
                relations_found=4,
                entities_added=7,
                relations_added=4,
                processing_time_ms=150.0,
            ),
        ]
        stats = aggregate_update_results(results)
        assert stats.total_documents == 2
        assert stats.successful_documents == 2
        assert stats.failed_documents == 0
        assert stats.entities_extracted == 12
        assert stats.relations_extracted == 7

    def test_aggregate_with_failures(self):
        """Test aggregating results with failures."""
        results = [
            UpdateResult(
                document_id="doc1",
                success=True,
                entities_found=5,
                relations_found=3,
                entities_added=5,
                relations_added=3,
            ),
            UpdateResult(
                document_id="doc2",
                success=False,
                error="Parse error",
            ),
        ]
        stats = aggregate_update_results(results)
        assert stats.total_documents == 2
        assert stats.successful_documents == 1
        assert stats.failed_documents == 1


class TestEvaluatePredictions:
    """Tests for evaluate_predictions function."""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test evaluate_predictions identifies exact matches."""
        from mellea_contribs.kg.utils import evaluate_predictions

        predictions = [
            {"query": "What is 2+2?", "answer": "4", "answer_aliases": ["4", "four"]},
        ]
        result = await evaluate_predictions(None, predictions)
        assert result[0]["correct"] is True
        assert result[0]["eval_method"] == "exact"

    @pytest.mark.asyncio
    async def test_fuzzy_match(self):
        """Test evaluate_predictions falls back to fuzzy match."""
        from mellea_contribs.kg.utils import evaluate_predictions

        predictions = [
            {"query": "Capital of France?", "answer": "Pari", "answer_aliases": ["Paris"]},
        ]
        result = await evaluate_predictions(None, predictions)
        assert result[0]["correct"] is True
        assert result[0]["eval_method"] == "fuzzy"

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Test evaluate_predictions handles wrong answers."""
        from mellea_contribs.kg.utils import evaluate_predictions

        predictions = [
            {"query": "What color is the sky?", "answer": "red", "answer_aliases": ["blue"]},
        ]
        result = await evaluate_predictions(None, predictions)
        assert result[0]["correct"] is False

    @pytest.mark.asyncio
    async def test_missing_gold_answers(self):
        """Test evaluate_predictions when no gold answers are provided."""
        from mellea_contribs.kg.utils import evaluate_predictions

        predictions = [{"query": "What is X?", "answer": "something"}]
        result = await evaluate_predictions(None, predictions)
        assert result[0]["correct"] is False
        assert result[0]["eval_method"] == "none"

    @pytest.mark.asyncio
    async def test_preserves_original_fields(self):
        """Test that evaluate_predictions keeps original dict fields."""
        from mellea_contribs.kg.utils import evaluate_predictions

        predictions = [
            {
                "query": "Q?",
                "answer": "A",
                "answer_aliases": ["A"],
                "custom_field": "preserved",
            }
        ]
        result = await evaluate_predictions(None, predictions)
        assert result[0]["custom_field"] == "preserved"
        assert "correct" in result[0]
        assert "eval_method" in result[0]

    @pytest.mark.asyncio
    async def test_custom_key_names(self):
        """Test evaluate_predictions with non-default key names."""
        from mellea_contribs.kg.utils import evaluate_predictions

        predictions = [
            {"question": "Q?", "prediction": "answer", "expected": ["answer"]},
        ]
        result = await evaluate_predictions(
            None,
            predictions,
            query_key="question",
            answer_key="prediction",
            gold_key="expected",
        )
        assert result[0]["correct"] is True

    @pytest.mark.asyncio
    async def test_string_gold_coerced_to_list(self):
        """Test evaluate_predictions accepts a string for gold answers."""
        from mellea_contribs.kg.utils import evaluate_predictions

        predictions = [
            {"query": "Q?", "answer": "Paris", "answer_aliases": "Paris"},
        ]
        result = await evaluate_predictions(None, predictions)
        assert result[0]["correct"] is True

    @pytest.mark.asyncio
    async def test_multiple_predictions(self):
        """Test evaluate_predictions processes all items in the list."""
        from mellea_contribs.kg.utils import evaluate_predictions

        predictions = [
            {"query": "Q1", "answer": "correct", "answer_aliases": ["correct"]},
            {"query": "Q2", "answer": "wrong", "answer_aliases": ["right"]},
        ]
        result = await evaluate_predictions(None, predictions)
        assert len(result) == 2
        assert result[0]["correct"] is True
        assert result[1]["correct"] is False


class TestIntegration:
    """Integration tests for eval_utils functions."""

    def test_workflow_qa_evaluation(self):
        """Test QA evaluation workflow."""
        # Create results
        results = [
            QAResult(
                question="What is 2+2?",
                answer="4",
                confidence=0.95,
                processing_time_ms=50.0,
            ),
            QAResult(
                question="What is the capital of France?",
                answer="Paris",
                confidence=0.9,
                processing_time_ms=60.0,
            ),
        ]

        # Aggregate
        stats = aggregate_qa_results(results)

        # Verify
        assert stats.total_questions == 2
        assert stats.successful_answers == 2
        assert 0.9 < stats.average_confidence < 0.95

    def test_workflow_update_evaluation(self):
        """Test update operation evaluation workflow."""
        # Create results
        results = [
            UpdateResult(
                document_id="doc1",
                success=True,
                entities_found=10,
                relations_found=8,
                entities_added=10,
                relations_added=8,
                processing_time_ms=200.0,
            ),
            UpdateResult(
                document_id="doc2",
                success=True,
                entities_found=12,
                relations_found=9,
                entities_added=12,
                relations_added=9,
                processing_time_ms=250.0,
            ),
        ]

        # Aggregate
        stats = aggregate_update_results(results)

        # Verify
        assert stats.total_documents == 2
        assert stats.entities_extracted == 22
        assert stats.relations_extracted == 17
