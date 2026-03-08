"""Tests for Phase 1 QA configuration and result models."""

import pytest

from mellea_contribs.kg.qa_models import (
    QAConfig,
    QASessionConfig,
    QADatasetConfig,
    QAResult,
    QAStats,
)
from mellea_contribs.kg.models import DirectAnswer, Entity


class TestQAConfig:
    """Tests for QAConfig model."""

    def test_create_qa_config_defaults(self):
        """Test creating QAConfig with defaults."""
        config = QAConfig()

        assert config.route_count == 3
        assert config.depth == 2
        assert config.width == 5
        assert config.domain is None
        assert config.consensus_threshold == 0.7
        assert config.format_style == "detailed"

    def test_create_qa_config_custom(self):
        """Test creating QAConfig with custom values."""
        config = QAConfig(
            route_count=5,
            depth=3,
            width=10,
            domain="movie",
            consensus_threshold=0.8,
            format_style="concise",
        )

        assert config.route_count == 5
        assert config.depth == 3
        assert config.width == 10
        assert config.domain == "movie"
        assert config.consensus_threshold == 0.8
        assert config.format_style == "concise"

    def test_qa_config_max_repair_attempts(self):
        """Test QAConfig max_repair_attempts field."""
        config = QAConfig(max_repair_attempts=3)
        assert config.max_repair_attempts == 3


class TestQASessionConfig:
    """Tests for QASessionConfig model."""

    def test_create_qa_session_config_defaults(self):
        """Test creating QASessionConfig with defaults."""
        config = QASessionConfig()

        assert config.llm_model is not None
        assert config.temperature == 0.1
        assert config.max_tokens == 2048
        assert config.eval_model is not None

    def test_create_qa_session_config_custom(self):
        """Test creating QASessionConfig with custom values."""
        config = QASessionConfig(
            llm_model="gpt-4",
            temperature=0.5,
            max_tokens=4096,
            eval_model="claude-3-sonnet",
            evaluation_threshold=0.75,
        )

        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096
        assert config.eval_model == "claude-3-sonnet"
        assert config.evaluation_threshold == 0.75

    def test_qa_session_config_few_shot_examples(self):
        """Test QASessionConfig with few-shot examples."""
        examples = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        config = QASessionConfig(few_shot_examples=examples)
        assert config.few_shot_examples == examples


class TestQADatasetConfig:
    """Tests for QADatasetConfig model."""

    def test_create_qa_dataset_config_defaults(self):
        """Test creating QADatasetConfig with defaults."""
        config = QADatasetConfig()

        assert config.dataset_path is not None
        assert config.batch_size == 32
        assert config.output_path is not None
        assert config.num_workers == 4
        assert config.shuffle is True

    def test_create_qa_dataset_config_custom(self):
        """Test creating QADatasetConfig with custom values."""
        config = QADatasetConfig(
            dataset_path="data/qa.jsonl",
            batch_size=64,
            output_path="output/qa_results.json",
            num_workers=8,
            shuffle=False,
            max_examples=100,
            skip_errors=True,
        )

        assert config.dataset_path == "data/qa.jsonl"
        assert config.batch_size == 64
        assert config.output_path == "output/qa_results.json"
        assert config.num_workers == 8
        assert config.shuffle is False
        assert config.max_examples == 100
        assert config.skip_errors is True


class TestQAResult:
    """Tests for QAResult model."""

    def test_create_qa_result_minimal(self):
        """Test creating QAResult with minimal fields."""
        result = QAResult(
            question="What is AI?",
            answer="Artificial Intelligence",
        )

        assert result.question == "What is AI?"
        assert result.answer == "Artificial Intelligence"

    def test_create_qa_result_comprehensive(self):
        """Test creating QAResult with comprehensive fields."""
        result = QAResult(
            question="What is AI?",
            answer="Artificial Intelligence",
            confidence=0.95,
            cypher_query="MATCH (n) RETURN n",
            graph_evidence=["node1", "node2"],
            reasoning="Based on knowledge graph",
        )

        assert result.question == "What is AI?"
        assert result.answer == "Artificial Intelligence"
        assert result.confidence == 0.95
        assert result.cypher_query == "MATCH (n) RETURN n"
        assert result.graph_evidence == ["node1", "node2"]
        assert result.reasoning == "Based on knowledge graph"

    def test_qa_result_with_direct_answer(self):
        """Test QAResult with DirectAnswer object."""
        direct_answer = DirectAnswer(
            answer="42",
            confidence=0.9,
            supporting_facts=["fact1", "fact2"],
        )
        result = QAResult(
            question="What is the answer?",
            answer="42",
            direct_answer=direct_answer,
        )

        assert result.direct_answer == direct_answer
        assert result.direct_answer.confidence == 0.9


class TestQAStats:
    """Tests for QAStats model."""

    def test_create_qa_stats_defaults(self):
        """Test creating QAStats with defaults."""
        stats = QAStats()

        assert stats.total_questions == 0
        assert stats.exact_match_count == 0
        assert stats.partial_match_count == 0
        assert stats.no_match_count == 0

    def test_create_qa_stats_custom(self):
        """Test creating QAStats with custom values."""
        stats = QAStats(
            total_questions=100,
            exact_match_count=80,
            partial_match_count=15,
            no_match_count=5,
            mean_reciprocal_rank=0.87,
            total_time_ms=5000,
            avg_time_per_question_ms=50,
        )

        assert stats.total_questions == 100
        assert stats.exact_match_count == 80
        assert stats.partial_match_count == 15
        assert stats.no_match_count == 5
        assert stats.mean_reciprocal_rank == 0.87
        assert stats.total_time_ms == 5000
        assert stats.avg_time_per_question_ms == 50

    def test_qa_stats_metrics(self):
        """Test QAStats metric calculations."""
        stats = QAStats(
            total_questions=100,
            exact_match_count=50,
            partial_match_count=30,
        )

        # Test derived metrics
        assert stats.exact_match_count == 50
        assert stats.partial_match_count == 30


class TestQAIntegration:
    """Integration tests for QA models."""

    def test_qa_config_and_session_together(self):
        """Test using QAConfig with QASessionConfig."""
        qa_config = QAConfig(domain="movie", route_count=3)
        session_config = QASessionConfig(temperature=0.1)

        assert qa_config.domain == "movie"
        assert session_config.temperature == 0.1

    def test_qa_result_list_with_stats(self):
        """Test creating multiple QA results with stats."""
        results = [
            QAResult(question="Q1", answer="A1", confidence=0.9),
            QAResult(question="Q2", answer="A2", confidence=0.8),
            QAResult(question="Q3", answer="A3", confidence=0.95),
        ]

        stats = QAStats(
            total_questions=len(results),
            exact_match_count=2,
            avg_time_per_question_ms=100,
        )

        assert len(results) == 3
        assert stats.total_questions == 3
        assert stats.exact_match_count == 2
