"""Tests for Phase 1 KG update configuration and result models."""

import pytest
from datetime import datetime

from mellea_contribs.kg.updater_models import (
    UpdateConfig,
    UpdateSessionConfig,
    UpdateStats,
    MergeConflict,
    UpdateResult,
    UpdateBatchResult,
)


class TestUpdateConfig:
    """Tests for UpdateConfig model."""

    def test_create_update_config_defaults(self):
        """Test creating UpdateConfig with defaults."""
        config = UpdateConfig()

        assert config.batch_size == 32
        assert config.merge_strategy == "merge_if_similar"
        assert config.similarity_threshold == 0.8

    def test_create_update_config_custom(self):
        """Test creating UpdateConfig with custom values."""
        config = UpdateConfig(
            batch_size=64,
            merge_strategy="skip",
            similarity_threshold=0.75,
            domain="movie",
        )

        assert config.batch_size == 64
        assert config.merge_strategy == "skip"
        assert config.similarity_threshold == 0.75
        assert config.domain == "movie"

    def test_update_config_merge_strategies(self):
        """Test UpdateConfig with different merge strategies."""
        strategies = ["merge_if_similar", "skip", "overwrite", "create_variant"]

        for strategy in strategies:
            config = UpdateConfig(merge_strategy=strategy)
            assert config.merge_strategy == strategy

    def test_update_config_entity_types(self):
        """Test UpdateConfig with entity and relation types."""
        config = UpdateConfig(
            entity_types=["Person", "Movie", "Award"],
            relation_types=["directed_by", "acted_in", "won"],
        )

        assert config.entity_types == ["Person", "Movie", "Award"]
        assert config.relation_types == ["directed_by", "acted_in", "won"]


class TestUpdateSessionConfig:
    """Tests for UpdateSessionConfig model."""

    def test_create_update_session_config_defaults(self):
        """Test creating UpdateSessionConfig with defaults."""
        config = UpdateSessionConfig()

        assert config.extraction_model is not None
        assert config.alignment_model is not None
        assert config.merge_decision_model is not None

    def test_create_update_session_config_custom(self):
        """Test creating UpdateSessionConfig with custom values."""
        config = UpdateSessionConfig(
            extraction_model="gpt-4",
            extraction_temperature=0.1,
            alignment_model="claude-3-sonnet",
            alignment_temperature=0.2,
            merge_decision_model="gpt-4-turbo",
            merge_decision_temperature=0.1,
        )

        assert config.extraction_model == "gpt-4"
        assert config.extraction_temperature == 0.1
        assert config.alignment_model == "claude-3-sonnet"
        assert config.alignment_temperature == 0.2
        assert config.merge_decision_model == "gpt-4-turbo"
        assert config.merge_decision_temperature == 0.1


class TestUpdateStats:
    """Tests for UpdateStats model."""

    def test_create_update_stats_defaults(self):
        """Test creating UpdateStats with defaults."""
        stats = UpdateStats()

        assert stats.entities_extracted == 0
        assert stats.entities_aligned == 0
        assert stats.entities_merged == 0
        assert stats.relations_extracted == 0

    def test_create_update_stats_custom(self):
        """Test creating UpdateStats with custom values."""
        stats = UpdateStats(
            entities_extracted=100,
            entities_new=30,
            entities_aligned=70,
            entities_merged=50,
            entities_skipped=20,
            relations_extracted=80,
            relations_new=20,
            relations_aligned=60,
            relations_merged=45,
            relations_skipped=15,
            processing_time_ms=5000,
        )

        assert stats.entities_extracted == 100
        assert stats.entities_new == 30
        assert stats.entities_aligned == 70
        assert stats.entities_merged == 50
        assert stats.relations_extracted == 80
        assert stats.relations_new == 20


class TestMergeConflict:
    """Tests for MergeConflict model."""

    def test_create_merge_conflict(self):
        """Test creating a MergeConflict."""
        conflict = MergeConflict(
            source_id="entity-1",
            target_id="entity-2",
            conflict_type="entity_merge",
            similarity_score=0.85,
            decision="merged",
            reason="High similarity score",
        )

        assert conflict.source_id == "entity-1"
        assert conflict.target_id == "entity-2"
        assert conflict.conflict_type == "entity_merge"
        assert conflict.similarity_score == 0.85
        assert conflict.decision == "merged"
        assert conflict.reason == "High similarity score"

    def test_merge_conflict_with_timestamp(self):
        """Test MergeConflict with timestamp."""
        conflict = MergeConflict(
            source_id="entity-1",
            target_id="entity-2",
            conflict_type="relation_merge",
            similarity_score=0.9,
            decision="skipped",
            reason="User review needed",
            timestamp="2026-03-08T10:30:00Z",
        )

        assert conflict.timestamp == "2026-03-08T10:30:00Z"
        assert conflict.decision == "skipped"

    def test_merge_conflict_types(self):
        """Test different merge conflict types."""
        types = ["entity_merge", "relation_merge", "property_conflict", "temporal_conflict"]

        for conflict_type in types:
            conflict = MergeConflict(
                source_id="a",
                target_id="b",
                conflict_type=conflict_type,
                similarity_score=0.8,
                decision="merged",
                reason="Test",
            )
            assert conflict.conflict_type == conflict_type


class TestUpdateResult:
    """Tests for UpdateResult model."""

    def test_create_update_result_minimal(self):
        """Test creating UpdateResult with minimal fields."""
        result = UpdateResult(
            document_id="doc-1",
            entities_added=10,
            entities_merged=5,
        )

        assert result.document_id == "doc-1"
        assert result.entities_added == 10
        assert result.entities_merged == 5

    def test_create_update_result_comprehensive(self):
        """Test creating UpdateResult with all fields."""
        conflicts = [
            MergeConflict(
                source_id="e1",
                target_id="e2",
                conflict_type="entity_merge",
                similarity_score=0.85,
                decision="merged",
                reason="High similarity",
            )
        ]

        result = UpdateResult(
            document_id="doc-1",
            entities_added=10,
            entities_merged=5,
            entities_skipped=2,
            relations_added=8,
            relations_merged=3,
            relations_skipped=1,
            conflicts=conflicts,
            processing_time_ms=1000,
        )

        assert result.document_id == "doc-1"
        assert result.entities_added == 10
        assert result.entities_merged == 5
        assert result.entities_skipped == 2
        assert result.relations_added == 8
        assert len(result.conflicts) == 1
        assert result.processing_time_ms == 1000

    def test_update_result_tracking(self):
        """Test UpdateResult for tracking changes."""
        result = UpdateResult(
            document_id="doc-movie",
            entities_added=15,
            entities_merged=8,
            relations_added=12,
            relations_merged=6,
        )

        # Verify all tracking fields are present
        assert result.entities_added > 0
        assert result.entities_merged > 0
        assert result.relations_added > 0
        assert result.relations_merged > 0


class TestUpdateBatchResult:
    """Tests for UpdateBatchResult model."""

    def test_create_update_batch_result_defaults(self):
        """Test creating UpdateBatchResult with defaults."""
        batch_result = UpdateBatchResult()

        assert batch_result.total_documents == 0
        assert batch_result.successful_documents == 0
        assert batch_result.failed_documents == 0
        assert batch_result.stats is not None

    def test_create_update_batch_result_custom(self):
        """Test creating UpdateBatchResult with custom values."""
        stats = UpdateStats(
            entities_extracted=100,
            entities_merged=50,
            relations_extracted=80,
            relations_merged=40,
        )

        results = [
            UpdateResult(
                document_id=f"doc-{i}",
                entities_added=10,
                entities_merged=5,
            )
            for i in range(3)
        ]

        batch_result = UpdateBatchResult(
            total_documents=3,
            successful_documents=3,
            failed_documents=0,
            stats=stats,
            results=results,
            total_time_ms=3000,
            avg_time_per_document_ms=1000,
        )

        assert batch_result.total_documents == 3
        assert batch_result.successful_documents == 3
        assert batch_result.failed_documents == 0
        assert len(batch_result.results) == 3
        assert batch_result.total_time_ms == 3000

    def test_update_batch_result_aggregation(self):
        """Test UpdateBatchResult for aggregating multiple documents."""
        results = [
            UpdateResult(
                document_id=f"doc-{i}",
                entities_added=5,
                entities_merged=2,
                relations_added=3,
                relations_merged=1,
            )
            for i in range(5)
        ]

        batch_result = UpdateBatchResult(
            total_documents=5,
            successful_documents=5,
            results=results,
        )

        assert batch_result.total_documents == 5
        assert len(batch_result.results) == 5


class TestUpdateIntegration:
    """Integration tests for Update models."""

    def test_update_config_with_session_config(self):
        """Test using UpdateConfig with UpdateSessionConfig."""
        config = UpdateConfig(
            batch_size=64,
            merge_strategy="merge_if_similar",
            domain="movie",
        )

        session = UpdateSessionConfig(
            extraction_model="gpt-4",
            merge_decision_model="claude-3",
        )

        assert config.domain == "movie"
        assert config.batch_size == 64
        assert session.extraction_model == "gpt-4"

    def test_merge_conflict_in_update_result(self):
        """Test MergeConflict within UpdateResult."""
        conflicts = [
            MergeConflict(
                source_id="entity-alice",
                target_id="entity-alice-2",
                conflict_type="entity_merge",
                similarity_score=0.92,
                decision="merged",
                reason="Duplicate person entry",
            ),
            MergeConflict(
                source_id="relation-directed",
                target_id="relation-directed-2",
                conflict_type="relation_merge",
                similarity_score=0.88,
                decision="merged",
                reason="Same relation type",
            ),
        ]

        result = UpdateResult(
            document_id="doc-movie",
            entities_added=5,
            entities_merged=3,
            relations_added=4,
            relations_merged=2,
            conflicts=conflicts,
        )

        assert len(result.conflicts) == 2
        assert result.conflicts[0].conflict_type == "entity_merge"
        assert result.conflicts[1].conflict_type == "relation_merge"

    def test_update_result_in_batch_result(self):
        """Test UpdateResult within UpdateBatchResult."""
        results = [
            UpdateResult(
                document_id=f"doc-{i}",
                entities_added=10 + i,
                entities_merged=5 + i,
                relations_added=8 + i,
                relations_merged=3 + i,
            )
            for i in range(3)
        ]

        batch = UpdateBatchResult(
            total_documents=3,
            successful_documents=3,
            results=results,
        )

        assert len(batch.results) == 3
        for i, result in enumerate(batch.results):
            assert result.entities_added == 10 + i
            assert result.entities_merged == 5 + i
