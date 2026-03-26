"""Tests for embedding configuration and result models."""

import pytest

from mellea_contribs.kg.embed_models import (
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingSimilarity,
    EmbeddingStats,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig model."""

    def test_create_embedding_config_defaults(self):
        """Test creating EmbeddingConfig with defaults."""
        config = EmbeddingConfig()

        assert config.model == "text-embedding-3-small"
        assert config.dimension == 1536
        assert config.api_base is None
        assert config.api_key is None

    def test_create_embedding_config_openai(self):
        """Test creating EmbeddingConfig for OpenAI."""
        config = EmbeddingConfig(
            model="text-embedding-3-large",
            dimension=3072,
        )

        assert config.model == "text-embedding-3-large"
        assert config.dimension == 3072

    def test_create_embedding_config_custom(self):
        """Test creating EmbeddingConfig with custom values."""
        config = EmbeddingConfig(
            model="all-MiniLM-L6-v2",
            dimension=384,
            api_base="http://localhost:8000",
            api_key="test-key",
        )

        assert config.model == "all-MiniLM-L6-v2"
        assert config.dimension == 384
        assert config.api_base == "http://localhost:8000"
        assert config.api_key == "test-key"

    def test_embedding_config_various_dimensions(self):
        """Test EmbeddingConfig with various embedding dimensions."""
        dimensions = [384, 768, 1024, 1536, 3072]

        for dim in dimensions:
            config = EmbeddingConfig(dimension=dim)
            assert config.dimension == dim


class TestEmbeddingResult:
    """Tests for EmbeddingResult model."""

    def test_create_embedding_result(self):
        """Test creating an EmbeddingResult."""
        embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = EmbeddingResult(
            text="Alice is a person",
            embedding=embedding_vector,
            model="text-embedding-3-small",
            dimension=1536,
        )

        assert result.text == "Alice is a person"
        assert result.embedding == embedding_vector
        assert result.model == "text-embedding-3-small"
        assert result.dimension == 1536

    def test_embedding_result_with_large_vector(self):
        """Test EmbeddingResult with large embedding."""
        result = EmbeddingResult(
            text="Oppenheimer is a 2023 film",
            embedding=[0.1] * 1536,
            model="text-embedding-3-small",
            dimension=1536,
        )

        assert result.text == "Oppenheimer is a 2023 film"
        assert len(result.embedding) == 1536
        assert result.dimension == 1536

    def test_embedding_result_various_sizes(self):
        """Test EmbeddingResult with various embedding sizes."""
        sizes = [384, 768, 1536, 3072]

        for size in sizes:
            result = EmbeddingResult(
                text=f"Test text for embedding {size}",
                embedding=[0.5] * size,
                model="test-model",
                dimension=size,
            )
            assert len(result.embedding) == size
            assert result.dimension == size


class TestEmbeddingSimilarity:
    """Tests for EmbeddingSimilarity model."""

    def test_create_embedding_similarity(self):
        """Test creating an EmbeddingSimilarity."""
        similarity = EmbeddingSimilarity(
            entity_id="entity-1",
            entity_name="Alice",
            similarity_score=0.92,
            entity_type="Person",
        )

        assert similarity.entity_id == "entity-1"
        assert similarity.entity_name == "Alice"
        assert similarity.similarity_score == 0.92
        assert similarity.entity_type == "Person"

    def test_embedding_similarity_high_score(self):
        """Test EmbeddingSimilarity with high similarity score."""
        similarity = EmbeddingSimilarity(
            entity_id="entity-alice",
            entity_name="Alice",
            similarity_score=0.99,
        )

        assert similarity.similarity_score == 0.99

    def test_embedding_similarity_low_score(self):
        """Test EmbeddingSimilarity with low similarity score."""
        similarity = EmbeddingSimilarity(
            entity_id="entity-1",
            entity_name="Alice",
            similarity_score=0.15,
        )

        assert similarity.similarity_score == 0.15

    def test_embedding_similarity_multiple_matches(self):
        """Test creating multiple EmbeddingSimilarity results."""
        similarities = [
            EmbeddingSimilarity(
                entity_id=f"match-{i}",
                entity_name=f"Match {i}",
                similarity_score=0.9 - (i * 0.1),
            )
            for i in range(3)
        ]

        assert len(similarities) == 3
        # Scores should be decreasing
        assert similarities[0].similarity_score > similarities[1].similarity_score
        assert similarities[1].similarity_score > similarities[2].similarity_score


class TestEmbeddingStats:
    """Tests for EmbeddingStats model."""

    def test_create_embedding_stats_defaults(self):
        """Test creating EmbeddingStats with defaults."""
        stats = EmbeddingStats(
            total_entities=0,
            successful_embeddings=0,
            failed_embeddings=0,
            skipped_embeddings=0,
            average_embedding_time=0.0,
            total_time=0.0,
            model_used="test-model",
        )

        assert stats.total_entities == 0
        assert stats.successful_embeddings == 0
        assert stats.failed_embeddings == 0

    def test_create_embedding_stats_custom(self):
        """Test creating EmbeddingStats with custom values."""
        stats = EmbeddingStats(
            total_entities=1000,
            successful_embeddings=950,
            failed_embeddings=30,
            skipped_embeddings=20,
            average_embedding_time=0.01,
            total_time=10.0,
            model_used="text-embedding-3-small",
        )

        assert stats.total_entities == 1000
        assert stats.successful_embeddings == 950
        assert stats.failed_embeddings == 30
        assert stats.skipped_embeddings == 20
        assert stats.average_embedding_time == 0.01
        assert stats.total_time == 10.0

    def test_embedding_stats_success_rate(self):
        """Test EmbeddingStats tracking success."""
        stats = EmbeddingStats(
            total_entities=100,
            successful_embeddings=90,
            failed_embeddings=5,
            skipped_embeddings=5,
            average_embedding_time=0.1,
            total_time=10.0,
            model_used="test-model",
        )

        # Verify totals add up
        total = stats.successful_embeddings + stats.failed_embeddings + stats.skipped_embeddings
        assert total == 100

    def test_embedding_stats_batch_processing(self):
        """Test EmbeddingStats for batch processing metrics."""
        stats = EmbeddingStats(
            total_entities=1000,
            successful_embeddings=950,
            failed_embeddings=50,
            skipped_embeddings=0,
            average_embedding_time=0.005,
            total_time=5.0,
            model_used="all-MiniLM-L6-v2",
        )

        assert stats.total_entities == 1000
        assert stats.average_embedding_time == 0.005
        assert stats.total_time == 5.0


class TestEmbeddingIntegration:
    """Integration tests for embedding models."""

    def test_embedding_config_and_result_together(self):
        """Test using EmbeddingConfig with EmbeddingResult."""
        config = EmbeddingConfig(
            model="text-embedding-3-small",
            dimension=1536,
        )

        result = EmbeddingResult(
            text="Alice",
            embedding=[0.1] * config.dimension,
            model=config.model,
            dimension=config.dimension,
        )

        assert config.dimension == 1536
        assert len(result.embedding) == config.dimension
        assert result.model == config.model

    def test_embedding_result_list_with_stats(self):
        """Test creating multiple EmbeddingResults with stats."""
        config = EmbeddingConfig(
            model="all-MiniLM-L6-v2",
            dimension=384,
        )

        results = [
            EmbeddingResult(
                text=f"Entity {i}",
                embedding=[0.1 + i * 0.01] * config.dimension,
                model=config.model,
                dimension=config.dimension,
            )
            for i in range(10)
        ]

        stats = EmbeddingStats(
            total_entities=len(results),
            successful_embeddings=len(results),
            failed_embeddings=0,
            skipped_embeddings=0,
            average_embedding_time=0.01,
            total_time=0.1,
            model_used=config.model,
        )

        assert len(results) == 10
        assert stats.total_entities == 10
        assert stats.successful_embeddings == 10

    def test_similarity_search_workflow(self):
        """Test similarity search workflow with embeddings."""
        # Query result
        query_result = EmbeddingResult(
            text="Alice",
            embedding=[0.5] * 384,
            model="all-MiniLM-L6-v2",
            dimension=384,
        )

        # Match results
        matches = [
            EmbeddingSimilarity(
                entity_id=f"candidate-{i}",
                entity_name=f"Candidate {i}",
                similarity_score=0.95 - (i * 0.1),
            )
            for i in range(3)
        ]

        assert len(matches) == 3
        assert matches[0].similarity_score > matches[1].similarity_score
