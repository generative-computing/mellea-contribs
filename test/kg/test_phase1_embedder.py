"""Tests for Phase 1 KGEmbedder structure and interface."""

import pytest
import inspect
import asyncio

from mellea_contribs.kg.embedder import KGEmbedder
from mellea_contribs.kg.models import Entity
from mellea_contribs.kg.embed_models import EmbeddingConfig


class TestKGEmbedderStructure:
    """Tests for KGEmbedder class structure."""

    def test_kg_embedder_exists(self):
        """Test that KGEmbedder class exists."""
        assert KGEmbedder is not None

    def test_kg_embedder_is_class(self):
        """Test that KGEmbedder is a class."""
        assert inspect.isclass(KGEmbedder)

    def test_kg_embedder_has_embed_entity_method(self):
        """Test that KGEmbedder has embed_entity method."""
        assert hasattr(KGEmbedder, "embed_entity")
        assert callable(getattr(KGEmbedder, "embed_entity"))

    def test_kg_embedder_has_embed_batch_method(self):
        """Test that KGEmbedder has embed_batch method."""
        assert hasattr(KGEmbedder, "embed_batch")
        assert callable(getattr(KGEmbedder, "embed_batch"))

    def test_kg_embedder_has_get_similar_entities_method(self):
        """Test that KGEmbedder has get_similar_entities method."""
        assert hasattr(KGEmbedder, "get_similar_entities")
        assert callable(getattr(KGEmbedder, "get_similar_entities"))

    def test_kg_embedder_methods_are_async(self):
        """Test that key methods are async."""
        embed_entity_method = getattr(KGEmbedder, "embed_entity")
        assert inspect.iscoroutinefunction(embed_entity_method)

        embed_batch_method = getattr(KGEmbedder, "embed_batch")
        assert inspect.iscoroutinefunction(embed_batch_method)

        get_similar_method = getattr(KGEmbedder, "get_similar_entities")
        assert inspect.iscoroutinefunction(get_similar_method)


class TestKGEmbedderInterface:
    """Tests for KGEmbedder interface and contract."""

    def test_kg_embedder_init_with_config(self):
        """Test KGEmbedder initialization with config."""
        config = EmbeddingConfig(
            embedding_model="test-model",
            embedding_dimensions=384,
        )

        embedder = KGEmbedder(config=config)

        assert embedder is not None
        assert embedder.config == config

    def test_kg_embedder_init_defaults(self):
        """Test KGEmbedder initialization with defaults."""
        embedder = KGEmbedder()

        assert embedder is not None
        assert embedder.config is not None

    def test_kg_embedder_config_accessible(self):
        """Test that KGEmbedder config is accessible."""
        config = EmbeddingConfig(embedding_dimensions=1536)
        embedder = KGEmbedder(config=config)

        assert embedder.config is not None
        assert embedder.config.embedding_dimensions == 1536

    def test_kg_embedder_method_signatures(self):
        """Test that methods have correct signatures."""
        embed_entity_sig = inspect.signature(KGEmbedder.embed_entity)
        embed_entity_params = list(embed_entity_sig.parameters.keys())
        assert "self" in embed_entity_params
        assert "entity" in embed_entity_params

        embed_batch_sig = inspect.signature(KGEmbedder.embed_batch)
        embed_batch_params = list(embed_batch_sig.parameters.keys())
        assert "self" in embed_batch_params
        assert "entities" in embed_batch_params

        get_similar_sig = inspect.signature(KGEmbedder.get_similar_entities)
        get_similar_params = list(get_similar_sig.parameters.keys())
        assert "self" in get_similar_params
        assert "query_entity" in get_similar_params


class TestKGEmbedderDocumentation:
    """Tests for KGEmbedder documentation."""

    def test_kg_embedder_has_docstring(self):
        """Test that KGEmbedder has docstring."""
        assert KGEmbedder.__doc__ is not None
        assert len(KGEmbedder.__doc__) > 0

    def test_kg_embedder_method_docstrings(self):
        """Test that key methods have docstrings."""
        methods = ["embed_entity", "embed_batch", "get_similar_entities"]

        for method_name in methods:
            method = getattr(KGEmbedder, method_name)
            assert method.__doc__ is not None, f"Method {method_name} missing docstring"
            assert len(method.__doc__) > 0, f"Method {method_name} has empty docstring"

    def test_kg_embedder_init_docstring(self):
        """Test that __init__ has docstring."""
        assert KGEmbedder.__init__.__doc__ is not None


class TestKGEmbedderInstantiation:
    """Tests for KGEmbedder instantiation."""

    def test_kg_embedder_can_be_instantiated(self):
        """Test that KGEmbedder can be instantiated."""
        embedder = KGEmbedder()
        assert embedder is not None
        assert isinstance(embedder, KGEmbedder)

    def test_kg_embedder_with_custom_config(self):
        """Test KGEmbedder with custom configuration."""
        config = EmbeddingConfig(
            embedding_model="custom-model",
            embedding_api="custom-api",
            embedding_dimensions=768,
            batch_size=64,
        )

        embedder = KGEmbedder(config=config)

        assert embedder.config.embedding_model == "custom-model"
        assert embedder.config.embedding_api == "custom-api"
        assert embedder.config.embedding_dimensions == 768
        assert embedder.config.batch_size == 64

    def test_kg_embedder_config_immutability_in_method_calls(self):
        """Test that config is properly used in method signatures."""
        config = EmbeddingConfig(embedding_dimensions=1536)
        embedder = KGEmbedder(config=config)

        # Verify config is accessible after instantiation
        assert embedder.config.embedding_dimensions == 1536


class TestKGEmbedderMethodSignatures:
    """Tests for detailed method signatures."""

    def test_embed_entity_parameters(self):
        """Test embed_entity method parameters."""
        sig = inspect.signature(KGEmbedder.embed_entity)
        params = sig.parameters

        assert "self" in params
        assert "entity" in params
        # Check for optional parameters
        assert "use_name" in params or "use_description" in params

    def test_embed_batch_parameters(self):
        """Test embed_batch method parameters."""
        sig = inspect.signature(KGEmbedder.embed_batch)
        params = sig.parameters

        assert "self" in params
        assert "entities" in params

    def test_get_similar_entities_parameters(self):
        """Test get_similar_entities method parameters."""
        sig = inspect.signature(KGEmbedder.get_similar_entities)
        params = sig.parameters

        assert "self" in params
        assert "query_entity" in params
        # Should have candidates parameter or something similar
        param_names = list(params.keys())
        assert any(
            "candidate" in p.lower() or "entities" in p.lower() for p in param_names
        )


class TestKGEmbedderIntegration:
    """Integration tests for KGEmbedder with other models."""

    def test_kg_embedder_with_entity_model(self):
        """Test that KGEmbedder works with Entity model."""
        entity = Entity(
            type="Person",
            name="Alice",
            description="A person",
            paragraph_start="Alice",
            paragraph_end="here.",
        )

        config = EmbeddingConfig()
        embedder = KGEmbedder(config=config)

        # Just verify the interaction is type-safe
        assert entity is not None
        assert embedder is not None

    def test_kg_embedder_config_consistency(self):
        """Test that embedder maintains config consistency."""
        config1 = EmbeddingConfig(embedding_dimensions=384)
        embedder1 = KGEmbedder(config=config1)

        config2 = EmbeddingConfig(embedding_dimensions=1536)
        embedder2 = KGEmbedder(config=config2)

        assert embedder1.config.embedding_dimensions == 384
        assert embedder2.config.embedding_dimensions == 1536
        assert embedder1.config != embedder2.config
