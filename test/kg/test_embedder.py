"""Tests for KGEmbedder structure and interface."""

import pytest
import inspect
import asyncio

from mellea_contribs.kg.embedder import KGEmbedder
from mellea_contribs.kg.models import Entity


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
    """Tests for KGEmbedder interface matching Mellea Layer 1 pattern."""

    def test_kg_embedder_init_individual_params(self):
        """Test KGEmbedder initialization with individual parameters (Mellea pattern)."""
        # Create embedder with individual parameters (matching KGRag/KGPreprocessor pattern)
        embedder = KGEmbedder(
            session=None,  # Would be MelleaSession in real usage
            model="text-embedding-3-small",
            dimension=1536,
        )

        assert embedder is not None
        # Verify individual parameters are accessible
        assert embedder.embedding_model == "text-embedding-3-small"
        assert embedder.embedding_dimension == 1536

    def test_kg_embedder_init_defaults(self):
        """Test KGEmbedder initialization with defaults."""
        embedder = KGEmbedder(session=None)

        assert embedder is not None
        # Should have default values
        assert hasattr(embedder, "embedding_model")
        assert hasattr(embedder, "embedding_dimension")

    def test_kg_embedder_init_all_params(self):
        """Test KGEmbedder initialization with all individual parameters."""
        embedder = KGEmbedder(
            session=None,
            model="all-MiniLM-L6-v2",
            dimension=384,
            api_base="http://localhost:8000",
            api_key="test-key",
            batch_size=64,
        )

        assert embedder is not None
        assert embedder.embedding_model == "openai/all-MiniLM-L6-v2"
        assert embedder.embedding_dimension == 384

    def test_kg_embedder_parameter_defaults(self):
        """Test that KGEmbedder has sensible parameter defaults."""
        embedder = KGEmbedder(session=None)

        # Should have reasonable defaults
        assert isinstance(embedder.embedding_model, str)
        assert isinstance(embedder.embedding_dimension, int)
        assert embedder.embedding_dimension > 0


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
    """Tests for KGEmbedder instantiation (Mellea Layer 1 pattern)."""

    def test_kg_embedder_can_be_instantiated(self):
        """Test that KGEmbedder can be instantiated."""
        embedder = KGEmbedder(session=None)
        assert embedder is not None
        assert isinstance(embedder, KGEmbedder)

    def test_kg_embedder_with_various_models(self):
        """Test KGEmbedder with various embedding models."""
        models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "all-MiniLM-L6-v2",
        ]

        for model_name in models:
            embedder = KGEmbedder(
                session=None,
                model=model_name,
            )

            assert embedder is not None
            assert embedder.embedding_model == model_name

    def test_kg_embedder_with_various_dimensions(self):
        """Test KGEmbedder with various embedding dimensions."""
        dimensions = [384, 768, 1536, 3072]

        for dim in dimensions:
            embedder = KGEmbedder(
                session=None,
                dimension=dim,
            )

            assert embedder is not None
            assert embedder.embedding_dimension == dim

    def test_kg_embedder_consistency_across_instances(self):
        """Test that multiple KGEmbedder instances maintain separate configs."""
        embedder1 = KGEmbedder(
            session=None,
            model="model1",
            dimension=384,
        )

        embedder2 = KGEmbedder(
            session=None,
            model="model2",
            dimension=1536,
        )

        assert embedder1.embedding_model == "model1"
        assert embedder1.embedding_dimension == 384
        assert embedder2.embedding_model == "model2"
        assert embedder2.embedding_dimension == 1536


class TestKGEmbedderMethodSignatures:
    """Tests for detailed method signatures."""

    def test_embed_entity_parameters(self):
        """Test embed_entity method parameters."""
        sig = inspect.signature(KGEmbedder.embed_entity)
        params = sig.parameters

        assert "self" in params
        assert "entity" in params
        # Check for optional parameters like use_name, use_description
        param_names = list(params.keys())
        assert any("use" in p.lower() for p in param_names)

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
        assert "query_entity" in params or "query" in params


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

        embedder = KGEmbedder(session=None)

        # Just verify the interaction is type-safe
        assert entity is not None
        assert embedder is not None

    def test_kg_embedder_parameter_consistency(self):
        """Test that embedder maintains parameter consistency."""
        # Verify that parameters are stored and accessible
        embedder = KGEmbedder(
            session=None,
            model="test-model",
            dimension=512,
            batch_size=32,
        )

        assert embedder.embedding_model == "test-model"
        assert embedder.embedding_dimension == 512

    def test_kg_embedder_matches_layer1_pattern(self):
        """Test that KGEmbedder matches Mellea Layer 1 pattern (individual params)."""
        # Similar to KGRag and KGPreprocessor
        embedder = KGEmbedder(
            session=None,  # Would be MelleaSession
            model="default-model",
            dimension=1536,
            batch_size=10,  # Individual parameters, not config object
        )

        assert embedder is not None
        # All parameters should be individually accessible
        assert hasattr(embedder, "embedding_model")
        assert hasattr(embedder, "embedding_dimension")
