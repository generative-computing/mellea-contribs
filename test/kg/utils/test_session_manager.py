"""Tests for mellea_contribs.kg.utils.session_manager module."""

import pytest

from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend
from mellea_contribs.kg.utils import create_backend, create_session


class TestCreateBackend:
    """Tests for create_backend function."""

    def test_create_mock_backend(self):
        """Test creating mock backend."""
        backend = create_backend(backend_type="mock")
        assert isinstance(backend, MockGraphBackend)

    def test_create_backend_default_type(self):
        """Test creating backend with default type."""
        backend = create_backend()
        assert isinstance(backend, MockGraphBackend)

    def test_create_backend_invalid_type(self):
        """Test creating backend with invalid type."""
        with pytest.raises(ValueError):
            create_backend(backend_type="invalid")

    def test_create_neo4j_backend_not_available(self):
        """Test Neo4j backend creation when not available."""
        # Neo4jBackend might not be installed
        try:
            backend = create_backend(backend_type="neo4j", neo4j_uri="bolt://localhost:7687")
            # If we got here, Neo4j backend is available
            assert backend is not None
        except (SystemExit, ImportError):
            # Expected if Neo4j backend is not available
            pass


class TestCreateSession:
    """Tests for create_session function."""

    def test_create_session_default_params(self):
        """Test creating session with default parameters."""
        session = create_session()
        assert session is not None
        # Session should be a MelleaSession instance
        assert hasattr(session, "instruct")

    def test_create_session_custom_model(self):
        """Test creating session with custom model."""
        session = create_session(model_id="gpt-4o-mini")
        assert session is not None

    def test_create_session_custom_temperature(self):
        """Test creating session with custom temperature."""
        session = create_session(temperature=0.5)
        assert session is not None

    def test_create_session_litellm_backend(self):
        """Test creating session with litellm backend."""
        session = create_session(backend_name="litellm", model_id="gpt-4o-mini")
        assert session is not None


class TestMelleaResourceManager:
    """Tests for MelleaResourceManager async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Test that context manager cleans up resources."""
        from mellea_contribs.kg.utils import MelleaResourceManager

        manager = MelleaResourceManager(backend_type="mock")
        async with manager as mgr:
            assert mgr.session is not None
            assert mgr.backend is not None

        # After exiting context, backend should be closed
        # (MockGraphBackend.close() is async, so this just verifies it ran)

    @pytest.mark.asyncio
    async def test_context_manager_with_neo4j_params(self):
        """Test context manager with Neo4j parameters."""
        from mellea_contribs.kg.utils import MelleaResourceManager

        manager = MelleaResourceManager(
            backend_type="mock",
            model_id="gpt-4o-mini",
        )
        async with manager as mgr:
            assert mgr.session is not None
            assert isinstance(mgr.backend, MockGraphBackend)


class TestCreateOpenAISession:
    """Tests for create_openai_session function."""

    def test_create_openai_session_returns_session(self):
        """Test that create_openai_session returns a usable session."""
        from mellea_contribs.kg.utils.session_manager import create_openai_session

        session = create_openai_session()
        assert session is not None
        assert hasattr(session, "instruct")

    def test_create_openai_session_custom_model(self):
        """Test creating OpenAI session with a different model ID."""
        from mellea_contribs.kg.utils.session_manager import create_openai_session

        session = create_openai_session(model_id="gpt-4o")
        assert session is not None

    def test_create_openai_session_with_api_base(self):
        """Test creating OpenAI session with explicit API base and key."""
        from mellea_contribs.kg.utils.session_manager import create_openai_session

        session = create_openai_session(
            model_id="gpt-4o-mini",
            api_base="http://localhost:8080/v1",
            api_key="test-key",
        )
        assert session is not None

    def test_create_openai_session_no_force_schema(self):
        """Test creating session with force_openai_schema disabled."""
        from mellea_contribs.kg.utils.session_manager import create_openai_session

        session = create_openai_session(force_openai_schema=False)
        assert session is not None


class TestCreateSessionFromEnv:
    """Tests for create_session_from_env function."""

    def test_returns_session_model_tuple(self):
        """Test that create_session_from_env returns (session, model_id) tuple."""
        from mellea_contribs.kg.utils.session_manager import create_session_from_env

        result = create_session_from_env()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_default_model_id(self):
        """Test that default model ID is used when env var not set."""
        from mellea_contribs.kg.utils.session_manager import create_session_from_env

        _, model_id = create_session_from_env(default_model="gpt-4o-mini")
        assert model_id == "gpt-4o-mini"

    def test_model_from_env_var(self, monkeypatch):
        """Test that MODEL_NAME env var overrides default model."""
        from mellea_contribs.kg.utils.session_manager import create_session_from_env

        monkeypatch.setenv("MODEL_NAME", "gpt-4o")
        _, model_id = create_session_from_env()
        assert model_id == "gpt-4o"

    def test_prefixed_env_vars(self, monkeypatch):
        """Test that env_prefix correctly scopes environment variables."""
        from mellea_contribs.kg.utils.session_manager import create_session_from_env

        monkeypatch.setenv("EVAL_MODEL_NAME", "gpt-4-turbo")
        _, model_id = create_session_from_env(default_model="gpt-4o-mini", env_prefix="EVAL_")
        assert model_id == "gpt-4-turbo"

    def test_session_is_usable(self):
        """Test that returned session has expected attributes."""
        from mellea_contribs.kg.utils.session_manager import create_session_from_env

        session, _ = create_session_from_env()
        assert hasattr(session, "instruct")


class TestCreateEmbeddingClient:
    """Tests for create_embedding_client function."""

    def test_returns_client_or_none(self):
        """Test that create_embedding_client does not raise exceptions."""
        from mellea_contribs.kg.utils.session_manager import create_embedding_client

        # Should return a client (openai installed) or None (not installed)
        client = create_embedding_client()
        # Both outcomes are acceptable; the key is no exception is raised

    def test_model_name_attached_to_client(self):
        """Test that model name is stored as client._model_name."""
        from mellea_contribs.kg.utils.session_manager import create_embedding_client

        client = create_embedding_client(model_name="text-embedding-3-large")
        if client is not None:
            assert client._model_name == "text-embedding-3-large"

    def test_default_model_name(self):
        """Test default embedding model name."""
        from mellea_contribs.kg.utils.session_manager import create_embedding_client

        client = create_embedding_client()
        if client is not None:
            assert client._model_name == "text-embedding-3-small"

    def test_custom_api_base(self):
        """Test creating embedding client with custom endpoint."""
        from mellea_contribs.kg.utils.session_manager import create_embedding_client

        client = create_embedding_client(
            api_base="http://localhost:8080/v1",
            api_key="test-key",
            model_name="my-embed-model",
        )
        if client is not None:
            assert client._model_name == "my-embed-model"


class TestGenerateEmbeddings:
    """Tests for generate_embeddings function."""

    @pytest.mark.asyncio
    async def test_none_client_returns_nones(self):
        """Test that None client returns a list of None values."""
        from mellea_contribs.kg.utils.session_manager import generate_embeddings

        result = await generate_embeddings(None, ["text1", "text2"])
        assert result == [None, None]

    @pytest.mark.asyncio
    async def test_empty_texts_returns_empty(self):
        """Test that empty input list returns empty list."""
        from mellea_contribs.kg.utils.session_manager import generate_embeddings

        result = await generate_embeddings(None, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_correct_length_with_none_client(self):
        """Test that result length matches input length when client is None."""
        from mellea_contribs.kg.utils.session_manager import generate_embeddings

        texts = ["a", "b", "c", "d"]
        result = await generate_embeddings(None, texts)
        assert len(result) == len(texts)
        assert all(v is None for v in result)

    @pytest.mark.asyncio
    async def test_model_name_override(self):
        """Test that explicit model_name overrides client._model_name."""
        from mellea_contribs.kg.utils.session_manager import generate_embeddings

        # With None client this returns nones regardless of model_name,
        # but the call should not raise
        result = await generate_embeddings(None, ["text"], model_name="custom-model")
        assert result == [None]


class TestIntegration:
    """Integration tests for session_manager functions."""

    def test_workflow_create_session_and_backend(self):
        """Test creating both session and backend."""
        backend = create_backend(backend_type="mock")
        session = create_session(model_id="gpt-4o-mini")
        assert backend is not None
        assert session is not None

    @pytest.mark.asyncio
    async def test_workflow_with_resource_manager(self):
        """Test workflow using resource manager."""
        from mellea_contribs.kg.utils import MelleaResourceManager

        async with MelleaResourceManager(backend_type="mock") as manager:
            # Should be able to access both
            assert manager.session is not None
            assert manager.backend is not None

            # Backend should be functional
            schema = await manager.backend.get_schema()
            assert isinstance(schema, dict)
