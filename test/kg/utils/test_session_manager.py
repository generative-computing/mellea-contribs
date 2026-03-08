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
