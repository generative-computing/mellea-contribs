"""Tests for MelleaLLM class."""

from unittest.mock import Mock

import pytest

# Mark all tests in this file
pytestmark = [pytest.mark.ollama, pytest.mark.llm]


class TestMelleaLLMBasic:
    """Basic tests for MelleaLLM functionality."""

    def test_import(self):
        """Test that MelleaLLM can be imported."""
        from mellea_crewai import MelleaLLM

        assert MelleaLLM is not None

    def test_initialization(self):
        """Test MelleaLLM initialization."""
        from mellea_crewai import MelleaLLM

        # Create mock session
        mock_session = Mock()

        # Initialize MelleaLLM
        llm = MelleaLLM(
            mellea_session=mock_session,
            model="test-model",
            temperature=0.7,
        )

        assert llm.mellea_session == mock_session
        assert llm.model == "test-model"
        assert llm.temperature == 0.7

    def test_initialization_with_requirements(self):
        """Test MelleaLLM initialization with requirements."""
        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        requirements = ["req1", "req2"]

        llm = MelleaLLM(
            mellea_session=mock_session,
            requirements=requirements,
        )

        assert llm._requirements == requirements

    def test_initialization_with_strategy(self):
        """Test MelleaLLM initialization with sampling strategy."""
        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        mock_strategy = Mock()

        llm = MelleaLLM(
            mellea_session=mock_session,
            strategy=mock_strategy,
        )

        assert llm._strategy == mock_strategy


class TestMelleaLLMCall:
    """Tests for MelleaLLM call method."""

    def test_call_with_string_message(self):
        """Test call with string message."""
        from mellea_crewai import MelleaLLM

        # Create mock session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response._tool_calls = []  # Empty list, not Mock
        # Mock usage with proper token counts
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_session)

        # Call with string message
        result = llm.call("Hello, world!")

        # Verify session.chat was called
        mock_session.chat.assert_called_once()
        assert result == "Test response"

    def test_call_with_list_messages(self):
        """Test call with list of messages."""
        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response._tool_calls = []  # Empty list, not Mock
        # Mock usage with proper token counts
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_session)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        result = llm.call(messages)

        mock_session.chat.assert_called_once()
        assert result == "Test response"

    def test_call_with_requirements_uses_instruct(self):
        """Test that call uses instruct when requirements are provided."""
        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response._tool_calls = []  # Empty list, not Mock
        # Mock usage with proper token counts
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_session.instruct.return_value = mock_response

        requirements = ["req1", "req2"]
        llm = MelleaLLM(
            mellea_session=mock_session,
            requirements=requirements,
        )

        result = llm.call("Hello")

        # Verify instruct was called instead of chat
        mock_session.instruct.assert_called_once()
        mock_session.chat.assert_not_called()
        assert result == "Test response"


class TestMelleaLLMAsync:
    """Tests for MelleaLLM async methods."""

    @pytest.mark.asyncio
    async def test_acall_with_string_message(self):
        """Test async call with string message."""
        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "Async response"
        mock_response._tool_calls = []  # Empty list, not Mock
        # Mock usage with proper token counts
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        # Create async mock
        async def async_chat(*args, **kwargs):
            return mock_response

        mock_session.achat = async_chat

        llm = MelleaLLM(mellea_session=mock_session)

        result = await llm.acall("Hello, async world!")

        assert result == "Async response"


class TestMelleaLLMFeatures:
    """Tests for MelleaLLM feature methods."""

    def test_supports_stop_words(self):
        """Test supports_stop_words method."""
        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        llm = MelleaLLM(mellea_session=mock_session, stop=["STOP"])

        assert llm.supports_stop_words() is True

    def test_get_context_window_size_default(self):
        """Test get_context_window_size with default."""
        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        # Mock backend to return a proper integer
        mock_backend = Mock()
        mock_backend.get_context_window_size.return_value = 4096
        mock_session.backend = mock_backend

        llm = MelleaLLM(mellea_session=mock_session)

        size = llm.get_context_window_size()
        assert isinstance(size, int)
        assert size > 0
        assert size == 4096

    def test_supports_multimodal_default(self):
        """Test supports_multimodal with default."""
        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        # Mock backend to return False
        mock_backend = Mock()
        mock_backend.supports_multimodal.return_value = False
        mock_session.backend = mock_backend

        llm = MelleaLLM(mellea_session=mock_session)

        # Should return False from backend
        assert llm.supports_multimodal() is False


@pytest.mark.integration
class TestMelleaLLMIntegration:
    """Integration tests with real Mellea session.

    These tests require Ollama to be running.
    """

    def test_real_call_with_ollama(self):
        """Test real call with Ollama backend."""
        pytest.importorskip("mellea")

        from mellea import start_session

        from mellea_crewai import MelleaLLM

        # Create real session
        m = start_session()
        llm = MelleaLLM(mellea_session=m)

        # Make real call
        result = llm.call("Say 'Hello, CrewAI!' and nothing else.")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "hello" in result.lower() or "crewai" in result.lower()

    @pytest.mark.asyncio
    async def test_real_acall_with_ollama(self):
        """Test real async call with Ollama backend."""
        pytest.importorskip("mellea")

        from mellea import start_session

        from mellea_crewai import MelleaLLM

        m = start_session()
        llm = MelleaLLM(mellea_session=m)

        result = await llm.acall("Say 'Async works!' and nothing else.")

        assert isinstance(result, str)
        assert len(result) > 0
