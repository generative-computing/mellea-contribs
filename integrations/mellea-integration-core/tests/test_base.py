"""Tests for MelleaIntegrationBase."""

from unittest.mock import AsyncMock, Mock

import pytest

from mellea_integration import BaseMessageConverter, MelleaIntegrationBase


class MockMessageConverter(BaseMessageConverter):
    """Mock message converter for testing."""

    def to_mellea(self, messages):
        """Convert mock messages to Mellea format."""
        return [Mock(role="user", content="Test message")]

    def from_mellea(self, response):
        """Convert Mellea response to mock format."""
        return {"content": str(response.content)}


class MockIntegration(MelleaIntegrationBase):
    """Mock integration for testing."""

    def generate(self, messages, **kwargs):
        """Mock generate method."""
        prompt, model_options, tool_calls_enabled = self._prepare_generation(
            messages, kwargs.get("tools"), **kwargs
        )
        response = self._generate_with_mellea(prompt, model_options, tool_calls_enabled)
        return self.message_converter.from_mellea(response)

    async def agenerate(self, messages, **kwargs):
        """Mock async generate method."""
        prompt, model_options, tool_calls_enabled = self._prepare_generation(
            messages, kwargs.get("tools"), **kwargs
        )
        response = await self._agenerate_with_mellea(prompt, model_options, tool_calls_enabled)
        return self.message_converter.from_mellea(response)


@pytest.fixture
def mock_session():
    """Create a mock Mellea session."""
    session = Mock()
    session.chat = Mock(return_value=Mock(content="Test response"))
    session.achat = AsyncMock(return_value=Mock(content="Test async response"))
    session.instruct = Mock(return_value=Mock(content="Test instruct response"))
    session.ainstruct = AsyncMock(return_value=Mock(content="Test async instruct response"))
    return session


@pytest.fixture
def integration(mock_session):
    """Create a mock integration instance."""
    return MockIntegration(mellea_session=mock_session, message_converter=MockMessageConverter())


def test_prepare_generation_basic(integration):
    """Test basic message preparation."""
    messages = [{"role": "user", "content": "Hello"}]

    prompt, model_options, tool_calls_enabled = integration._prepare_generation(messages)

    assert prompt == "Test message"
    assert isinstance(model_options, dict)
    assert tool_calls_enabled is False


def test_prepare_generation_no_messages(integration):
    """Test preparation with no messages raises error."""
    with pytest.raises(ValueError, match="No messages provided"):
        integration._prepare_generation([])


def test_generate_with_mellea_chat(integration, mock_session):
    """Test generation using chat method."""
    response = integration._generate_with_mellea("Test prompt", {}, False)

    assert response.content == "Test response"
    mock_session.chat.assert_called_once()
    mock_session.instruct.assert_not_called()


def test_generate_with_mellea_instruct(integration, mock_session):
    """Test generation using instruct method with requirements."""
    response = integration._generate_with_mellea(
        "Test prompt", {}, False, requirements=["req1", "req2"]
    )

    assert response.content == "Test instruct response"
    mock_session.instruct.assert_called_once()
    mock_session.chat.assert_not_called()


@pytest.mark.asyncio
async def test_agenerate_with_mellea_chat(integration, mock_session):
    """Test async generation using achat method."""
    response = await integration._agenerate_with_mellea("Test prompt", {}, False)

    assert response.content == "Test async response"
    mock_session.achat.assert_called_once()
    mock_session.ainstruct.assert_not_called()


@pytest.mark.asyncio
async def test_agenerate_with_mellea_instruct(integration, mock_session):
    """Test async generation using ainstruct method with strategy."""
    response = await integration._agenerate_with_mellea("Test prompt", {}, False, strategy=Mock())

    assert response.content == "Test async instruct response"
    mock_session.ainstruct.assert_called_once()
    mock_session.achat.assert_not_called()


def test_handle_sampling_results_success(integration):
    """Test handling successful sampling results."""
    mock_result = Mock(content="Success")
    mock_response = Mock(success=True, result=mock_result)

    result = integration._handle_sampling_results(mock_response)

    assert result == mock_result


def test_handle_sampling_results_failure_with_samples(integration):
    """Test handling failed sampling with available samples."""
    mock_sample = Mock(content="Sample")
    mock_response = Mock(success=False, result=None, sample_generations=[mock_sample])

    result = integration._handle_sampling_results(mock_response)

    assert result == mock_sample


def test_handle_sampling_results_failure_no_samples(integration):
    """Test handling failed sampling without samples raises error."""
    mock_response = Mock(success=False, result=None, sample_generations=[])

    with pytest.raises(ValueError, match="No samples generated"):
        integration._handle_sampling_results(mock_response)


def test_handle_sampling_results_not_sampling_result(integration):
    """Test handling non-sampling result passes through."""
    mock_response = Mock(content="Regular response")
    # Remove success attribute to simulate non-SamplingResult
    del mock_response.success

    result = integration._handle_sampling_results(mock_response)

    assert result == mock_response


def test_integration_with_requirements(mock_session):
    """Test integration initialized with requirements."""
    integration = MockIntegration(
        mellea_session=mock_session,
        message_converter=MockMessageConverter(),
        requirements=["req1", "req2"],
    )

    assert integration._requirements == ["req1", "req2"]


def test_integration_with_strategy(mock_session):
    """Test integration initialized with strategy."""
    mock_strategy = Mock()
    integration = MockIntegration(
        mellea_session=mock_session,
        message_converter=MockMessageConverter(),
        strategy=mock_strategy,
    )

    assert integration._strategy == mock_strategy


def test_full_generate_flow(integration, mock_session):
    """Test complete generation flow."""
    messages = [{"role": "user", "content": "Hello"}]

    result = integration.generate(messages)

    assert result["content"] == "Test response"
    mock_session.chat.assert_called_once()


@pytest.mark.asyncio
async def test_full_agenerate_flow(integration, mock_session):
    """Test complete async generation flow."""
    messages = [{"role": "user", "content": "Hello"}]

    result = await integration.agenerate(messages)

    assert result["content"] == "Test async response"
    mock_session.achat.assert_called_once()


