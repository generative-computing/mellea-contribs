"""Tests for DSPyMessageConverter."""

from unittest.mock import Mock

import pytest
from mellea_dspy.message_conversion import DSPyMessageConverter


@pytest.fixture
def converter():
    """Create a DSPyMessageConverter fixture."""
    return DSPyMessageConverter()


@pytest.fixture
def mock_mellea_response():
    """Create a mock Mellea response."""
    return Mock(content="test content")


class TestToMellea:
    """Tests for DSPyMessageConverter.to_mellea."""

    def test_to_mellea_user_message(self, converter):
        """Test converting a simple user message."""
        messages = [{"role": "user", "content": "hello"}]
        result = converter.to_mellea(messages)

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "hello"

    def test_to_mellea_multi_turn(self, converter):
        """Test converting multi-turn conversation."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = converter.to_mellea(messages)

        assert len(result) == 4
        assert result[0].role == "system"
        assert result[1].role == "user"
        assert result[2].role == "assistant"
        assert result[3].role == "user"

    def test_to_mellea_missing_content_key(self, converter):
        """Test message without content key normalizes gracefully."""
        messages = [{"role": "user"}]
        result = converter.to_mellea(messages)

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == ""

    def test_to_mellea_default_role(self, converter):
        """Test message without role defaults to user."""
        messages = [{"content": "hello"}]
        result = converter.to_mellea(messages)

        assert len(result) == 1
        assert result[0].role == "user"


class TestFromMellea:
    """Tests for DSPyMessageConverter.from_mellea."""

    def test_from_mellea_produces_openai_compatible_response(
        self, converter, mock_mellea_response
    ):
        """Test response has OpenAI-compatible structure."""
        response = converter.from_mellea(mock_mellea_response)

        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].index == 0

    def test_from_mellea_usage_is_dict_convertible(
        self, converter, mock_mellea_response
    ):
        """Test usage object can be converted to dict."""
        response = converter.from_mellea(mock_mellea_response)

        usage_dict = dict(response.usage)
        assert "prompt_tokens" in usage_dict
        assert "completion_tokens" in usage_dict
        assert "total_tokens" in usage_dict

    def test_from_mellea_id_starts_with_mellea(self, converter, mock_mellea_response):
        """Test response ID has mellea prefix."""
        response = converter.from_mellea(mock_mellea_response)

        assert response.id.startswith("mellea-")

    def test_from_mellea_object_type(self, converter, mock_mellea_response):
        """Test response object type is correct."""
        response = converter.from_mellea(mock_mellea_response)

        assert response.object == "chat.completion"

    def test_from_mellea_model_default(self, converter, mock_mellea_response):
        """Test default model is 'mellea'."""
        response = converter.from_mellea(mock_mellea_response)

        assert response.model == "mellea"

    def test_from_mellea_content_preserved(self, converter, mock_mellea_response):
        """Test content from Mellea response is preserved."""
        response = converter.from_mellea(mock_mellea_response)

        assert response.choices[0].message.content == "test content"

    def test_from_mellea_different_content_different_id(self, converter):
        """Test different content produces different IDs."""
        response1_mock = Mock(content="content1")
        response2_mock = Mock(content="content2")

        response1 = converter.from_mellea(response1_mock)
        response2 = converter.from_mellea(response2_mock)

        assert response1.id != response2.id

    def test_from_mellea_usage_values_numeric(self, converter, mock_mellea_response):
        """Test usage values are numeric."""
        response = converter.from_mellea(mock_mellea_response)
        usage_dict = dict(response.usage)

        assert isinstance(usage_dict["prompt_tokens"], int)
        assert isinstance(usage_dict["completion_tokens"], int)
        assert isinstance(usage_dict["total_tokens"], int)
