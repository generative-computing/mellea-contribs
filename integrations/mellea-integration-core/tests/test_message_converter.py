"""Tests for message conversion utilities."""

from unittest.mock import Mock

import pytest

from mellea_integration import BaseMessageConverter


class TestMessageConverter(BaseMessageConverter):
    """Test implementation of message converter."""

    def to_mellea(self, messages):
        """Simple test implementation."""
        return [self.create_mellea_message("user", msg["content"]) for msg in messages]

    def from_mellea(self, response):
        """Simple test implementation."""
        return {"content": self.extract_content_from_response(response)}


@pytest.fixture
def converter():
    """Create a test message converter."""
    return TestMessageConverter()


def test_extract_last_user_message(converter):
    """Test extracting last user message."""
    messages = [
        Mock(role="system", content="System message"),
        Mock(role="user", content="First user message"),
        Mock(role="assistant", content="Assistant response"),
        Mock(role="user", content="Second user message"),
    ]

    result = converter.extract_last_user_message(messages)

    assert result == "Second user message"


def test_extract_last_user_message_no_user(converter):
    """Test error when no user message exists."""
    messages = [
        Mock(role="system", content="System message"),
        Mock(role="assistant", content="Assistant response"),
    ]

    with pytest.raises(ValueError, match="No user message found"):
        converter.extract_last_user_message(messages)


def test_create_mellea_message_valid_roles(converter):
    """Test creating messages with valid roles."""
    for role in ["system", "user", "assistant", "tool"]:
        msg = converter.create_mellea_message(role, "Test content")
        assert msg.role == role
        assert msg.content == "Test content"


def test_create_mellea_message_invalid_role(converter):
    """Test creating message with invalid role defaults to user."""
    msg = converter.create_mellea_message("invalid_role", "Test content")
    assert msg.role == "user"
    assert msg.content == "Test content"


def test_normalize_content_string(converter):
    """Test normalizing string content."""
    result = converter.normalize_content("Hello world")
    assert result == "Hello world"


def test_normalize_content_none(converter):
    """Test normalizing None content."""
    result = converter.normalize_content(None)
    assert result == ""


def test_normalize_content_list(converter):
    """Test normalizing list content."""
    result = converter.normalize_content(["item1", "item2"])
    assert "item1" in result
    assert "item2" in result


def test_normalize_content_dict(converter):
    """Test normalizing dict content."""
    result = converter.normalize_content({"key": "value"})
    assert "key" in result
    assert "value" in result


def test_normalize_content_other(converter):
    """Test normalizing other types."""
    result = converter.normalize_content(42)
    assert result == "42"


def test_extract_content_from_response_with_content(converter):
    """Test extracting content from response with content attribute."""
    response = Mock(content="Response content")
    result = converter.extract_content_from_response(response)
    assert result == "Response content"


def test_extract_content_from_response_with_value(converter):
    """Test extracting content from response with value attribute."""
    response = Mock(spec=["value"])
    response.value = "Response value"
    result = converter.extract_content_from_response(response)
    assert result == "Response value"


def test_extract_content_from_response_fallback(converter):
    """Test extracting content returns empty string when no content/value."""
    response = Mock(spec=[])
    # Remove any default attributes to simulate object with no content/value
    result = converter.extract_content_from_response(response)
    # The implementation returns empty string as fallback
    assert result == ""


def test_extract_content_from_response_none_content(converter):
    """Test extracting None content returns empty string."""
    response = Mock(content=None)
    result = converter.extract_content_from_response(response)
    assert result == ""


def test_to_mellea_not_implemented():
    """Test that to_mellea must be implemented."""
    converter = BaseMessageConverter()
    with pytest.raises(NotImplementedError):
        converter.to_mellea([])


def test_from_mellea_not_implemented():
    """Test that from_mellea must be implemented."""
    converter = BaseMessageConverter()
    with pytest.raises(NotImplementedError):
        converter.from_mellea(Mock())


def test_full_conversion_flow(converter):
    """Test complete message conversion flow."""
    # Framework messages
    messages = [
        {"content": "Hello"},
        {"content": "World"},
    ]

    # Convert to Mellea
    mellea_messages = converter.to_mellea(messages)
    assert len(mellea_messages) == 2
    assert all(hasattr(msg, "role") for msg in mellea_messages)
    assert all(hasattr(msg, "content") for msg in mellea_messages)

    # Convert from Mellea
    response = Mock(content="Response text")
    result = converter.from_mellea(response)
    assert result["content"] == "Response text"


