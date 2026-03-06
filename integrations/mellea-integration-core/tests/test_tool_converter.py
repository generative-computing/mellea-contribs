"""Tests for tool conversion utilities."""

from unittest.mock import Mock

import pytest

from mellea_integration import BaseToolConverter


class TestToolConverter(BaseToolConverter):
    """Test implementation of tool converter."""

    def to_mellea(self, tools):
        """Simple test implementation."""
        return [Mock(name=tool.name) for tool in tools]


@pytest.fixture
def converter():
    """Create a test tool converter."""
    return TestToolConverter()


def test_extract_tool_schema_basic(converter):
    """Test extracting basic tool schema."""
    tool = Mock(spec=["name", "description"])
    tool.name = "test_tool"
    tool.description = "Test description"

    schema = converter.extract_tool_schema(tool)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "test_tool"
    assert schema["function"]["description"] == "Test description"
    assert schema["function"]["parameters"] == {}


def test_extract_tool_schema_with_args_schema(converter):
    """Test extracting schema with args_schema."""
    mock_args_schema = Mock()
    mock_args_schema.schema.return_value = {"type": "object", "properties": {}}

    tool = Mock(name="test_tool", description="Test description", args_schema=mock_args_schema)

    schema = converter.extract_tool_schema(tool)

    assert schema["function"]["parameters"]["type"] == "object"


def test_extract_tool_schema_missing_attributes(converter):
    """Test extracting schema with missing attributes."""
    tool = Mock(spec=[])

    schema = converter.extract_tool_schema(tool)

    assert schema["function"]["name"] == "unknown_tool"
    assert schema["function"]["description"] == ""


def test_get_tool_callable_func_attribute(converter):
    """Test getting callable from func attribute."""
    mock_func = Mock()
    tool = Mock(func=mock_func)

    result = converter.get_tool_callable(tool)

    assert result == mock_func


def test_get_tool_callable_run_attribute(converter):
    """Test getting callable from _run attribute."""
    mock_run = Mock()
    tool = Mock(spec=["_run"])
    tool._run = mock_run

    result = converter.get_tool_callable(tool)

    assert result == mock_run


def test_get_tool_callable_run_method(converter):
    """Test getting callable from run method."""
    mock_run = Mock()
    tool = Mock(spec=["run"])
    tool.run = mock_run

    result = converter.get_tool_callable(tool)

    assert result == mock_run


def test_get_tool_callable_tool_itself(converter):
    """Test getting callable when tool itself is callable."""
    tool = Mock()
    tool.__call__ = Mock()

    result = converter.get_tool_callable(tool)

    assert callable(result)


def test_get_tool_callable_not_found(converter):
    """Test error when no callable found."""
    # Create a non-callable object
    class NonCallable:
        pass

    tool = NonCallable()

    with pytest.raises(ValueError, match="Cannot extract callable"):
        converter.get_tool_callable(tool)


def test_parse_tool_calls_from_string_single(converter):
    """Test parsing single tool call from string."""
    content = "[ToolCall(function=Function(name='get_weather', arguments={'location': 'NYC'}))]"

    tool_calls = converter.parse_tool_calls_from_string(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"
    assert tool_calls[0]["args"]["location"] == "NYC"
    assert "id" in tool_calls[0]


def test_parse_tool_calls_from_string_multiple(converter):
    """Test parsing multiple tool calls from string."""
    content = (
        "[ToolCall(function=Function(name='tool1', arguments={'arg1': 'val1'})), "
        "ToolCall(function=Function(name='tool2', arguments={'arg2': 'val2'}))]"
    )

    tool_calls = converter.parse_tool_calls_from_string(content)

    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "tool1"
    assert tool_calls[1]["name"] == "tool2"


def test_parse_tool_calls_from_string_invalid(converter):
    """Test parsing invalid string returns empty list."""
    content = "Not a tool call"

    tool_calls = converter.parse_tool_calls_from_string(content)

    assert tool_calls == []


def test_parse_tool_calls_from_string_malformed_args(converter):
    """Test parsing with malformed arguments skips that call."""
    content = "[ToolCall(function=Function(name='tool1', arguments={invalid}))]"

    tool_calls = converter.parse_tool_calls_from_string(content)

    assert tool_calls == []


def test_extract_tool_calls_from_response_tool_calls_attr(converter):
    """Test extracting from tool_calls attribute."""
    mock_tc = Mock()
    mock_tc.id = "call_1"
    mock_tc.name = "test_tool"
    mock_tc.arguments = {"arg": "val"}
    response = Mock(tool_calls=[mock_tc])

    tool_calls = converter.extract_tool_calls_from_response(response)

    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["name"] == "test_tool"
    assert tool_calls[0]["args"] == {"arg": "val"}


def test_extract_tool_calls_from_response_private_attr(converter):
    """Test extracting from _tool_calls attribute."""
    mock_tc = Mock()
    mock_tc.id = "call_1"
    mock_tc.name = "test_tool"
    mock_tc.arguments = {"arg": "val"}
    response = Mock(spec=["_tool_calls"])
    response._tool_calls = [mock_tc]

    tool_calls = converter.extract_tool_calls_from_response(response)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "test_tool"


def test_extract_tool_calls_from_response_string_content(converter):
    """Test extracting from string content."""
    response = Mock(
        spec=["content"],
        content="[ToolCall(function=Function(name='tool1', arguments={'arg': 'val'}))]",
    )

    tool_calls = converter.extract_tool_calls_from_response(response)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "tool1"


def test_extract_tool_calls_from_response_empty(converter):
    """Test extracting when no tool calls present."""
    response = Mock(spec=["content"], content="Regular response")

    tool_calls = converter.extract_tool_calls_from_response(response)

    assert tool_calls == []


def test_extract_tool_calls_from_response_empty_list(converter):
    """Test extracting when tool_calls is empty list."""
    response = Mock(tool_calls=[])

    tool_calls = converter.extract_tool_calls_from_response(response)

    assert tool_calls == []


def test_parse_tool_calls_default_implementation(converter):
    """Test default parse_tool_calls implementation."""
    mock_tc = Mock()
    mock_tc.id = "call_1"
    mock_tc.name = "test_tool"
    mock_tc.arguments = {}
    response = Mock(tool_calls=[mock_tc])

    tool_calls = converter.parse_tool_calls(response)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "test_tool"


def test_to_mellea_not_implemented():
    """Test that to_mellea must be implemented."""
    converter = BaseToolConverter()
    with pytest.raises(NotImplementedError):
        converter.to_mellea([])


def test_full_tool_conversion_flow(converter):
    """Test complete tool conversion flow."""
    # Framework tools
    tools = [
        Mock(name="tool1", description="First tool"),
        Mock(name="tool2", description="Second tool"),
    ]

    # Convert to Mellea
    mellea_tools = converter.to_mellea(tools)
    assert len(mellea_tools) == 2
    assert all(hasattr(tool, "name") for tool in mellea_tools)


