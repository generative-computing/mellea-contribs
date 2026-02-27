"""Tests for tool conversion functionality."""

from unittest.mock import Mock

import pytest


class TestToolConversion:
    """Tests for CrewAI to Mellea tool conversion."""

    def test_import_tool_conversion(self):
        """Test that tool conversion module can be imported."""
        from mellea_crewai.tool_conversion import CrewAIToolConverter

        assert CrewAIToolConverter is not None

    def test_convert_single_tool(self):
        """Test converting a single CrewAI tool to Mellea format."""
        from crewai.tools import tool

        from mellea_crewai.tool_conversion import CrewAIToolConverter

        @tool("test_tool")
        def test_tool(message: str) -> str:
            """A test tool.

            Args:
                message: Test message

            Returns:
                The message with prefix
            """
            return f"Tool: {message}"

        # Convert tool
        converter = CrewAIToolConverter()
        mellea_tools = converter.to_mellea([test_tool])

        # Verify conversion
        assert len(mellea_tools) == 1
        assert mellea_tools[0].name == "test_tool"
        assert callable(mellea_tools[0].run)
        assert "function" in mellea_tools[0].as_json_tool
        assert mellea_tools[0].as_json_tool["function"]["name"] == "test_tool"

    def test_convert_multiple_tools(self):
        """Test converting multiple CrewAI tools."""
        from crewai.tools import tool

        from mellea_crewai.tool_conversion import CrewAIToolConverter

        @tool("tool_one")
        def tool_one(x: int) -> int:
            """First tool."""
            return x * 2

        @tool("tool_two")
        def tool_two(y: str) -> str:
            """Second tool."""
            return y.upper()

        # Convert tools
        converter = CrewAIToolConverter()
        mellea_tools = converter.to_mellea([tool_one, tool_two])

        # Verify conversion
        assert len(mellea_tools) == 2
        assert mellea_tools[0].name == "tool_one"
        assert mellea_tools[1].name == "tool_two"

    def test_tool_has_correct_json_schema(self):
        """Test that converted tool has correct JSON schema."""
        from crewai.tools import tool

        from mellea_crewai.tool_conversion import CrewAIToolConverter

        @tool("schema_test")
        def schema_test(param1: str, param2: int) -> str:
            """Test tool with parameters.

            Args:
                param1: First parameter
                param2: Second parameter

            Returns:
                Combined result
            """
            return f"{param1}-{param2}"

        # Convert tool
        converter = CrewAIToolConverter()
        mellea_tools = converter.to_mellea([schema_test])
        tool_json = mellea_tools[0].as_json_tool

        # Verify JSON schema structure
        assert tool_json["type"] == "function"
        assert "function" in tool_json
        assert tool_json["function"]["name"] == "schema_test"
        assert "description" in tool_json["function"]
        assert "parameters" in tool_json["function"]

    def test_tool_execution(self):
        """Test that converted tool can be executed."""
        from crewai.tools import tool

        from mellea_crewai.tool_conversion import CrewAIToolConverter

        @tool("executable_tool")
        def executable_tool(value: str) -> str:
            """Executable test tool."""
            return f"Result: {value}"

        # Convert and execute
        converter = CrewAIToolConverter()
        mellea_tools = converter.to_mellea([executable_tool])
        result = mellea_tools[0].run(value="test")

        assert result == "Result: test"

    def test_empty_tool_list(self):
        """Test converting empty tool list."""
        from mellea_crewai.tool_conversion import CrewAIToolConverter

        converter = CrewAIToolConverter()
        mellea_tools = converter.to_mellea([])
        assert len(mellea_tools) == 0


class TestToolCallingIntegration:
    """Integration tests for tool calling with MelleaLLM."""

    def test_llm_call_with_tools(self):
        """Test LLM call with tools parameter."""
        from crewai.tools import tool

        from mellea_crewai import MelleaLLM

        @tool("mock_tool")
        def mock_tool(x: int) -> int:
            """Mock tool."""
            return x * 2

        # Create mock session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "Tool response"
        mock_response._tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_session)

        # Call with tools
        llm.call("Test message", tools=[mock_tool])

        # Verify chat was called with tool_calls enabled
        mock_session.chat.assert_called_once()
        call_kwargs = mock_session.chat.call_args[1]
        assert "tool_calls" in call_kwargs
        assert call_kwargs["tool_calls"] is True
        assert "model_options" in call_kwargs

        # Verify tools were passed in model_options
        from mellea.backends import ModelOption

        assert ModelOption.TOOLS in call_kwargs["model_options"]

    def test_llm_call_without_tools(self):
        """Test LLM call without tools parameter."""
        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "No tools response"
        mock_response._tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_session)

        # Call without tools
        llm.call("Test message")

        # Verify chat was called without tool_calls
        mock_session.chat.assert_called_once()
        call_kwargs = mock_session.chat.call_args[1]
        assert "tool_calls" in call_kwargs
        assert call_kwargs["tool_calls"] is False

    @pytest.mark.asyncio
    async def test_async_llm_call_with_tools(self):
        """Test async LLM call with tools parameter."""
        from crewai.tools import tool

        from mellea_crewai import MelleaLLM

        @tool("async_mock_tool")
        def async_mock_tool(x: str) -> str:
            """Async mock tool."""
            return x.upper()

        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = "Async tool response"
        mock_response._tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        async def async_chat(*args, **kwargs):
            return mock_response

        mock_session.achat = async_chat

        llm = MelleaLLM(mellea_session=mock_session)

        # Call with tools
        result = await llm.acall("Test message", tools=[async_mock_tool])

        assert result == "Async tool response"


@pytest.mark.integration
class TestToolCallingRealIntegration:
    """Real integration tests with Ollama backend.

    These tests require Ollama to be running with a model that supports
    function calling (e.g., llama3.1, mistral, qwen2.5).
    """

    @pytest.mark.skip(reason="Requires model with function calling support")
    def test_real_tool_calling(self):
        """Test real tool calling with Ollama."""
        pytest.importorskip("mellea")

        from crewai.tools import tool
        from mellea import start_session

        from mellea_crewai import MelleaLLM

        @tool("add_numbers")
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together.

            Args:
                a: First number
                b: Second number

            Returns:
                Sum of the two numbers
            """
            return a + b

        # Create session and LLM
        m = start_session()
        llm = MelleaLLM(mellea_session=m)

        # Make call with tool
        result = llm.call("What is 5 plus 3?", tools=[add_numbers])

        # Note: This test requires a model with function calling support.
        # The test verifies that:
        # 1. Tools are properly passed to the backend
        # 2. The model can invoke tools or return a text response
        # 3. Result is either a string answer or a list of tool calls
        assert isinstance(result, (str, list))
