"""Shared test fixtures for mellea-crewai integration tests."""

import os
from typing import Any
from unittest.mock import Mock

import pytest

# Suppress CrewAI telemetry before importing
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "1")


class FakeFunction:
    """Fake function object for tool calls."""

    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    """Fake tool call object."""

    def __init__(self, name: str, arguments: dict):
        self.function = FakeFunction(name, arguments)


@pytest.fixture
def mock_mellea_session():
    """Mock Mellea session with standard response setup.

    Returns a Mock session with chat, achat, instruct, and ainstruct methods.
    Default response has empty tool calls (real empty list, not Mock).
    """
    mock_session = Mock()
    mock_backend = Mock()
    mock_backend.get_context_window_size.return_value = 4096
    mock_backend.supports_multimodal.return_value = False
    mock_session.backend = mock_backend

    # Create default response
    mock_response = Mock()
    mock_response.content = "Mock LLM response"
    mock_response._tool_calls = []  # Real empty list, NOT Mock()
    mock_response.tool_calls = []  # Also set tool_calls attribute

    # Mock usage with proper token counts
    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15
    mock_response.usage = mock_usage

    # Set up sync methods
    mock_session.chat.return_value = mock_response
    mock_session.instruct.return_value = mock_response

    # Set up async methods (using plain async def, not AsyncMock)
    async def async_chat(*args, **kwargs):
        return mock_response

    async def async_instruct(*args, **kwargs):
        return mock_response

    mock_session.achat = async_chat
    mock_session.ainstruct = async_instruct

    return mock_session


@pytest.fixture
def mock_tool_call_response(mock_mellea_session):
    """Reconfigure mock_mellea_session to return a tool call response.

    Returns the FakeToolCall class so tests can build custom tool calls.
    """
    mock_response = Mock()
    mock_response.content = "Calling tool"
    fake_tool_call = FakeToolCall("simple_echo_tool", {"text": "hello"})
    mock_response._tool_calls = [fake_tool_call]
    mock_response.tool_calls = [fake_tool_call]

    # Mock usage
    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15
    mock_response.usage = mock_usage

    mock_mellea_session.chat.return_value = mock_response
    mock_mellea_session.instruct.return_value = mock_response

    return FakeToolCall


@pytest.fixture
def simple_crewai_tool():
    """Simple CrewAI tool for testing."""
    from crewai.tools import tool

    @tool("simple_echo_tool")
    def simple_echo(text: str) -> str:
        """Echo the input text.

        Args:
            text: Text to echo

        Returns:
            Echoed text with prefix
        """
        return f"Echo: {text}"

    return simple_echo


@pytest.fixture
def calculator_tool():
    """Calculator tool for testing."""
    from crewai.tools import tool

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

    return add_numbers


@pytest.fixture
def mock_task_with_tools(simple_crewai_tool):
    """Mock Task with tools."""
    mock_task = Mock()
    mock_task.tools = [simple_crewai_tool]
    mock_task.name = "test_task"  # Plain string for Pydantic validation
    return mock_task


@pytest.fixture
def mock_agent_with_tools(simple_crewai_tool):
    """Mock Agent with tools."""
    mock_agent = Mock()
    mock_agent.tools = [simple_crewai_tool]
    mock_agent.role = "Test Helper"  # Plain string for Pydantic validation
    return mock_agent


@pytest.fixture(scope="session")
def mellea_session():
    """Real Mellea session for integration tests.

    Skips if mellea not importable or backend unreachable.
    """
    pytest.importorskip("mellea")

    try:
        from mellea import start_session

        session = start_session()
        return session
    except Exception as e:
        pytest.skip(f"Could not create Mellea session: {e}")
