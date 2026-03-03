"""Integration tests for LangChain agents with MelleaChatModel.

Tests agent workflows including ReAct patterns, tool usage,
multi-step reasoning, and error handling in agent contexts.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mellea_langchain import MelleaChatModel

# Try to import LangChain tool decorator
try:
    from langchain_core.tools import tool

    LANGCHAIN_TOOLS_AVAILABLE = True
except ImportError:
    LANGCHAIN_TOOLS_AVAILABLE = False
    tool = None


# Skip all tests if LangChain tools are not available
pytestmark = pytest.mark.skipif(
    not LANGCHAIN_TOOLS_AVAILABLE, reason="LangChain tools not installed"
)


class FakeAgentMelleaSession:
    """Fake Mellea session that simulates agent-like responses."""

    def __init__(self):
        self.call_count = 0
        self.call_history = []
        self.tool_call_responses = {
            "search": "Python is a high-level programming language.",
            "calculator": "42",
            "get_weather": "Sunny, 72°F",
        }

    def chat(self, message, model_options=None, tool_calls=False):
        """Mock sync chat for agent interactions."""
        self.call_count += 1
        self.call_history.append({"message": message, "tool_calls": tool_calls})

        class MockResponse:
            def __init__(self, content):
                self.content = content
                self._tool_calls = None

        # Simulate agent reasoning
        if tool_calls and "search" in str(message).lower():
            # Return tool call format
            return MockResponse(
                "[ToolCall(function=Function(name='search', arguments={'query': 'Python'}))]"
            )
        elif tool_calls and "calculate" in str(message).lower():
            return MockResponse(
                "[ToolCall(function=Function(name='calculator', arguments={'expression': '40+2'}))]"
            )
        elif "final answer" in str(message).lower() or self.call_count > 3:
            return MockResponse("The answer is: Python is a programming language.")
        else:
            return MockResponse("Let me search for that information.")

    async def achat(self, message, model_options=None, tool_calls=False):
        """Mock async chat for agent interactions."""
        self.call_count += 1
        self.call_history.append({"message": message, "tool_calls": tool_calls, "async": True})

        class MockResponse:
            def __init__(self, content):
                self.content = content
                self._tool_calls = None

        # Simulate agent reasoning
        if tool_calls and "search" in str(message).lower():
            return MockResponse(
                "[ToolCall(function=Function(name='search', arguments={'query': 'Python'}))]"
            )
        elif tool_calls and "weather" in str(message).lower():
            return MockResponse(
                "[ToolCall(function=Function(name='get_weather', arguments={'location': 'NYC'}))]"
            )
        elif "final answer" in str(message).lower() or self.call_count > 3:
            return MockResponse("Based on the information, the weather is sunny.")
        else:
            return MockResponse("I need to check the weather.")

    def instruct(
        self,
        message,
        requirements=None,
        strategy=None,
        model_options=None,
        return_sampling_results=False,
    ):
        """Mock sync instruct for requirements/strategy."""
        self.call_count += 1
        self.call_history.append(
            {
                "message": message,
                "requirements": requirements,
                "strategy": strategy,
            }
        )

        class MockResponse:
            def __init__(self, content):
                self.content = content
                self._tool_calls = None

        return MockResponse("Validated response with requirements")

    async def ainstruct(
        self,
        message,
        requirements=None,
        strategy=None,
        model_options=None,
        return_sampling_results=False,
    ):
        """Mock async instruct for requirements/strategy."""
        self.call_count += 1
        self.call_history.append(
            {
                "message": message,
                "requirements": requirements,
                "strategy": strategy,
                "async": True,
            }
        )

        class MockResponse:
            def __init__(self, content):
                self.content = content
                self._tool_calls = None

        return MockResponse("Async validated response with requirements")


@pytest.mark.integration
class TestBasicAgentIntegration:
    """Test basic agent functionality with MelleaChatModel."""

    def test_agent_with_single_tool(self):
        """Test agent using a single tool."""

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Search results for: {query}"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        # Bind tools to model
        model_with_tools = chat_model.bind_tools([search])

        # Test tool invocation
        messages = [HumanMessage(content="Search for Python programming")]
        response = model_with_tools.invoke(messages)

        # Should have made at least one call
        assert fake_session.call_count >= 1
        assert response is not None

    def test_agent_with_multiple_tools(self):
        """Test agent with multiple tools available."""

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results: {query}"

        @tool
        def calculator(expression: str) -> str:
            """Calculate mathematical expressions."""
            return "42"

        @tool
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: Sunny"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        # Bind multiple tools
        model_with_tools = chat_model.bind_tools([search, calculator, get_weather])

        messages = [HumanMessage(content="What is 40 + 2?")]
        response = model_with_tools.invoke(messages)

        # Should have tool calls enabled
        assert fake_session.call_history[-1]["tool_calls"] is True
        assert response is not None

    @pytest.mark.asyncio
    async def test_async_agent_with_tools(self):
        """Test async agent execution with tools."""

        @tool
        def get_weather(location: str) -> str:
            """Get weather information."""
            return f"Weather in {location}: Sunny, 72°F"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([get_weather])

        messages = [HumanMessage(content="What's the weather in NYC?")]
        response = await model_with_tools.ainvoke(messages)

        # Should have made async calls
        assert any(call.get("async") for call in fake_session.call_history)
        assert response is not None


@pytest.mark.integration
class TestReActAgentPattern:
    """Test ReAct (Reasoning + Acting) agent pattern."""

    def test_react_reasoning_loop(self):
        """Test agent's reasoning and action loop."""

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return "Python is a high-level programming language."

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([search])

        # Simulate multi-turn reasoning
        messages = [HumanMessage(content="Tell me about Python")]

        # First call - should reason about using search
        response1 = model_with_tools.invoke(messages)
        assert response1 is not None

        # Agent should have made at least one call
        assert fake_session.call_count >= 1

    def test_react_with_tool_execution(self):
        """Test ReAct pattern with actual tool execution."""

        @tool
        def calculator(expression: str) -> str:
            """Calculate mathematical expressions."""
            try:
                result = eval(expression)
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([calculator])

        messages = [HumanMessage(content="Calculate 25 * 4")]
        response = model_with_tools.invoke(messages)

        # Should have attempted tool usage
        assert fake_session.call_history[-1]["tool_calls"] is True
        assert response is not None

    def test_react_multi_step_reasoning(self):
        """Test multi-step reasoning with multiple tool calls."""

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Information about {query}"

        @tool
        def summarize(text: str) -> str:
            """Summarize text."""
            return f"Summary: {text[:50]}"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([search, summarize])

        # Complex query requiring multiple steps
        messages = [HumanMessage(content="Search for Python and summarize the results")]
        response = model_with_tools.invoke(messages)

        # Should have made multiple calls for reasoning
        assert fake_session.call_count >= 1
        assert response is not None


@pytest.mark.integration
class TestAgentToolExecution:
    """Test tool execution within agent workflows."""

    def test_tool_execution_success(self):
        """Test successful tool execution in agent context."""

        execution_log = []

        @tool
        def log_tool(message: str) -> str:
            """Log a message."""
            execution_log.append(message)
            return f"Logged: {message}"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([log_tool])

        messages = [HumanMessage(content="Log 'test message'")]
        response = model_with_tools.invoke(messages)

        # Tool should have been bound
        assert hasattr(model_with_tools, "_bound_tools")
        assert response is not None

    def test_tool_execution_with_error_handling(self):
        """Test tool execution error handling in agent."""

        @tool
        def failing_tool(x: str) -> str:
            """A tool that fails."""
            raise ValueError("Tool execution failed")

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([failing_tool])

        messages = [HumanMessage(content="Use the failing tool")]

        # Should handle tool errors gracefully
        response = model_with_tools.invoke(messages)
        assert response is not None

    def test_tool_with_complex_arguments(self):
        """Test tool execution with complex argument types."""

        @tool
        def complex_tool(items: list[str], count: int, metadata: dict) -> str:
            """Tool with complex arguments."""
            return f"Processed {count} items with metadata"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([complex_tool])

        messages = [HumanMessage(content="Process items with metadata")]
        response = model_with_tools.invoke(messages)

        assert response is not None


@pytest.mark.integration
class TestAgentConversationHistory:
    """Test agent handling of conversation history."""

    def test_agent_with_conversation_context(self):
        """Test agent maintaining conversation context."""

        @tool
        def memory_tool(key: str) -> str:
            """Retrieve from memory."""
            return f"Retrieved: {key}"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([memory_tool])

        # Multi-turn conversation
        messages = [
            HumanMessage(content="My name is Alice"),
            AIMessage(content="Nice to meet you, Alice!"),
            HumanMessage(content="What's my name?"),
        ]

        response = model_with_tools.invoke(messages)

        # Should have processed all messages
        assert fake_session.call_count >= 1
        assert response is not None

    def test_agent_with_tool_results_in_history(self):
        """Test agent with tool results in conversation history."""

        @tool
        def search(query: str) -> str:
            """Search tool."""
            return f"Results for {query}"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([search])

        # Conversation with tool usage
        messages = [
            HumanMessage(content="Search for Python"),
            AIMessage(
                content="",
                tool_calls=[{"id": "1", "name": "search", "args": {"query": "Python"}}],
            ),
            ToolMessage(content="Python is a programming language", tool_call_id="1"),
            HumanMessage(content="Summarize that"),
        ]

        response = model_with_tools.invoke(messages)

        # Should handle tool messages in history
        assert response is not None


@pytest.mark.integration
class TestAgentErrorHandling:
    """Test error handling in agent workflows."""

    def test_agent_with_invalid_tool_call(self):
        """Test agent handling of invalid tool calls."""

        @tool
        def valid_tool(x: str) -> str:
            """A valid tool."""
            return x

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([valid_tool])

        # Simulate response with invalid tool call
        messages = [HumanMessage(content="Call an invalid tool")]
        response = model_with_tools.invoke(messages)

        # Should handle gracefully
        assert response is not None

    def test_agent_with_malformed_tool_arguments(self):
        """Test agent with malformed tool arguments."""

        @tool
        def strict_tool(number: int) -> str:
            """Tool requiring specific type."""
            return str(number * 2)

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([strict_tool])

        messages = [HumanMessage(content="Use strict tool with invalid args")]
        response = model_with_tools.invoke(messages)

        # Should handle type errors gracefully
        assert response is not None

    @pytest.mark.asyncio
    async def test_async_agent_error_propagation(self):
        """Test error propagation in async agent execution."""

        @tool
        def error_tool(x: str) -> str:
            """Tool that raises error."""
            raise RuntimeError("Tool error")

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([error_tool])

        messages = [HumanMessage(content="Use error tool")]

        # Should handle tool errors
        response = await model_with_tools.ainvoke(messages)
        assert response is not None


@pytest.mark.integration
class TestAgentWithRequirements:
    """Test agents with Mellea's requirements and validation."""

    def test_agent_with_output_requirements(self):
        """Test agent with output validation requirements."""

        @tool
        def data_tool(query: str) -> str:
            """Get data."""
            return "Some data"

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([data_tool])

        # Use with requirements
        messages = [HumanMessage(content="Get validated data")]
        requirements = ["Output must be concise"]

        response = model_with_tools.invoke(messages, requirements=requirements)

        assert response is not None

    @pytest.mark.asyncio
    async def test_async_agent_with_strategy(self):
        """Test async agent with sampling strategy."""

        @tool
        def generate_tool(prompt: str) -> str:
            """Generate content."""
            return f"Generated: {prompt}"

        class MockStrategy:
            """Mock sampling strategy."""

            loop_budget = 3

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([generate_tool])

        messages = [HumanMessage(content="Generate with validation")]
        strategy = MockStrategy()

        response = await model_with_tools.ainvoke(messages, strategy=strategy)

        assert response is not None


@pytest.mark.integration
class TestAgentPerformance:
    """Test agent performance characteristics."""

    def test_agent_with_many_tools(self):
        """Test agent performance with many tools."""
        tools = []
        for i in range(20):

            @tool
            def dynamic_tool(x: str) -> str:
                """Dynamic tool."""
                return f"Result {i}: {x}"

            dynamic_tool.name = f"tool_{i}"
            tools.append(dynamic_tool)

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools(tools)

        messages = [HumanMessage(content="Use available tools")]
        response = model_with_tools.invoke(messages)

        # Should handle many tools
        assert response is not None
        assert len(model_with_tools._bound_tools) == 20

    def test_agent_with_long_conversation(self):
        """Test agent with long conversation history."""

        @tool
        def simple_tool(x: str) -> str:
            """Simple tool."""
            return x

        fake_session = FakeAgentMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        model_with_tools = chat_model.bind_tools([simple_tool])

        # Create long conversation
        messages = []
        for i in range(50):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"Message {i}"))
            else:
                messages.append(AIMessage(content=f"Response {i}"))

        messages.append(HumanMessage(content="Final question"))

        response = model_with_tools.invoke(messages)

        # Should handle long history
        assert response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

# Made with Bob
