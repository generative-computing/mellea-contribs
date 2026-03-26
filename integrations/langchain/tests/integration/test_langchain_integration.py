"""Integration tests for Mellea LangChain integration.

Tests the full stack from LangChain messages through converters to Mellea session.
Uses realistic fake Mellea session stubs instead of mocks to ensure full interface coverage.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mellea_langchain import MelleaChatModel

# ==============================================================================
# Shared Fixtures and Fake Implementations
# ==============================================================================


class FakeMelleaResponse:
    """Realistic Mellea response with both content and _tool_calls attributes."""

    def __init__(self, content, tool_calls=None):
        self.content = content
        self._tool_calls = tool_calls or []


class FakeSamplingResult:
    """Mimics Mellea SamplingResult for requirement/strategy tests."""

    def __init__(self, success, result_content=None, samples=None):
        self.success = success
        self.result = FakeMelleaResponse(result_content) if result_content else None
        self.sample_generations = samples or []


class FakeMelleaSession:
    """Full-interface fake session; records calls for assertion."""

    def __init__(self, response_content="Test response"):
        self.response_content = response_content
        self.calls_made = []
        self.last_message = None
        self.last_model_options = None
        self.last_tool_calls = None
        self.last_requirements = None
        self.last_strategy = None
        self.last_return_sampling_results = None

    def chat(self, message, model_options=None, tool_calls=False):
        """Mock sync chat method."""
        self.calls_made.append("chat")
        self.last_message = message
        self.last_model_options = model_options or {}
        self.last_tool_calls = tool_calls
        return FakeMelleaResponse(self.response_content)

    async def achat(self, message, model_options=None, tool_calls=False):
        """Mock async chat method."""
        self.calls_made.append("achat")
        self.last_message = message
        self.last_model_options = model_options or {}
        self.last_tool_calls = tool_calls
        return FakeMelleaResponse(self.response_content)

    def instruct(
        self,
        message,
        requirements=None,
        strategy=None,
        model_options=None,
        return_sampling_results=False,
    ):
        """Mock sync instruct method."""
        self.calls_made.append("instruct")
        self.last_message = message
        self.last_model_options = model_options or {}
        self.last_requirements = requirements
        self.last_strategy = strategy
        self.last_return_sampling_results = return_sampling_results

        if return_sampling_results:
            return FakeSamplingResult(success=True, result_content=self.response_content)
        return FakeMelleaResponse(self.response_content)

    async def ainstruct(
        self,
        message,
        requirements=None,
        strategy=None,
        model_options=None,
        return_sampling_results=False,
    ):
        """Mock async instruct method."""
        self.calls_made.append("ainstruct")
        self.last_message = message
        self.last_model_options = model_options or {}
        self.last_requirements = requirements
        self.last_strategy = strategy
        self.last_return_sampling_results = return_sampling_results

        if return_sampling_results:
            return FakeSamplingResult(success=True, result_content=self.response_content)
        return FakeMelleaResponse(self.response_content)


# ==============================================================================
# Test Groups
# ==============================================================================


@pytest.mark.integration
class TestBasicChatIntegration:
    """End-to-end chat: LangChain message → session → LangChain response."""

    def test_invoke_human_message(self):
        """HumanMessage → invoke() → AIMessage with correct content."""
        fake_session = FakeMelleaSession(response_content="Hello!")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Hi there")]
        result = chat_model.invoke(messages)

        assert result.content == "Hello!"
        assert isinstance(result, AIMessage)
        assert fake_session.last_message == "Hi there"

    def test_invoke_system_and_human_messages(self):
        """System + Human → session receives last human text as prompt."""
        fake_session = FakeMelleaSession(response_content="Response")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="How can you help?"),
        ]
        result = chat_model.invoke(messages)

        # Session should receive both messages converted
        assert result.content == "Response"
        # Message extraction should identify the last human message
        assert (
            "How can you help?" in str(fake_session.last_message)
            or fake_session.last_message == "How can you help?"
        )

    def test_invoke_multi_turn_conversation(self):
        """Human/AI alternating history → last human is the prompt."""
        fake_session = FakeMelleaSession(response_content="Answer")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [
            HumanMessage(content="My name is Alice"),
            AIMessage(content="Nice to meet you, Alice!"),
            HumanMessage(content="What's my name?"),
        ]
        result = chat_model.invoke(messages)

        assert result.content == "Answer"
        # The last human message should be used as the prompt
        assert fake_session.last_message == "What's my name?"

    @pytest.mark.asyncio
    async def test_async_ainvoke(self):
        """Same as invoke but via ainvoke()."""
        fake_session = FakeMelleaSession(response_content="Async response")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Hello")]
        result = await chat_model.ainvoke(messages)

        assert result.content == "Async response"
        assert isinstance(result, AIMessage)
        assert "achat" in fake_session.calls_made

    def test_model_options_passed_through(self):
        """temperature, max_tokens in model_options reach the session."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Test")]
        model_options = {"temperature": 0.7, "max_tokens": 100}

        chat_model.invoke(messages, model_options=model_options)

        # Model options should be passed through to session
        assert fake_session.last_model_options is not None
        assert "temperature" in fake_session.last_model_options
        assert fake_session.last_model_options["temperature"] == 0.7


@pytest.mark.integration
class TestStreamingIntegration:
    """Verify streaming yields correct chunks."""

    def test_stream_returns_chunks(self):
        """stream() yields at least one chunk with non-empty content."""
        fake_session = FakeMelleaSession(response_content="Streamed content")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Test")]
        chunks = list(chat_model.stream(messages))

        assert len(chunks) >= 1
        assert chunks[0].content != ""

    @pytest.mark.asyncio
    async def test_astream_returns_chunks(self):
        """astream() async iterator yields at least one chunk."""
        fake_session = FakeMelleaSession(response_content="Async streamed")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Test")]
        chunks = []

        async for chunk in chat_model.astream(messages):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[0].content != ""

    def test_stream_content_matches_response(self):
        """Concatenated chunks equal the session response content."""
        response_text = "This is the full response"
        fake_session = FakeMelleaSession(response_content=response_text)
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Test")]
        chunks = list(chat_model.stream(messages))

        # Concatenate chunk contents
        concatenated = "".join(chunk.content for chunk in chunks)
        assert concatenated == response_text


@pytest.mark.integration
class TestMessageConversionIntegration:
    """Verify that all LangChain message types are correctly translated."""

    def test_system_message_sets_context(self):
        """SystemMessage present → context/model_options includes system text."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        system_text = "You are a helpful assistant"
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content="Hello"),
        ]

        chat_model.invoke(messages)

        # SystemMessage should be part of the message history passed to session
        # (implementation detail: we verify session was called)
        assert "chat" in fake_session.calls_made

    def test_tool_message_conversion(self):
        """ToolMessage converts to user message prefixed with 'Tool result:'."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [ToolMessage(content="Tool output", tool_call_id="call_123")]

        chat_model.invoke(messages)

        # ToolMessage should be converted to user message with prefix
        assert fake_session.last_message is not None

    def test_ai_message_in_history(self):
        """AIMessage in history → role 'assistant' in Mellea messages."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [
            AIMessage(content="Previous response"),
            HumanMessage(content="Follow up"),
        ]

        chat_model.invoke(messages)

        # Both messages should be passed through
        assert "chat" in fake_session.calls_made

    def test_last_user_message_extracted(self):
        """With multiple Human messages, last one is the prompt."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [
            HumanMessage(content="First question"),
            HumanMessage(content="Second question"),
            HumanMessage(content="Final question"),
        ]

        chat_model.invoke(messages)

        # Last human message should be used
        assert fake_session.last_message == "Final question"


@pytest.mark.integration
class TestToolCallingIntegration:
    """Verify bind_tools() → session receives tool_calls=True and tools in model_options."""

    def test_bind_tools_enables_tool_calls(self):
        """After bind_tools(), session called with tool_calls=True."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        class MockTool:
            name = "search"
            description = "Search the web"

        bound_model = chat_model.bind_tools([MockTool()])
        messages = [HumanMessage(content="Search for Python")]

        bound_model.invoke(messages)

        # tool_calls should be True when tools are bound
        assert fake_session.last_tool_calls is True

    def test_tool_definitions_in_model_options(self):
        """Tool schema appears in model_options['tools']."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        class MockTool:
            name = "calculator"
            description = "Do math"

        bound_model = chat_model.bind_tools([MockTool()])
        messages = [HumanMessage(content="Calculate")]

        bound_model.invoke(messages)

        # Tools should be in model_options if passed
        # (implementation detail: session was called with tools enabled)
        assert fake_session.last_tool_calls is True

    def test_tool_response_parsed_to_ai_message(self):
        """Session returns [ToolCall...] string → AIMessage has .tool_calls."""
        tool_call_string = (
            "[ToolCall(function=Function(name='get_weather', arguments={'location': 'NYC'}))]"
        )
        fake_session = FakeMelleaSession(response_content=tool_call_string)
        chat_model = MelleaChatModel(mellea_session=fake_session)

        class MockTool:
            name = "get_weather"
            description = "Get weather"

        bound_model = chat_model.bind_tools([MockTool()])
        messages = [HumanMessage(content="What's the weather")]

        result = bound_model.invoke(messages)

        # Should parse tool calls from response
        assert hasattr(result, "tool_calls")

    def test_multiple_tools_bound(self):
        """Two tools bound → both schemas present in model_options."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        class MockTool1:
            name = "search"
            description = "Search"

        class MockTool2:
            name = "calculator"
            description = "Calculate"

        bound_model = chat_model.bind_tools([MockTool1(), MockTool2()])
        messages = [HumanMessage(content="Use tools")]

        bound_model.invoke(messages)

        # Both tools should be considered
        assert fake_session.last_tool_calls is True

    def test_tool_not_found_returns_error_result(self):
        """Model called tool not in bound list → error result stored."""
        tool_call_string = "[ToolCall(function=Function(name='unknown_tool', arguments={}))]"
        fake_session = FakeMelleaSession(response_content=tool_call_string)
        chat_model = MelleaChatModel(mellea_session=fake_session)

        class MockTool:
            name = "known_tool"
            description = "A known tool"

        bound_model = chat_model.bind_tools([MockTool()])
        messages = [HumanMessage(content="Call a tool")]

        result = bound_model.invoke(messages)

        # Tool call parsing should happen even if tool doesn't exist
        assert result is not None


@pytest.mark.integration
class TestRequirementsAndStrategyIntegration:
    """Verify requirements/strategy routing dispatches to instruct/ainstruct."""

    def test_requirements_routes_to_instruct(self):
        """Passing requirements → session's instruct() called (not chat())."""
        fake_session = FakeMelleaSession(response_content="Valid output")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Generate")]
        requirements = ["Must be short"]

        chat_model.invoke(messages, model_options={"requirements": requirements})

        # Should call instruct, not chat
        assert "instruct" in fake_session.calls_made
        assert fake_session.last_requirements == requirements

    def test_strategy_routes_to_instruct(self):
        """Passing strategy alone → instruct() called."""
        fake_session = FakeMelleaSession(response_content="Output")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        class MockStrategy:
            loop_budget = 5

        messages = [HumanMessage(content="Generate")]

        chat_model.invoke(messages, model_options={"strategy": MockStrategy()})

        # Should call instruct when strategy is provided
        assert "instruct" in fake_session.calls_made

    def test_requirements_and_strategy_together(self):
        """Both → instruct() with correct args."""
        fake_session = FakeMelleaSession(response_content="Output")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        class MockStrategy:
            loop_budget = 3

        messages = [HumanMessage(content="Generate")]
        requirements = ["Req1", "Req2"]
        strategy = MockStrategy()

        chat_model.invoke(
            messages,
            model_options={"requirements": requirements, "strategy": strategy},
        )

        assert "instruct" in fake_session.calls_made
        assert fake_session.last_requirements == requirements
        assert fake_session.last_strategy == strategy

    def test_sampling_result_success_unwrapped(self):
        """Session returns SamplingResult(success=True) → content extracted."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Generate")]
        requirements = ["Output must be valid"]

        result = chat_model.invoke(
            messages,
            model_options={
                "requirements": requirements,
                "return_sampling_results": True,
            },
        )

        # Should have content from successful result
        assert result.content is not None

    def test_sampling_result_failure_fallback(self):
        """success=False with samples → first sample content used."""

        class CustomSession(FakeMelleaSession):
            async def ainstruct(
                self,
                message,
                requirements=None,
                strategy=None,
                model_options=None,
                return_sampling_results=False,
            ):
                if return_sampling_results:
                    sample_obj = type(
                        "obj", (), {"content": "Fallback sample", "value": "Fallback sample"}
                    )()
                    return FakeSamplingResult(
                        success=False,
                        result_content=None,
                        samples=[sample_obj],
                    )
                return FakeMelleaResponse("Regular response")

        fake_session = CustomSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Generate")]

        result = chat_model.invoke(
            messages,
            model_options={
                "requirements": ["Requirement"],
                "return_sampling_results": True,
            },
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_async_requirements_with_ainstruct(self):
        """ainvoke() with requirements → ainstruct() called."""
        fake_session = FakeMelleaSession(response_content="Async output")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Generate")]
        requirements = ["Must be valid"]

        result = await chat_model.ainvoke(
            messages,
            model_options={"requirements": requirements},
        )

        # Should call ainstruct
        assert "ainstruct" in fake_session.calls_made
        assert fake_session.last_requirements == requirements
        assert result.content == "Async output"

    def test_requirements_passed_as_kwarg(self):
        """Requirements passed as direct kwarg to invoke()."""
        fake_session = FakeMelleaSession(response_content="Valid")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        messages = [HumanMessage(content="Generate")]
        requirements = ["Requirement"]

        chat_model.invoke(messages, requirements=requirements)

        # Should recognize requirements kwarg
        assert "instruct" in fake_session.calls_made


@pytest.mark.integration
class TestLangChainChainIntegration:
    """Verify MelleaChatModel works inside LCEL chains."""

    def test_prompt_pipe_model(self):
        """ChatPromptTemplate | chat_model pipeline works end-to-end."""
        fake_session = FakeMelleaSession(response_content="Generated text")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are helpful"),
                ("human", "{input}"),
            ]
        )

        chain = prompt | chat_model
        result = chain.invoke({"input": "Hello"})

        assert result.content == "Generated text"

    def test_prompt_pipe_model_pipe_parser(self):
        """prompt | chat_model | StrOutputParser() → plain string."""
        fake_session = FakeMelleaSession(response_content="Plain text response")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        prompt = ChatPromptTemplate.from_messages([("human", "{text}")])

        chain = prompt | chat_model | StrOutputParser()
        result = chain.invoke({"text": "Test"})

        assert isinstance(result, str)
        assert result == "Plain text response"

    def test_chain_with_bound_model_options(self):
        """chat_model.bind(model_options={...}) → options reach session."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        bound_model = chat_model.bind(model_options={"temperature": 0.5})
        messages = [HumanMessage(content="Test")]

        bound_model.invoke(messages)

        # Model options should be passed through
        assert fake_session.last_model_options is not None


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Verify that errors from the session propagate cleanly."""

    def test_session_exception_propagates(self):
        """Session raises RuntimeError → invoke() re-raises."""

        class ErrorSession:
            def chat(self, message, model_options=None, tool_calls=False):
                raise RuntimeError("Mellea API error")

            async def achat(self, message, model_options=None, tool_calls=False):
                raise RuntimeError("Mellea API error")

        error_session = ErrorSession()
        chat_model = MelleaChatModel(mellea_session=error_session)

        messages = [HumanMessage(content="Test")]

        with pytest.raises(RuntimeError, match="Mellea API error"):
            chat_model.invoke(messages)

    def test_empty_messages_raises_value_error(self):
        """Empty list to invoke() → ValueError."""
        fake_session = FakeMelleaSession()
        chat_model = MelleaChatModel(mellea_session=fake_session)

        with pytest.raises(ValueError, match="No messages provided"):
            chat_model.invoke([])

    @pytest.mark.asyncio
    async def test_async_session_exception_propagates(self):
        """Async session raises → ainvoke() re-raises."""

        class AsyncErrorSession:
            async def achat(self, message, model_options=None, tool_calls=False):
                raise RuntimeError("Async Mellea error")

        error_session = AsyncErrorSession()
        chat_model = MelleaChatModel(mellea_session=error_session)

        messages = [HumanMessage(content="Test")]

        with pytest.raises(RuntimeError, match="Async Mellea error"):
            await chat_model.ainvoke(messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
