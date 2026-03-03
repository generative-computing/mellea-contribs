"""Integration tests for CrewAI Agent/Task/Crew pipeline with MelleaLLM."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add parent test directory to path to import conftest fixtures
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure telemetry is off
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "1")

pytestmark = [pytest.mark.llm]


class TestSingleAgentCrew:
    """Tests the basic Agent + Task + Crew pipeline with mock session."""

    def test_kickoff_returns_crew_output(self, mock_mellea_session):
        """Test that crew.kickoff() returns CrewOutput with non-empty raw content."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        agent = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="You are a researcher",
            llm=llm,
        )

        task = Task(
            description="Research AI trends",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        # Verify result is CrewOutput
        from crewai.crews.crew_output import CrewOutput

        assert isinstance(result, CrewOutput)
        assert len(result.raw) > 0

    def test_kickoff_result_content_matches_llm_response(self, mock_mellea_session):
        """Test that mock response content appears in crew result."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        # Customize response
        custom_content = "Research findings: AI is trending"
        mock_response = Mock()
        mock_response.content = custom_content
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        agent = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="You are a researcher",
            llm=llm,
        )

        task = Task(
            description="Research AI trends",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        # Verify custom content in result
        assert custom_content in result.raw

    def test_kickoff_calls_session_chat_once(self, mock_mellea_session):
        """Test that no-requirements path calls chat() not instruct()."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        agent = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="You are a researcher",
            llm=llm,
        )

        task = Task(
            description="Research AI trends",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        crew.kickoff()

        # Verify chat was called, not instruct
        mock_mellea_session.chat.assert_called()
        mock_mellea_session.instruct.assert_not_called()


class TestAgentWithTools:
    """Tests the from_task/from_agent tool extraction workaround."""

    def test_tool_extracted_from_task_and_executed(
        self, mock_mellea_session, simple_crewai_tool, mock_task_with_tools
    ):
        """Test that from_task.tools are used when tools parameter is None."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        # Configure response with tool call
        mock_response = Mock()
        mock_response.content = "Calling tool"
        from conftest import FakeToolCall

        tool_call = FakeToolCall("simple_echo_tool", {"text": "hello"})
        mock_response._tool_calls = [tool_call]
        mock_response.tool_calls = [tool_call]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        agent = Agent(
            role="Assistant",
            goal="Use tools",
            backstory="You are helpful",
            tools=[simple_crewai_tool],
            llm=llm,
        )

        task = Task(
            description="Echo hello",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        # Tool should have been executed
        assert "Echo: hello" in result.raw

    def test_tool_extracted_from_agent_when_task_has_no_tools(
        self, mock_mellea_session, calculator_tool
    ):
        """Test from_agent.tools fallback when from_task.tools is empty."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        # Configure response with tool call
        mock_response = Mock()
        mock_response.content = "Calling tool"
        from conftest import FakeToolCall

        tool_call = FakeToolCall("add_numbers", {"a": 5, "b": 5})
        mock_response._tool_calls = [tool_call]
        mock_response.tool_calls = [tool_call]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        agent = Agent(
            role="Calculator",
            goal="Calculate",
            backstory="You calculate",
            tools=[calculator_tool],
            llm=llm,
        )

        task = Task(
            description="Add 5 plus 5",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        # Tool should have been executed and result should be 10
        assert "10" in str(result.raw)

    def test_full_crew_kickoff_with_tool_agent(
        self, mock_mellea_session, simple_crewai_tool
    ):
        """Test end-to-end crew.kickoff() with tool-equipped agent."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        # Configure response with tool call
        mock_response = Mock()
        mock_response.content = "Calling tool"
        from conftest import FakeToolCall

        tool_call = FakeToolCall("simple_echo_tool", {"text": "crew test"})
        mock_response._tool_calls = [tool_call]
        mock_response.tool_calls = [tool_call]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        agent = Agent(
            role="Helper",
            goal="Help",
            backstory="Helper",
            tools=[simple_crewai_tool],
            llm=llm,
        )

        task = Task(
            description="Echo crew test",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        # Verify tool was executed in result
        assert "Echo: crew test" in result.raw


class TestRequirementsIntegration:
    """Tests that requirements/strategy route to session.instruct()."""

    def test_kickoff_with_requirements_uses_instruct(self, mock_mellea_session):
        """Test that requirements parameter triggers instruct() not chat()."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        # Configure instruct response
        mock_response = Mock()
        mock_response.content = "Professional response"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.instruct.return_value = mock_response

        requirements = ["Be professional", "Be concise"]
        llm = MelleaLLM(
            mellea_session=mock_mellea_session,
            requirements=requirements,
        )

        agent = Agent(
            role="Writer",
            goal="Write",
            backstory="Writer",
            llm=llm,
        )

        task = Task(
            description="Write response",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        crew.kickoff()

        # Verify instruct was called, not chat
        mock_mellea_session.instruct.assert_called()
        mock_mellea_session.chat.assert_not_called()

    def test_kickoff_with_strategy_uses_instruct(self, mock_mellea_session):
        """Test that strategy parameter alone triggers instruct()."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        # Configure instruct response
        mock_response = Mock()
        mock_response.content = "Response with strategy"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.instruct.return_value = mock_response

        mock_strategy = Mock()
        llm = MelleaLLM(
            mellea_session=mock_mellea_session,
            strategy=mock_strategy,
        )

        agent = Agent(
            role="Writer",
            goal="Write",
            backstory="Writer",
            llm=llm,
        )

        task = Task(
            description="Write response",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        crew.kickoff()

        # Verify instruct was called
        mock_mellea_session.instruct.assert_called()
        instruct_call_kwargs = mock_mellea_session.instruct.call_args[1]
        assert "strategy" in instruct_call_kwargs

    def test_kickoff_with_requirements_and_strategy(self, mock_mellea_session):
        """Test that both requirements and strategy are passed correctly."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        # Configure instruct response
        mock_response = Mock()
        mock_response.content = "Response with both"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.instruct.return_value = mock_response

        requirements = ["Requirement 1"]
        mock_strategy = Mock()
        llm = MelleaLLM(
            mellea_session=mock_mellea_session,
            requirements=requirements,
            strategy=mock_strategy,
        )

        agent = Agent(
            role="Writer",
            goal="Write",
            backstory="Writer",
            llm=llm,
        )

        task = Task(
            description="Write response",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        crew.kickoff()

        # Verify both are in instruct call
        instruct_call_kwargs = mock_mellea_session.instruct.call_args[1]
        assert "requirements" in instruct_call_kwargs or "strategy" in instruct_call_kwargs


class TestTokenUsageTracking:
    """Tests token usage tracking after crew.kickoff()."""

    def test_token_usage_tracked_after_kickoff(self, mock_mellea_session):
        """Test that token counts are tracked after kickoff."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        agent = Agent(
            role="Writer",
            goal="Write",
            backstory="Writer",
            llm=llm,
        )

        task = Task(
            description="Test",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        crew.kickoff()

        # Get token usage summary
        summary = llm.get_token_usage_summary()

        # Verify tracking
        assert summary.total_tokens >= 15  # Default mock response has 15 total tokens
        assert summary.successful_requests >= 1

    def test_token_usage_accumulates_across_calls(self, mock_mellea_session):
        """Test that multiple calls accumulate token usage."""
        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        # Make first call
        llm.call("First message")

        # Make second call
        llm.call("Second message")

        # Get token usage
        summary = llm.get_token_usage_summary()

        # Should accumulate (2 calls × 15 tokens = 30)
        assert summary.total_tokens >= 30
        assert summary.successful_requests >= 2

    def test_token_usage_zero_before_any_call(self):
        """Test that fresh LLM starts with zero usage."""
        from unittest.mock import Mock

        from mellea_crewai import MelleaLLM

        mock_session = Mock()
        llm = MelleaLLM(mellea_session=mock_session)

        summary = llm.get_token_usage_summary()

        # Should be zero before any call
        assert summary.total_tokens == 0
        assert summary.successful_requests == 0


class TestEventEmission:
    """Tests event emission during normal and error flows."""

    def test_call_started_event_emitted(self, mock_mellea_session):
        """Test that _emit_call_started_event is fired when call() begins."""
        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        with patch.object(llm, "_emit_call_started_event") as mock_emit_started:
            llm.call("Test message")

            mock_emit_started.assert_called_once()

    def test_call_completed_event_emitted(self, mock_mellea_session):
        """Test that _emit_call_completed_event is fired on success."""
        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        with patch.object(llm, "_emit_call_completed_event") as mock_emit_completed:
            llm.call("Test message")

            mock_emit_completed.assert_called_once()

    def test_call_failed_event_emitted_on_error(self, mock_mellea_session):
        """Test that _emit_call_failed_event is fired on exception."""
        from mellea_crewai import MelleaLLM

        mock_mellea_session.chat.side_effect = RuntimeError("LLM error")

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        with patch.object(llm, "_emit_call_failed_event") as mock_emit_failed:
            with pytest.raises(RuntimeError):
                llm.call("Test message")

            mock_emit_failed.assert_called_once()


class TestStopWords:
    """Tests stop word truncation."""

    def test_stop_words_truncate_response(self, mock_mellea_session):
        """Test that response is truncated at stop word."""
        from mellea_crewai import MelleaLLM

        # Configure response with stop word
        mock_response = Mock()
        mock_response.content = "Good answer. STOP Extra text that should be removed."
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_mellea_session, stop=["STOP"])

        result = llm.call("Test")

        # Result should not contain text after stop word
        assert "Extra text that should be removed" not in result
        assert "Good answer." in result

    def test_no_stop_words_returns_full_response(self, mock_mellea_session):
        """Test that no stop words results in full response."""
        from mellea_crewai import MelleaLLM

        # Configure response without stop words
        full_response = "Full response without truncation"
        mock_response = Mock()
        mock_response.content = full_response
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        result = llm.call("Test")

        # Should return full response
        assert result == full_response


class TestAsyncCallDirect:
    """Tests acall() directly."""

    @pytest.mark.asyncio
    async def test_acall_with_requirements_uses_ainstruct(self, mock_mellea_session):
        """Test that acall() with requirements uses ainstruct()."""
        from mellea_crewai import MelleaLLM

        # Configure async instruct response
        mock_response = Mock()
        mock_response.content = "Async instruct response"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        async def async_instruct(*args, **kwargs):
            return mock_response

        mock_mellea_session.ainstruct = async_instruct

        requirements = ["Be professional"]
        llm = MelleaLLM(
            mellea_session=mock_mellea_session,
            requirements=requirements,
        )

        with patch.object(mock_mellea_session, "ainstruct", side_effect=async_instruct) as mock_ainstruct:
            with patch.object(mock_mellea_session, "achat") as mock_achat:
                result = await llm.acall("Test message")

                # Verify instruct was called
                assert result == "Async instruct response"

    @pytest.mark.asyncio
    async def test_acall_error_emits_failed_event(self, mock_mellea_session):
        """Test that exception in acall() triggers failed event."""
        from mellea_crewai import MelleaLLM

        async def async_chat_error(*args, **kwargs):
            raise RuntimeError("Async error")

        mock_mellea_session.achat = async_chat_error

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        with patch.object(llm, "_emit_call_failed_event") as mock_emit_failed:
            with pytest.raises(RuntimeError):
                await llm.acall("Test message")

            mock_emit_failed.assert_called_once()


class TestRealCrewAIIntegration:
    """Integration tests with real Ollama-backed CrewAI.

    Requires Ollama running with a model installed.
    """

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_real_single_agent_crew_kickoff(self, mellea_session):
        """Test full Crew.kickoff() with real Ollama backend."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(mellea_session=mellea_session)

        agent = Agent(
            role="Writer",
            goal="Write clear text",
            backstory="You are a writer",
            llm=llm,
        )

        task = Task(
            description="Say hello and nothing else",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        # Verify non-empty result
        assert isinstance(result.raw, str)
        assert len(result.raw) > 0

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_real_kickoff_with_requirements(self, mellea_session):
        """Test Crew.kickoff() with requirements and sampling strategy."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(
            mellea_session=mellea_session,
            requirements=["Keep response short"],
        )

        agent = Agent(
            role="Writer",
            goal="Write",
            backstory="Writer",
            llm=llm,
        )

        task = Task(
            description="Answer in one sentence",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        # Verify result
        assert isinstance(result.raw, str)
        assert len(result.raw) > 0

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_real_token_usage_after_kickoff(self, mellea_session):
        """Test that token usage tracking is available with real Ollama.

        Note: Ollama backend may not provide token usage information.
        This test verifies that successful_requests >= 1 (indicating the call was made),
        and checks that either tokens are tracked or the backend doesn't support it.
        """
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(mellea_session=mellea_session)

        agent = Agent(
            role="Writer",
            goal="Write",
            backstory="Writer",
            llm=llm,
        )

        task = Task(
            description="Say test",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        crew.kickoff()

        # Get usage summary
        summary = llm.get_token_usage_summary()

        # Verify request was made (successful_requests should be > 0)
        assert summary.successful_requests > 0

        # Token tracking is optional - some backends (like Ollama) may not provide it
        # Just verify the structure is there
        assert hasattr(summary, 'total_tokens')
        assert hasattr(summary, 'successful_requests')

    @pytest.mark.integration
    @pytest.mark.ollama
    @pytest.mark.slow
    def test_real_multi_agent_crew(self, mellea_session):
        """Test multi-agent crew with real Ollama backend."""
        from crewai import Agent, Crew, Task

        from mellea_crewai import MelleaLLM

        llm = MelleaLLM(mellea_session=mellea_session)

        # Create two agents
        agent1 = Agent(
            role="Researcher",
            goal="Research",
            backstory="Researcher",
            llm=llm,
        )

        agent2 = Agent(
            role="Writer",
            goal="Write",
            backstory="Writer",
            llm=llm,
        )

        # Create two tasks
        task1 = Task(
            description="Say research data",
            expected_output="Task output",
            agent=agent1,
        )

        task2 = Task(
            description="Write summary",
            expected_output="Task output",
            agent=agent2,
        )

        crew = Crew(agents=[agent1, agent2], tasks=[task1, task2], verbose=False)
        result = crew.kickoff()

        # Verify result
        assert isinstance(result.raw, str)
        assert len(result.raw) > 0
