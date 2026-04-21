"""Integration tests for validators with CrewAI."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add parent test directory to path to import conftest fixtures
sys.path.insert(0, str(Path(__file__).parent.parent))

from mellea_crewai.validators import (
    create_guardrail,
    create_guardrails,
)

pytestmark = [pytest.mark.llm]


# Helper function to create simple validators
def simple_validate(fn, description):
    """Create a simple validator function with description."""

    def validator(text):
        return fn(text)

    validator.description = description
    validator.__doc__ = description
    return validator


# Mock TaskOutput for testing
class MockTaskOutput:
    """Mock TaskOutput for testing."""

    def __init__(self, raw: str):
        self.raw = raw


class TestGuardrailIntegration:
    """Test guardrail integration with CrewAI tasks."""

    def test_guardrail_with_valid_output(self, mock_mellea_session):
        """Test guardrail passes with valid task output."""

        from mellea_crewai import MelleaLLM

        # Setup mock to return valid output
        mock_response = Mock()
        mock_response.content = "This is a test message with enough words to pass validation"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        MelleaLLM(mellea_session=mock_mellea_session)

        # Create guardrail
        requirement = simple_validate(lambda x: len(x.split()) > 5, "At least 5 words")
        guardrail = create_guardrail(requirement)

        # Test guardrail directly
        result = MockTaskOutput(mock_response.content)
        passed, value = guardrail(result)

        assert passed is True
        assert value == mock_response.content

    def test_multiple_guardrails_sequential(self, mock_mellea_session):
        """Test multiple guardrails applied sequentially."""
        # Setup mock
        mock_response = Mock()
        mock_response.content = "This is a test message about AI and machine learning"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        # Create multiple guardrails
        requirements = [
            simple_validate(lambda x: len(x.split()) > 5, "At least 5 words"),
            simple_validate(lambda x: "AI" in x, "Must mention AI"),
            simple_validate(lambda x: len(x) < 200, "Under 200 chars"),
        ]
        guardrails = create_guardrails(requirements)

        # Test each guardrail
        result = MockTaskOutput(mock_response.content)
        for guardrail in guardrails:
            passed, value = guardrail(result)
            assert passed is True

    def test_guardrail_fails_invalid_output(self, mock_mellea_session):
        """Test guardrail fails with invalid output."""
        # Setup mock with short output
        mock_response = Mock()
        mock_response.content = "Short"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        # Create guardrail requiring more words
        requirement = simple_validate(lambda x: len(x.split()) > 10, "At least 10 words")
        guardrail = create_guardrail(requirement)

        # Test guardrail
        result = MockTaskOutput(mock_response.content)
        passed, value = guardrail(result)

        assert passed is False
        assert "Validation failed" in value

    def test_guardrail_with_mellea_llm(self, mock_mellea_session):
        """Test guardrail works with MelleaLLM."""
        from crewai import Agent

        from mellea_crewai import MelleaLLM

        # Setup mock
        mock_response = Mock()
        mock_response.content = "This is a comprehensive test message with sufficient content"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        llm = MelleaLLM(mellea_session=mock_mellea_session)

        # Create agent
        Agent(
            role="Writer",
            goal="Write content",
            backstory="You are a writer",
            llm=llm,
            verbose=False,
        )

        # Create guardrail
        requirement = simple_validate(lambda x: len(x) > 10, "At least 10 chars")
        guardrail = create_guardrail(requirement)

        # Note: We can't fully test Task.guardrails without actual CrewAI execution,
        # but we can verify the guardrail works with the mock output
        result = MockTaskOutput(mock_response.content)
        passed, value = guardrail(result)

        assert passed is True
        assert value == mock_response.content

    def test_guardrail_composition(self, mock_mellea_session):
        """Test composing multiple guardrails."""
        # Setup mock
        mock_response = Mock()
        mock_response.content = "AI and machine learning are transforming technology"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        # Create multiple requirements
        requirements = [
            simple_validate(lambda x: len(x.split()) >= 5, "At least 5 words"),
            simple_validate(lambda x: "AI" in x, "Must mention AI"),
        ]

        # Convert to guardrails
        guardrails = create_guardrails(requirements)

        # Test all guardrails pass
        result = MockTaskOutput(mock_response.content)
        for guardrail in guardrails:
            passed, value = guardrail(result)
            assert passed is True

    def test_guardrail_error_messages(self, mock_mellea_session):
        """Test guardrail provides clear error messages."""
        # Setup mock with invalid output
        mock_response = Mock()
        mock_response.content = "Short text"
        mock_response._tool_calls = []
        mock_response.tool_calls = []
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_mellea_session.chat.return_value = mock_response

        # Create guardrail with specific requirement
        requirement = simple_validate(lambda x: len(x.split()) >= 20, "Must have at least 20 words")
        guardrail = create_guardrail(requirement)

        # Test guardrail
        result = MockTaskOutput(mock_response.content)
        passed, value = guardrail(result)

        assert passed is False
        assert "Validation failed" in value
        assert "Must have at least 20 words" in value
