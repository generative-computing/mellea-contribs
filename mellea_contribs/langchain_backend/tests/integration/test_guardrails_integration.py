"""Integration tests for Mellea guardrails with LangChain.

Tests the full integration of MelleaOutputParser and MelleaGuardrail
with LangChain chains and real validation scenarios.
"""

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mellea_langchain import MelleaChatModel, MelleaGuardrail, MelleaOutputParser

# ==============================================================================
# Shared Fixtures
# ==============================================================================


class FakeMelleaResponse:
    """Realistic Mellea response."""

    def __init__(self, content):
        self.content = content
        self._tool_calls = []


class FakeMelleaSession:
    """Fake session for testing."""

    def __init__(self, response_content="Test response"):
        self.response_content = response_content
        self.calls_made = []

    def chat(self, message, model_options=None, tool_calls=False):
        """Mock sync chat method."""
        self.calls_made.append("chat")
        return FakeMelleaResponse(self.response_content)

    async def achat(self, message, model_options=None, tool_calls=False):
        """Mock async chat method."""
        self.calls_made.append("achat")
        return FakeMelleaResponse(self.response_content)


def simple_validate(fn, description):
    """Create a simple validation function with description."""
    fn.__doc__ = description
    fn.description = description
    return fn


# ==============================================================================
# Test Groups
# ==============================================================================


@pytest.mark.integration
class TestOutputParserInChain:
    """Test MelleaOutputParser integrated in LangChain chains."""

    def test_parser_in_basic_chain(self):
        """prompt | model | parser works end-to-end."""
        fake_session = FakeMelleaSession(
            response_content="This is a valid response with enough words."
        )
        chat_model = MelleaChatModel(mellea_session=fake_session)

        # Create parser with word count requirement
        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x.split()) >= 5, "Must have at least 5 words"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        result = chain.invoke({"input": "Generate text"})

        assert isinstance(result, str)
        assert result == "This is a valid response with enough words."

    def test_parser_validation_failure_in_chain(self):
        """Parser raises OutputParserException when validation fails."""
        fake_session = FakeMelleaSession(response_content="Short")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x.split()) >= 10, "Must have at least 10 words"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        with pytest.raises(OutputParserException, match="Output validation failed"):
            chain.invoke({"input": "Generate text"})

    def test_parser_with_multiple_requirements(self):
        """Parser validates multiple requirements in chain."""
        response = "This is a proper sentence with capital letter and period."
        fake_session = FakeMelleaSession(response_content=response)
        chat_model = MelleaChatModel(mellea_session=fake_session)

        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 20, "Must be longer than 20 chars"),
                simple_validate(lambda x: x[0].isupper(), "Must start with capital"),
                simple_validate(lambda x: x.endswith("."), "Must end with period"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        result = chain.invoke({"input": "Generate text"})
        assert result == response

    def test_parser_non_strict_mode_in_chain(self):
        """Non-strict parser returns text even when validation fails."""
        fake_session = FakeMelleaSession(response_content="invalid")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 100, "Must be very long"),
            ],
            strict=False,
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        # Should not raise, returns text anyway
        result = chain.invoke({"input": "Generate text"})
        assert result == "invalid"

    def test_parser_with_str_output_parser(self):
        """Parser can be chained with StrOutputParser."""
        fake_session = FakeMelleaSession(response_content="Valid output text")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 5, "Must have content"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser | StrOutputParser()

        result = chain.invoke({"input": "Generate"})
        assert isinstance(result, str)
        assert result == "Valid output text"


@pytest.mark.integration
class TestGuardrailWithChain:
    """Test MelleaGuardrail used alongside chains."""

    def test_guardrail_validates_chain_output(self):
        """Guardrail validates output from a chain."""
        fake_session = FakeMelleaSession(response_content="This is a valid response.")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model

        # Run chain
        result = chain.invoke({"input": "Generate"})

        # Validate with guardrail
        guardrail = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: len(x) > 10, "Must be substantial"),
                simple_validate(lambda x: x.endswith("."), "Must end with period"),
            ],
            name="output_validator",
        )

        validation = guardrail.validate(result.content)
        assert validation.passed
        assert len(validation.errors) == 0

    def test_guardrail_catches_invalid_chain_output(self):
        """Guardrail detects validation failures from chain output."""
        fake_session = FakeMelleaSession(response_content="bad")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model

        result = chain.invoke({"input": "Generate"})

        guardrail = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: len(x) > 10, "Must be substantial"),
            ],
            name="validator",
        )

        validation = guardrail.validate(result.content)
        assert not validation.passed
        assert len(validation.errors) > 0

    def test_composed_guardrails_with_chain(self):
        """Multiple guardrails composed and used with chain."""
        fake_session = FakeMelleaSession(response_content="This is a proper response.")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model

        # Create separate guardrails
        length_guard = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: len(x) > 10, "Must be long enough"),
            ],
            name="length",
        )

        format_guard = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: x[0].isupper(), "Must start with capital"),
                simple_validate(lambda x: x.endswith("."), "Must end with period"),
            ],
            name="format",
        )

        # Compose guardrails
        combined = length_guard & format_guard

        result = chain.invoke({"input": "Generate"})
        validation = combined.validate(result.content)

        assert validation.passed
        assert validation.metadata["total_requirements"] == 3


@pytest.mark.integration
class TestParserAndGuardrailTogether:
    """Test using both parser and guardrail in the same workflow."""

    def test_parser_in_chain_guardrail_post_validation(self):
        """Parser validates in chain, guardrail does additional checks."""
        fake_session = FakeMelleaSession(response_content="This is a valid response.")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        # Parser checks basic format
        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 5, "Must have content"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        result = chain.invoke({"input": "Generate"})

        # Guardrail does additional semantic checks
        guardrail = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: "valid" in x.lower(), "Must mention validity"),
            ],
            name="semantic_check",
        )

        validation = guardrail.validate(result)
        assert validation.passed

    def test_parser_strict_guardrail_lenient(self):
        """Parser is strict (raises), guardrail is lenient (returns result)."""
        fake_session = FakeMelleaSession(response_content="short")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        # Strict parser
        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 100, "Must be very long"),
            ],
            strict=True,
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        # Parser should raise
        with pytest.raises(OutputParserException):
            chain.invoke({"input": "Generate"})

        # But guardrail just reports
        guardrail = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: len(x) > 100, "Must be very long"),
            ],
            name="lenient_check",
        )

        validation = guardrail.validate("short")
        assert not validation.passed
        assert len(validation.errors) > 0


@pytest.mark.integration
class TestAsyncIntegration:
    """Test async operations with guardrails."""

    @pytest.mark.asyncio
    async def test_async_chain_with_parser(self):
        """Async chain with parser works correctly."""
        fake_session = FakeMelleaSession(response_content="Valid async response.")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 5, "Must have content"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        result = await chain.ainvoke({"input": "Generate"})
        assert isinstance(result, str)
        assert len(result) > 5

    @pytest.mark.asyncio
    async def test_async_chain_validation_failure(self):
        """Async chain raises OutputParserException on validation failure."""
        fake_session = FakeMelleaSession(response_content="bad")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 100, "Must be very long"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        with pytest.raises(OutputParserException):
            await chain.ainvoke({"input": "Generate"})


@pytest.mark.integration
class TestComplexValidationScenarios:
    """Test complex real-world validation scenarios."""

    def test_multi_stage_validation_pipeline(self):
        """Multiple validation stages in a pipeline."""
        fake_session = FakeMelleaSession(
            response_content="This is a professional response with proper formatting."
        )
        chat_model = MelleaChatModel(mellea_session=fake_session)

        # Stage 1: Basic format validation (parser)
        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 10, "Must have content"),
                simple_validate(lambda x: x[0].isupper(), "Must start with capital"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        result = chain.invoke({"input": "Generate"})

        # Stage 2: Content validation (guardrail)
        content_guard = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: "professional" in x.lower(), "Must be professional"),
            ],
            name="content_check",
        )

        validation = content_guard.validate(result)
        assert validation.passed

    def test_progressive_validation_with_fallback(self):
        """Try strict validation, fall back to lenient on failure."""
        fake_session = FakeMelleaSession(response_content="Somewhat valid text")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model

        result = chain.invoke({"input": "Generate"})

        # Try strict validation first
        strict_guard = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: len(x) > 100, "Must be very long"),
                simple_validate(lambda x: "perfect" in x.lower(), "Must be perfect"),
            ],
            name="strict",
        )

        strict_validation = strict_guard.validate(result.content)

        if not strict_validation.passed:
            # Fall back to lenient validation
            lenient_guard = MelleaGuardrail(
                requirements=[
                    simple_validate(lambda x: len(x) > 5, "Must have some content"),
                ],
                name="lenient",
            )

            lenient_validation = lenient_guard.validate(result.content)
            assert lenient_validation.passed

    def test_validation_metadata_tracking(self):
        """Track detailed validation metadata through pipeline."""
        fake_session = FakeMelleaSession(response_content="This is a test response.")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model

        result = chain.invoke({"input": "Generate"})

        # Create guardrail with multiple requirements
        guardrail = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: len(x) > 10, "Length check"),
                simple_validate(lambda x: x[0].isupper(), "Capital check"),
                simple_validate(lambda x: x.endswith("."), "Period check"),
                simple_validate(lambda x: "test" in x.lower(), "Keyword check"),
            ],
            name="comprehensive",
        )

        validation = guardrail.validate(result.content)

        # Check metadata
        assert "total_requirements" in validation.metadata
        assert validation.metadata["total_requirements"] == 4
        assert "passed_requirements" in validation.metadata
        assert "failed_requirements" in validation.metadata
        assert validation.metadata["guardrail_name"] == "comprehensive"


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery patterns with guardrails."""

    def test_retry_on_validation_failure(self):
        """Retry generation when validation fails."""
        call_count = [0]

        class RetrySession:
            def chat(self, message, model_options=None, tool_calls=False):
                call_count[0] += 1
                if call_count[0] == 1:
                    return FakeMelleaResponse("bad")
                else:
                    return FakeMelleaResponse("This is a good response.")

        session = RetrySession()
        chat_model = MelleaChatModel(mellea_session=session)

        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 10, "Must be substantial"),
            ],
            strict=False,  # Non-strict to allow retry logic
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        # First attempt
        result1 = chain.invoke({"input": "Generate"})

        # Check with guardrail
        guardrail = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: len(x) > 10, "Must be substantial"),
            ],
            name="checker",
        )

        validation1 = guardrail.validate(result1)

        if not validation1.passed:
            # Retry
            result2 = chain.invoke({"input": "Generate better"})
            validation2 = guardrail.validate(result2)
            assert validation2.passed

    def test_custom_error_messages_in_chain(self):
        """Custom error messages propagate through chain."""
        fake_session = FakeMelleaSession(response_content="bad")
        chat_model = MelleaChatModel(mellea_session=fake_session)

        custom_template = "VALIDATION ERROR: {errors}\nPlease fix the output."

        parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) > 100, "Output too short"),
            ],
            error_message_template=custom_template,
        )

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | chat_model | parser

        try:
            chain.invoke({"input": "Generate"})
            assert False, "Should have raised"
        except OutputParserException as e:
            assert "VALIDATION ERROR" in str(e)
            assert "Output too short" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
