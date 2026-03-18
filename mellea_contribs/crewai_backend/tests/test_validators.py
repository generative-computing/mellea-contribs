"""Unit tests for validators module."""

from mellea_crewai.validators import (
    create_guardrail,
    create_guardrails,
)


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


class TestCreateGuardrail:
    """Test create_guardrail functionality."""

    def test_create_guardrail_with_simple_validate(self):
        """Test creating guardrail from simple_validate requirement."""
        requirement = simple_validate(lambda x: len(x) < 100, "Under 100 chars")
        guardrail = create_guardrail(requirement)

        assert callable(guardrail)
        assert "guardrail_" in guardrail.__name__
        assert "Under 100 chars" in guardrail.__doc__

    def test_guardrail_passes_valid_output(self):
        """Test guardrail passes with valid output."""
        requirement = simple_validate(lambda x: len(x) < 100, "Under 100 chars")
        guardrail = create_guardrail(requirement)

        result = MockTaskOutput("This is a short text")
        passed, value = guardrail(result)

        assert passed is True
        assert value == "This is a short text"

    def test_guardrail_fails_invalid_output(self):
        """Test guardrail fails with invalid output."""
        requirement = simple_validate(lambda x: len(x) < 10, "Under 10 chars")
        guardrail = create_guardrail(requirement)

        result = MockTaskOutput("This is a very long text that exceeds the limit")
        passed, value = guardrail(result)

        assert passed is False
        assert "Validation failed" in value
        assert "Under 10 chars" in value

    def test_guardrail_with_custom_error_prefix(self):
        """Test guardrail with custom error prefix."""
        requirement = simple_validate(lambda x: False, "Always fails")
        guardrail = create_guardrail(requirement, error_prefix="Custom error")

        result = MockTaskOutput("test")
        passed, value = guardrail(result)

        assert passed is False
        assert "Custom error" in value

    def test_guardrail_handles_exception(self):
        """Test guardrail handles validation exceptions."""

        def failing_validator(text):
            raise ValueError("Test error")

        requirement = simple_validate(failing_validator, "Failing validator")
        guardrail = create_guardrail(requirement)

        result = MockTaskOutput("test")
        passed, value = guardrail(result)

        assert passed is False
        assert "validation error" in value

    def test_guardrail_extracts_text_from_task_output(self):
        """Test guardrail correctly extracts text from TaskOutput."""
        requirement = simple_validate(lambda x: "test" in x, "Contains test")
        guardrail = create_guardrail(requirement)

        result = MockTaskOutput("this is a test message")
        passed, value = guardrail(result)

        assert passed is True
        assert value == "this is a test message"


class TestCreateGuardrails:
    """Test create_guardrails functionality."""

    def test_create_guardrails_multiple_requirements(self):
        """Test creating multiple guardrails at once."""
        requirements = [
            simple_validate(lambda x: len(x) > 10, "At least 10 chars"),
            simple_validate(lambda x: "test" in x, "Contains test"),
            simple_validate(lambda x: x.strip() == x, "No whitespace"),
        ]

        guardrails = create_guardrails(requirements)

        assert len(guardrails) == 3
        assert all(callable(g) for g in guardrails)

    def test_create_guardrails_empty_list(self):
        """Test creating guardrails with empty list."""
        guardrails = create_guardrails([])

        assert len(guardrails) == 0
        assert isinstance(guardrails, list)

    def test_create_guardrails_with_custom_error_prefix(self):
        """Test creating guardrails with custom error prefix."""
        requirements = [simple_validate(lambda x: False, "Always fails")]
        guardrails = create_guardrails(requirements, error_prefix="Custom")

        result = MockTaskOutput("test")
        passed, value = guardrails[0](result)

        assert passed is False
        assert "Custom" in value


