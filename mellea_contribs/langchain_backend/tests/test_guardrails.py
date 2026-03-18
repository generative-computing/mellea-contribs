"""Unit tests for guardrails module."""

import pytest
from langchain_core.exceptions import OutputParserException

from mellea_langchain.guardrails import (
    MelleaGuardrail,
    MelleaOutputParser,
    ValidationResult,
)


# Helper function to create simple validators
def simple_validate(fn, description):
    """Create a simple validator function with description."""

    def validator(text):
        return fn(text)

    validator.description = description
    validator.__doc__ = description
    return validator


class TestMelleaOutputParser:
    """Test MelleaOutputParser functionality."""

    def test_init_with_valid_requirements(self):
        """Test initialization with valid requirements."""
        requirements = [
            simple_validate(lambda x: len(x) < 100, "Under 100 chars"),
            simple_validate(lambda x: x.strip() == x, "No extra whitespace"),
        ]
        parser = MelleaOutputParser(requirements=requirements)

        assert len(parser.requirements) == 2
        assert len(parser.requirement_descriptions) == 2
        assert parser.strict is True

    def test_init_with_non_strict_mode(self):
        """Test initialization with non-strict mode."""
        requirements = [simple_validate(lambda x: True, "Always pass")]
        parser = MelleaOutputParser(requirements=requirements, strict=False)

        assert parser.strict is False

    def test_init_with_custom_error_template(self):
        """Test initialization with custom error message template."""
        requirements = [simple_validate(lambda x: True, "Test")]
        template = "Custom error: {errors}"
        parser = MelleaOutputParser(requirements=requirements, error_message_template=template)

        assert parser.error_message_template == template

    def test_init_with_invalid_requirement(self):
        """Test initialization with invalid requirement raises error."""
        with pytest.raises(ValueError, match="must be a callable"):
            MelleaOutputParser(requirements=["not a callable"])

    def test_parse_valid_output(self):
        """Test parsing valid output passes."""
        requirements = [
            simple_validate(lambda x: len(x) < 100, "Under 100 chars"),
            simple_validate(lambda x: "test" in x.lower(), "Contains 'test'"),
        ]
        parser = MelleaOutputParser(requirements=requirements)

        text = "This is a test message"
        result = parser.parse(text)

        assert result == text

    def test_parse_invalid_output_strict_mode(self):
        """Test parsing invalid output raises exception in strict mode."""
        requirements = [
            simple_validate(lambda x: len(x) < 10, "Under 10 chars"),
        ]
        parser = MelleaOutputParser(requirements=requirements, strict=True)

        text = "This is a very long message that exceeds the limit"

        with pytest.raises(OutputParserException) as exc_info:
            parser.parse(text)

        assert "Under 10 chars" in str(exc_info.value)
        assert text in exc_info.value.llm_output

    def test_parse_invalid_output_non_strict_mode(self):
        """Test parsing invalid output returns text in non-strict mode."""
        requirements = [
            simple_validate(lambda x: len(x) < 10, "Under 10 chars"),
        ]
        parser = MelleaOutputParser(requirements=requirements, strict=False)

        text = "This is a very long message that exceeds the limit"
        result = parser.parse(text)

        # Should return text even though validation failed
        assert result == text

    def test_parse_multiple_failed_requirements(self):
        """Test parsing with multiple failed requirements."""
        requirements = [
            simple_validate(lambda x: len(x) < 10, "Under 10 chars"),
            simple_validate(lambda x: x.isupper(), "All uppercase"),
            simple_validate(lambda x: "test" in x, "Contains 'test'"),
        ]
        parser = MelleaOutputParser(requirements=requirements, strict=True)

        text = "This is a long lowercase message"

        with pytest.raises(OutputParserException) as exc_info:
            parser.parse(text)

        error_msg = str(exc_info.value)
        assert "Under 10 chars" in error_msg
        assert "All uppercase" in error_msg
        assert "Contains 'test'" in error_msg

    def test_parse_with_validation_error(self):
        """Test parsing when validation function raises exception."""

        def buggy_validator(text):
            raise ValueError("Validation error")

        requirements = [
            simple_validate(buggy_validator, "Buggy requirement"),
        ]
        parser = MelleaOutputParser(requirements=requirements, strict=True)

        with pytest.raises(OutputParserException) as exc_info:
            parser.parse("test")

        assert "validation error" in str(exc_info.value).lower()

    def test_get_format_instructions_with_requirements(self):
        """Test get_format_instructions returns proper instructions."""
        requirements = [
            simple_validate(lambda x: True, "First requirement"),
            simple_validate(lambda x: True, "Second requirement"),
        ]
        parser = MelleaOutputParser(requirements=requirements)

        instructions = parser.get_format_instructions()

        assert "First requirement" in instructions
        assert "Second requirement" in instructions
        assert "1." in instructions
        assert "2." in instructions

    def test_get_format_instructions_empty_requirements(self):
        """Test get_format_instructions with no requirements."""
        parser = MelleaOutputParser(requirements=[])

        instructions = parser.get_format_instructions()

        assert "No specific requirements" in instructions

    def test_type_property(self):
        """Test _type property returns correct value."""
        parser = MelleaOutputParser(requirements=[])

        assert parser._type == "mellea_output_parser"

    def test_parse_empty_text(self):
        """Test parsing empty text."""
        requirements = [
            simple_validate(lambda x: len(x) > 0, "Not empty"),
        ]
        parser = MelleaOutputParser(requirements=requirements, strict=True)

        with pytest.raises(OutputParserException):
            parser.parse("")

    def test_parse_with_special_characters(self):
        """Test parsing text with special characters."""
        requirements = [
            simple_validate(lambda x: len(x) < 100, "Under 100 chars"),
        ]
        parser = MelleaOutputParser(requirements=requirements)

        text = "Test with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
        result = parser.parse(text)

        assert result == text


class TestMelleaGuardrail:
    """Test MelleaGuardrail functionality."""

    def test_init_with_valid_requirements(self):
        """Test initialization with valid requirements."""
        requirements = [
            simple_validate(lambda x: len(x) < 100, "Under 100 chars"),
        ]
        guardrail = MelleaGuardrail(requirements=requirements, name="test_guard")

        assert guardrail.name == "test_guard"
        assert len(guardrail.requirements) == 1

    def test_init_without_name(self):
        """Test initialization without name uses default."""
        requirements = [simple_validate(lambda x: True, "Test")]
        guardrail = MelleaGuardrail(requirements=requirements)

        assert guardrail.name == "unnamed_guardrail"

    def test_init_with_invalid_requirement(self):
        """Test initialization with invalid requirement raises error."""
        with pytest.raises(ValueError, match="must be a callable"):
            MelleaGuardrail(requirements=["not a callable"])

    def test_validate_passing(self):
        """Test validation with passing requirements."""
        requirements = [
            simple_validate(lambda x: len(x) < 100, "Under 100 chars"),
            simple_validate(lambda x: "test" in x.lower(), "Contains 'test'"),
        ]
        guardrail = MelleaGuardrail(requirements=requirements)

        result = guardrail.validate("This is a test message")

        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert len(result.errors) == 0
        assert result.text == "This is a test message"

    def test_validate_failing(self):
        """Test validation with failing requirements."""
        requirements = [
            simple_validate(lambda x: len(x) < 10, "Under 10 chars"),
            simple_validate(lambda x: x.isupper(), "All uppercase"),
        ]
        guardrail = MelleaGuardrail(requirements=requirements)

        result = guardrail.validate("This is a long lowercase message")

        assert result.passed is False
        assert len(result.errors) == 2
        assert "Under 10 chars" in result.errors
        assert "All uppercase" in result.errors

    def test_validate_partial_failure(self):
        """Test validation with some passing and some failing requirements."""
        requirements = [
            simple_validate(lambda x: len(x) > 5, "At least 5 chars"),  # Pass
            simple_validate(lambda x: x.isupper(), "All uppercase"),  # Fail
        ]
        guardrail = MelleaGuardrail(requirements=requirements)

        result = guardrail.validate("test message")

        assert result.passed is False
        assert len(result.errors) == 1
        assert "All uppercase" in result.errors

    def test_validate_metadata(self):
        """Test validation result includes metadata."""
        requirements = [
            simple_validate(lambda x: len(x) < 100, "Under 100 chars"),
            simple_validate(lambda x: len(x) > 5, "At least 5 chars"),
        ]
        guardrail = MelleaGuardrail(requirements=requirements, name="test_guard")

        result = guardrail.validate("test message")

        assert "guardrail_name" in result.metadata
        assert result.metadata["guardrail_name"] == "test_guard"
        assert result.metadata["total_requirements"] == 2
        assert result.metadata["passed_requirements"] == 2
        assert result.metadata["failed_requirements"] == 0

    def test_validate_with_exception(self):
        """Test validation when requirement function raises exception."""

        def buggy_validator(text):
            raise ValueError("Validation error")

        requirements = [
            simple_validate(buggy_validator, "Buggy requirement"),
        ]
        guardrail = MelleaGuardrail(requirements=requirements)

        result = guardrail.validate("test")

        assert result.passed is False
        assert len(result.errors) == 1
        assert "validation error" in result.errors[0].lower()

    def test_compose_two_guardrails(self):
        """Test composing two guardrails."""
        guard1 = MelleaGuardrail(
            requirements=[simple_validate(lambda x: len(x) < 100, "Under 100 chars")],
            name="length_guard",
        )
        guard2 = MelleaGuardrail(
            requirements=[simple_validate(lambda x: x.strip() == x, "No whitespace")],
            name="format_guard",
        )

        combined = guard1.compose(guard2)

        assert combined.name == "length_guard+format_guard"
        assert len(combined.requirements) == 2
        assert len(combined.requirement_descriptions) == 2

    def test_compose_multiple_guardrails(self):
        """Test composing multiple guardrails."""
        guard1 = MelleaGuardrail(
            requirements=[simple_validate(lambda x: True, "Req1")], name="guard1"
        )
        guard2 = MelleaGuardrail(
            requirements=[simple_validate(lambda x: True, "Req2")], name="guard2"
        )
        guard3 = MelleaGuardrail(
            requirements=[simple_validate(lambda x: True, "Req3")], name="guard3"
        )

        combined = guard1.compose(guard2).compose(guard3)

        assert len(combined.requirements) == 3
        assert "guard1" in combined.name
        assert "guard2" in combined.name
        assert "guard3" in combined.name

    def test_compose_with_and_operator(self):
        """Test composing guardrails with & operator."""
        guard1 = MelleaGuardrail(
            requirements=[simple_validate(lambda x: True, "Req1")], name="guard1"
        )
        guard2 = MelleaGuardrail(
            requirements=[simple_validate(lambda x: True, "Req2")], name="guard2"
        )

        combined = guard1 & guard2

        assert len(combined.requirements) == 2
        assert combined.name == "guard1+guard2"

    def test_compose_with_invalid_type(self):
        """Test composing with invalid type raises error."""
        guard = MelleaGuardrail(requirements=[simple_validate(lambda x: True, "Test")])

        with pytest.raises(TypeError, match="Can only compose with MelleaGuardrail"):
            guard.compose("not a guardrail")

    def test_repr(self):
        """Test string representation of guardrail."""
        guardrail = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: True, "Req1"),
                simple_validate(lambda x: True, "Req2"),
            ],
            name="test_guard",
        )

        repr_str = repr(guardrail)

        assert "MelleaGuardrail" in repr_str
        assert "test_guard" in repr_str
        assert "requirements=2" in repr_str


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_create_validation_result(self):
        """Test creating ValidationResult."""
        result = ValidationResult(passed=True, text="test", errors=[], metadata={"key": "value"})

        assert result.passed is True
        assert result.text == "test"
        assert result.errors == []
        assert result.metadata == {"key": "value"}

    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        result = ValidationResult(passed=False, text="test")

        assert result.errors == []
        assert result.metadata == {}

    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult(passed=False, text="test", errors=errors)

        assert result.errors == errors
        assert len(result.errors) == 2


class TestIntegration:
    """Integration tests for parser and guardrail working together."""

    def test_parser_and_guardrail_same_requirements(self):
        """Test parser and guardrail with same requirements produce consistent results."""
        requirements = [
            simple_validate(lambda x: len(x) < 50, "Under 50 chars"),
            simple_validate(lambda x: "test" in x.lower(), "Contains 'test'"),
        ]

        parser = MelleaOutputParser(requirements=requirements, strict=False)
        guardrail = MelleaGuardrail(requirements=requirements)

        valid_text = "This is a test"
        invalid_text = "This is a very long message without the required word and exceeds limit"

        # Both should pass for valid text
        parser_result = parser.parse(valid_text)
        guard_result = guardrail.validate(valid_text)
        assert parser_result == valid_text
        assert guard_result.passed is True

        # Both should fail for invalid text (parser in non-strict mode returns text)
        parser_result = parser.parse(invalid_text)
        guard_result = guardrail.validate(invalid_text)
        assert parser_result == invalid_text  # Non-strict returns text
        assert guard_result.passed is False

    def test_composed_guardrail_validation(self):
        """Test composed guardrail validates all requirements."""
        guard1 = MelleaGuardrail(
            requirements=[simple_validate(lambda x: len(x) > 10, "At least 10 chars")]
        )
        guard2 = MelleaGuardrail(
            requirements=[simple_validate(lambda x: len(x) < 50, "Under 50 chars")]
        )

        combined = guard1 & guard2

        # Valid text (between 10 and 50 chars)
        result = combined.validate("This is a valid message")
        assert result.passed is True

        # Too short
        result = combined.validate("Short")
        assert result.passed is False
        assert any("At least 10 chars" in err for err in result.errors)

        # Too long
        result = combined.validate(
            "This is a very long message that exceeds the fifty character limit"
        )
        assert result.passed is False
        assert any("Under 50 chars" in err for err in result.errors)
