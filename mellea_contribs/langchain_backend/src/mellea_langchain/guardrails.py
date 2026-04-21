"""LangChain-compatible output parsers and guardrails using Mellea requirements.

This module provides output validation components that integrate Mellea's
requirements system with LangChain's output parser pattern.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from pydantic import Field


@dataclass
class ValidationResult:
    """Result of validation with detailed information.

    Attributes:
        passed: Whether validation passed
        text: The validated text
        errors: List of error messages for failed requirements
        metadata: Additional metadata about validation
    """

    passed: bool
    text: str
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MelleaOutputParser(BaseOutputParser[str]):
    """Output parser that validates using Mellea requirements.

    This parser validates LLM outputs against a list of requirements
    using deterministic validation functions. It's designed to be stateless
    and work seamlessly with LangChain chains.

    The parser uses `simple_validate` functions from mellea.stdlib.requirements
    for deterministic, fast validation without requiring an LLM session.

    Example:
        ```python
        from mellea_langchain import MelleaOutputParser
        from mellea.stdlib.requirements import simple_validate

        parser = MelleaOutputParser(
            requirements=[
                simple_validate(
                    lambda x: len(x.split()) < 100,
                    "Must be under 100 words"
                ),
                simple_validate(
                    lambda x: "AI" in x,
                    "Must mention AI"
                ),
            ]
        )

        # Use in a chain
        chain = prompt | model | parser
        result = chain.invoke({"input": "..."})
        ```

    For LLM-based validation using req() or check(), use MelleaChatModel
    with requirements during generation instead:
        ```python
        from mellea.stdlib.requirements import req

        model = MelleaChatModel(
            mellea_session=m,
            requirements=[req("Must be professional")]
        )
        chain = prompt | model | parser  # LLM validation + deterministic validation
        ```
    """

    requirements: list[Callable[[str], bool]] = Field(default_factory=list)
    requirement_descriptions: list[str] = Field(default_factory=list)
    strict: bool = Field(default=True)
    error_message_template: str = Field(
        default="Output validation failed. The following requirements were not met:\n{errors}"
    )

    def __init__(
        self,
        requirements: list[Any] | None = None,
        strict: bool = True,
        error_message_template: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the output parser.

        Args:
            requirements: List of validation functions. Each should be created
                using simple_validate(fn, description) from mellea.stdlib.requirements.
                The function should take a string and return bool.
            strict: If True, raise OutputParserException on validation failure.
                If False, return the text even if validation fails.
            error_message_template: Custom error message format. Use {errors}
                placeholder for the list of failed requirements.
            **kwargs: Additional arguments passed to BaseOutputParser
        """
        # Extract validation functions and descriptions
        req_list = []
        desc_list = []

        for req in requirements or []:
            if hasattr(req, "__call__"):
                # It's a callable (simple_validate result)
                req_list.append(req)
                # Try to get description from the callable
                desc = getattr(req, "__doc__", None) or getattr(req, "description", "Requirement")
                desc_list.append(desc)
            else:
                raise ValueError(
                    f"Requirement must be a callable (use simple_validate). Got: {type(req)}"
                )

        # Initialize with Pydantic
        super().__init__(
            requirements=req_list,
            requirement_descriptions=desc_list,
            strict=strict,
            error_message_template=error_message_template
            or "Output validation failed. The following requirements were not met:\n{errors}",
            **kwargs,
        )

    def parse(self, text: str) -> str:
        """Parse and validate the output text.

        Args:
            text: The text to validate

        Returns:
            The validated text

        Raises:
            OutputParserException: If validation fails and strict=True
        """
        failed_requirements = []

        # Validate each requirement
        for req_fn, description in zip(self.requirements, self.requirement_descriptions):
            try:
                if not req_fn(text):
                    failed_requirements.append(description)
            except Exception as e:
                # If validation function raises an exception, treat as failure
                failed_requirements.append(f"{description} (validation error: {e})")

        # Handle validation results
        if failed_requirements:
            if self.strict:
                error_list = "\n".join(f"  - {req}" for req in failed_requirements)
                error_msg = self.error_message_template.format(errors=error_list)
                raise OutputParserException(
                    error=error_msg,
                    llm_output=text,
                    observation=(
                        f"Validation failed for {len(failed_requirements)} requirement(s). "
                        "Consider using OutputFixingParser for auto-repair."
                    ),
                )
            # Non-strict mode: return text anyway

        return text

    def get_format_instructions(self) -> str:
        """Get instructions for the LLM about requirements.

        Returns:
            String describing the requirements that the output should meet
        """
        if not self.requirement_descriptions:
            return "No specific requirements."

        instructions = "The output must meet the following requirements:\n"
        for i, desc in enumerate(self.requirement_descriptions, 1):
            instructions += f"{i}. {desc}\n"

        return instructions.strip()

    @property
    def _type(self) -> str:
        """Return the type key for this parser."""
        return "mellea_output_parser"


class MelleaGuardrail:
    """Guardrail component for validating LLM outputs.

    This is an independent validation component that can be used standalone
    or composed with other guardrails. It provides a flexible interface for
    requirement-based validation without being tied to LangChain chains.

    Example:
        ```python
        from mellea_langchain import MelleaGuardrail
        from mellea.stdlib.requirements import simple_validate

        # Create guardrail
        guardrail = MelleaGuardrail(
            requirements=[
                simple_validate(lambda x: len(x) > 50, "At least 50 chars"),
                simple_validate(lambda x: len(x) < 500, "Under 500 chars"),
            ],
            name="length_check"
        )

        # Validate output
        result = guardrail.validate(text)
        if not result.passed:
            print(f"Validation failed: {result.errors}")
        ```

    Composing guardrails:
        ```python
        tone_guard = MelleaGuardrail(
            requirements=[simple_validate(lambda x: x.islower(), "Lowercase only")]
        )
        length_guard = MelleaGuardrail(
            requirements=[simple_validate(lambda x: len(x) < 100, "Under 100 chars")]
        )

        # Compose using method or & operator
        combined = tone_guard.compose(length_guard)
        # or
        combined = tone_guard & length_guard
        ```
    """

    def __init__(
        self,
        requirements: list[Any],
        name: str | None = None,
    ):
        """Initialize the guardrail.

        Args:
            requirements: List of validation functions created using
                simple_validate(fn, description) from mellea.stdlib.requirements
            name: Optional name for this guardrail (useful for debugging)
        """
        self.name = name or "unnamed_guardrail"

        # Extract validation functions and descriptions
        self.requirements = []
        self.requirement_descriptions = []

        for req in requirements:
            if hasattr(req, "__call__"):
                self.requirements.append(req)
                desc = getattr(req, "__doc__", None) or getattr(req, "description", "Requirement")
                self.requirement_descriptions.append(desc)
            else:
                raise ValueError(
                    f"Requirement must be a callable (use simple_validate). Got: {type(req)}"
                )

    def validate(self, text: str) -> ValidationResult:
        """Validate text against requirements.

        Args:
            text: Text to validate

        Returns:
            ValidationResult with passed status, errors, and metadata
        """
        failed_requirements = []
        passed_requirements = []

        # Validate each requirement
        for req_fn, description in zip(self.requirements, self.requirement_descriptions):
            try:
                if req_fn(text):
                    passed_requirements.append(description)
                else:
                    failed_requirements.append(description)
            except Exception as e:
                # If validation function raises an exception, treat as failure
                failed_requirements.append(f"{description} (validation error: {e})")

        # Build result
        passed = len(failed_requirements) == 0
        metadata = {
            "guardrail_name": self.name,
            "total_requirements": len(self.requirements),
            "passed_requirements": len(passed_requirements),
            "failed_requirements": len(failed_requirements),
            "passed_requirement_list": passed_requirements,
        }

        return ValidationResult(
            passed=passed,
            text=text,
            errors=failed_requirements,
            metadata=metadata,
        )

    def compose(self, other: "MelleaGuardrail") -> "MelleaGuardrail":
        """Compose with another guardrail.

        Creates a new guardrail that combines the requirements from both
        guardrails. The new guardrail will validate against all requirements
        from both sources.

        Args:
            other: Another guardrail to combine with

        Returns:
            New guardrail with combined requirements
        """
        if not isinstance(other, MelleaGuardrail):
            raise TypeError(f"Can only compose with MelleaGuardrail, got {type(other)}")

        # Combine requirements and descriptions
        combined_requirements = self.requirements + other.requirements
        combined_descriptions = self.requirement_descriptions + other.requirement_descriptions

        # Create new guardrail
        new_guardrail = MelleaGuardrail(
            requirements=combined_requirements,
            name=f"{self.name}+{other.name}",
        )

        # Manually set descriptions since we already extracted them
        new_guardrail.requirement_descriptions = combined_descriptions

        return new_guardrail

    def __and__(self, other: "MelleaGuardrail") -> "MelleaGuardrail":
        """Allow composition with & operator.

        Example:
            combined = guardrail1 & guardrail2 & guardrail3
        """
        return self.compose(other)

    def __repr__(self) -> str:
        """String representation of the guardrail."""
        return f"MelleaGuardrail(name='{self.name}', requirements={len(self.requirements)})"


__all__ = [
    "MelleaOutputParser",
    "MelleaGuardrail",
    "ValidationResult",
]
