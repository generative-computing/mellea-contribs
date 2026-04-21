"""Validators for CrewAI task guardrails using Mellea requirements.

This module provides wrapper functions that convert Mellea requirements
into CrewAI-compatible guardrail functions.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# Set up logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from crewai import TaskOutput
else:
    try:
        from crewai import TaskOutput
    except ImportError:
        # Fallback if crewai is not installed
        TaskOutput = Any  # type: ignore


def create_guardrail(
    requirement: Any,
    error_prefix: str = "Validation failed",
) -> Callable[[TaskOutput], tuple[bool, Any]]:
    """Create a CrewAI guardrail function from a Mellea requirement.

    This function wraps a Mellea requirement (created with req(), check(),
    or simple_validate()) into a CrewAI-compatible guardrail function.

    Args:
        requirement: A Mellea requirement function. Can be:
            - simple_validate(fn, description) - deterministic validation
            - req(description) - LLM-based validation (requires session)
            - check(description) - stricter LLM-based validation
        error_prefix: Prefix for error messages (default: "Validation failed")

    Returns:
        A guardrail function with signature (TaskOutput) -> Tuple[bool, Any]

    Example:
        ```python
        from mellea.stdlib.requirements import simple_validate
        from mellea_crewai import create_guardrail

        # Create a Mellea requirement
        word_count_req = simple_validate(
            lambda x: 100 <= len(x.split()) <= 500,
            "Must be between 100-500 words"
        )

        # Convert to CrewAI guardrail
        word_count_guardrail = create_guardrail(word_count_req)

        # Use in task
        task = Task(
            description="Write a blog post",
            expected_output="A blog post",
            agent=agent,
            guardrails=[word_count_guardrail],
            guardrail_max_retries=3
        )
        ```
    """
    # Get requirement description
    description = (
        getattr(requirement, "description", None)
        or getattr(requirement, "__doc__", None)
        or "Requirement"
    )

    def guardrail(result: TaskOutput) -> tuple[bool, Any]:
        """Validate task output against requirement."""
        try:
            # Extract text from TaskOutput
            text = result.raw if hasattr(result, "raw") else str(result)

            # Log validation attempt
            text_preview = text[:100] + "..." if len(text) > 100 else text
            logger.info(f"Validating output against: {description}")
            logger.debug(f"Output preview: {text_preview}")

            # Validate using the requirement
            if callable(requirement):
                passed = requirement(text)
            else:
                # Handle other requirement types
                passed = False

            if passed:
                logger.info(f"✓ Validation passed: {description}")
                return (True, text)
            else:
                error_msg = f"{error_prefix}: {description}"
                logger.warning(f"✗ Validation failed: {description}")
                logger.debug(f"Failed output: {text_preview}")
                return (False, error_msg)

        except Exception as e:
            error_msg = f"{error_prefix}: {description} (validation error: {e})"
            logger.error(f"✗ Validation error: {description} - {e}")
            return (False, error_msg)

    # Set function name and docstring for better debugging
    safe_name = description.replace(" ", "_").replace("-", "_")[:30]
    guardrail.__name__ = f"guardrail_{safe_name}"
    guardrail.__doc__ = f"Guardrail for: {description}"

    return guardrail


def create_guardrails(
    requirements: list[Any],
    error_prefix: str = "Validation failed",
) -> list[Callable[[TaskOutput], tuple[bool, Any]]]:
    """Create multiple CrewAI guardrail functions from Mellea requirements.

    This is a convenience function that converts a list of Mellea requirements
    into a list of CrewAI-compatible guardrail functions.

    Args:
        requirements: List of Mellea requirement functions
        error_prefix: Prefix for error messages

    Returns:
        List of guardrail functions

    Example:
        ```python
        from mellea.stdlib.requirements import simple_validate
        from mellea_crewai import create_guardrails

        requirements = [
            simple_validate(lambda x: len(x) > 100, "At least 100 chars"),
            simple_validate(lambda x: "AI" in x, "Must mention AI"),
            simple_validate(lambda x: x.strip() == x, "No extra whitespace"),
        ]

        guardrails = create_guardrails(requirements)

        task = Task(
            description="Write about AI",
            expected_output="AI article",
            agent=agent,
            guardrails=guardrails,
            guardrail_max_retries=3
        )
        ```
    """
    return [create_guardrail(req, error_prefix) for req in requirements]
