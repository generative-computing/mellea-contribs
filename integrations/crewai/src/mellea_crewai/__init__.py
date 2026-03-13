"""Mellea LLM for CrewAI.

This package provides a CrewAI-compatible LLM that wraps Mellea,
enabling CrewAI applications to use Mellea's generative programming
capabilities including requirements, validation, and sampling strategies.

It also provides validators that convert Mellea requirements into
CrewAI-compatible guardrail functions for task output validation.
"""

from .llm import MelleaLLM
from .message_conversion import CrewAIMessageConverter
from .tool_conversion import CrewAIToolConverter
from .validators import (
    create_guardrail,
    create_guardrails,
)

__version__ = "0.1.0"

__all__ = [
    "MelleaLLM",
    "CrewAIMessageConverter",
    "CrewAIToolConverter",
    # Validators
    "create_guardrail",
    "create_guardrails",
]
