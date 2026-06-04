"""Mellea LLM for CrewAI.

This subpackage provides a CrewAI-compatible LLM that wraps Mellea, enabling
CrewAI applications to use Mellea's generative programming capabilities
including requirements, validation, and sampling strategies. It also provides
helpers that convert Mellea requirements into CrewAI-compatible guardrail
functions for task output validation.
"""

from .core.llm import MelleaLLM
from .core.message_conversion import CrewAIMessageConverter
from .core.tool_conversion import CrewAIToolConverter
from .core.validators import create_guardrail, create_guardrails

__version__ = "0.1.0"

__all__ = [
    "MelleaLLM",
    "CrewAIMessageConverter",
    "CrewAIToolConverter",
    "create_guardrail",
    "create_guardrails",
]
