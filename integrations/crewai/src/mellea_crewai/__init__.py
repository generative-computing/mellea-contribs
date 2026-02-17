"""Mellea LLM for CrewAI.

This package provides a CrewAI-compatible LLM that wraps Mellea,
enabling CrewAI applications to use Mellea's generative programming
capabilities including requirements, validation, and sampling strategies.
"""

from .llm import MelleaLLM
from .message_conversion import (
    CrewAIMessageConverter,
    crewai_to_mellea_messages,
    mellea_to_crewai_response,
)
from .tool_conversion import CrewAIToolConverter, crewai_to_mellea_tools

__version__ = "0.1.0"

__all__ = [
    "MelleaLLM",
    "CrewAIMessageConverter",
    "CrewAIToolConverter",
    "crewai_to_mellea_messages",
    "mellea_to_crewai_response",
    "crewai_to_mellea_tools",
]
