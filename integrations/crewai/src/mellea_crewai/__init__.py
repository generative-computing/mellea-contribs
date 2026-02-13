"""Mellea LLM for CrewAI.

This package provides a CrewAI-compatible LLM that wraps Mellea,
enabling CrewAI applications to use Mellea's generative programming
capabilities including requirements, validation, and sampling strategies.
"""

from .llm import MelleaLLM
from .message_conversion import crewai_to_mellea_messages
from .tool_conversion import crewai_to_mellea_tools

__version__ = "0.1.0"

__all__ = [
    "MelleaLLM",
    "crewai_to_mellea_messages",
    "crewai_to_mellea_tools",
]
