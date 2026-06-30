"""Mellea Chat Model for LangChain.

This package provides a LangChain-compatible chat model that wraps Mellea,
enabling LangChain applications to use Mellea's generative programming
capabilities through the standard LangChain interface.

It also provides output parsers and guardrails for validating LLM outputs
using Mellea's requirements system.
"""

from .core.chat_model import MelleaChatModel
from .core.guardrails import MelleaGuardrail, MelleaOutputParser, ValidationResult
from .core.message_conversion import LangChainMessageConverter
from .core.tool_conversion import LangChainToolConverter

__version__ = "0.1.0"

__all__ = [
    "MelleaChatModel",
    "MelleaGuardrail",
    "MelleaOutputParser",
    "ValidationResult",
    "LangChainMessageConverter",
    "LangChainToolConverter",
]
