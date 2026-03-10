"""Mellea Chat Model for LangChain.

This package provides a LangChain-compatible chat model that wraps Mellea,
enabling LangChain applications to use Mellea's generative programming
capabilities through the standard LangChain interface.
<<<<<<< HEAD
"""

from .chat_model import MelleaChatModel
=======

It also provides output parsers and guardrails for validating LLM outputs
using Mellea's requirements system.
"""

from .chat_model import MelleaChatModel
from .guardrails import MelleaGuardrail, MelleaOutputParser, ValidationResult
>>>>>>> aaec38e (adding missing changes)
from .message_conversion import LangChainMessageConverter
from .tool_conversion import LangChainToolConverter

__version__ = "0.1.0"

__all__ = [
    "MelleaChatModel",
    "MelleaGuardrail",
    "MelleaOutputParser",
    "ValidationResult",
    "LangChainMessageConverter",
    "LangChainToolConverter",
]
