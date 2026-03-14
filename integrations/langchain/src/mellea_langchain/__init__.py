"""Mellea Chat Model for LangChain.

This package provides a LangChain-compatible chat model that wraps Mellea,
enabling LangChain applications to use Mellea's generative programming
capabilities through the standard LangChain interface.
"""

from .chat_model import MelleaChatModel
from .message_conversion import LangChainMessageConverter
from .tool_conversion import LangChainToolConverter

__version__ = "0.1.0"

__all__ = [
    "MelleaChatModel",
    "LangChainMessageConverter",
    "LangChainToolConverter",
]
