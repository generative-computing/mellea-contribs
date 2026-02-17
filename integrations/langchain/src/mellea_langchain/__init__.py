"""Mellea Chat Model for LangChain.

This package provides a LangChain-compatible chat model that wraps Mellea,
enabling LangChain applications to use Mellea's generative programming
capabilities through the standard LangChain interface.
"""

from .chat_model import MelleaChatModel
from .message_conversion import (
    LangChainMessageConverter,
    langchain_to_mellea_messages,
    mellea_to_langchain_result,
)
from .tool_conversion import LangChainToolConverter, langchain_to_mellea_tools

__version__ = "0.1.0"

__all__ = [
    "MelleaChatModel",
    "LangChainMessageConverter",
    "LangChainToolConverter",
    "langchain_to_mellea_messages",
    "langchain_to_mellea_tools",
    "mellea_to_langchain_result",
]

# Made with Bob
