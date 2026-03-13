"""Core abstractions for Mellea framework integrations.

This package provides base classes and utilities for creating clean,
maintainable integrations between Mellea and various AI frameworks
(LangChain, CrewAI, DSPy, etc.).
"""

from .base import MelleaIntegrationBase
from .message_converter import BaseMessageConverter
from .tool_converter import BaseToolConverter
from .types import GenerationResult, MessageConverter, ModelOptions, ToolConverter

__version__ = "0.1.0"

__all__ = [
    "MelleaIntegrationBase",
    "BaseMessageConverter",
    "MessageConverter",
    "BaseToolConverter",
    "ToolConverter",
    "GenerationResult",
    "ModelOptions",
]
