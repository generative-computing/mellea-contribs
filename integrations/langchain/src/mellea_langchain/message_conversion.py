"""Message conversion utilities between LangChain and Mellea formats."""

import logging
import re
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from mellea_integration import BaseMessageConverter

logger = logging.getLogger(__name__)

try:
    from mellea.core import ModelToolCall
    from mellea.stdlib.components import Message
except ImportError:
    # Fallback for type hints if mellea is not installed
    Message = Any  # type: ignore
    ModelToolCall = Any  # type: ignore


class LangChainMessageConverter(BaseMessageConverter):
    """Convert between LangChain and Mellea message formats.

    Extends BaseMessageConverter to provide LangChain-specific conversion logic.
    """

    def to_mellea(self, messages: list[BaseMessage]) -> list[Message]:
        """Convert LangChain messages to Mellea format.

        Args:
            messages: List of LangChain BaseMessage objects

        Returns:
            List of Mellea Message objects

        Mapping:
            - SystemMessage -> Message(role="system", content=...)
            - HumanMessage -> Message(role="user", content=...)
            - AIMessage -> Message(role="assistant", content=...)
            - ToolMessage -> Message(role="user", content=...) [Note: tool_call_id is lost]

        Known Limitations:
            - AIMessage tool_calls are not preserved during conversion (pending Mellea API support)
            - ToolMessage tool_call_id association is lost (converted to user message with prefix)
            - These limitations should be addressed when Mellea's tool message API is finalized
        """
        mellea_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                mellea_messages.append(
                    self.create_mellea_message("system", self.normalize_content(msg.content))
                )
            elif isinstance(msg, HumanMessage):
                mellea_messages.append(
                    self.create_mellea_message("user", self.normalize_content(msg.content))
                )
            elif isinstance(msg, AIMessage):
                # TODO: Preserve tool_calls when Mellea's Message API supports it
                # Currently, tool_calls in AIMessage are dropped during conversion.
                # This is a known limitation that should be addressed when:
                # 1. Mellea's Message class supports a tool_calls parameter, OR
                # 2. Mellea provides a separate ToolCallMessage type
                #
                # For now, we only preserve the text content of the message.
                # If the AIMessage has tool_calls but no content, this may result
                # in an empty message being sent to Mellea.
                content = self.normalize_content(msg.content) if msg.content else ""
                mellea_messages.append(self.create_mellea_message("assistant", content))
            elif isinstance(msg, ToolMessage):
                # TODO: Preserve tool_call_id when Mellea's Message API supports it
                # Currently, ToolMessages are converted to user messages with a prefix,
                # which loses the tool_call_id association. This is a known limitation.
                #
                # Ideal implementation would be:
                #   Message(role="tool", content=..., tool_call_id=msg.tool_call_id)
                #
                # This should be updated when Mellea supports:
                # 1. A "tool" role for messages, AND
                # 2. A tool_call_id parameter to associate results with calls
                #
                # For now, we use a simple prefix to indicate this is a tool result.
                content = f"Tool result: {self.normalize_content(msg.content)}"
                mellea_messages.append(self.create_mellea_message("user", content))
            else:
                # Fallback for unknown message types - treat as user message
                mellea_messages.append(
                    self.create_mellea_message("user", self.normalize_content(msg.content))
                )

        return mellea_messages

    def from_mellea(self, response: Any, **kwargs: Any) -> ChatResult:
        """Convert Mellea response to LangChain ChatResult.

        Args:
            response: Mellea ModelOutputThunk
            **kwargs: Additional metadata (generation_info, llm_output)

        Returns:
            LangChain ChatResult with AIMessage
        """
        # Extract content from response
        content = self.extract_content_from_response(response)

        # Extract tool calls if present
        tool_calls = self._extract_tool_calls(response)

        # Create AIMessage with or without tool_calls
        if tool_calls:
            message = AIMessage(content=content if content else "", tool_calls=tool_calls)
        else:
            message = AIMessage(content=content)

        # Create ChatGeneration with generation_info if provided
        generation = ChatGeneration(
            message=message, generation_info=kwargs.get("generation_info", {})
        )

        # Create ChatResult with llm_output if provided
        return ChatResult(generations=[generation], llm_output=kwargs.get("llm_output", {}))

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Extract tool calls from Mellea response.

        Args:
            response: Mellea response object

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []

        # Check if the content looks like tool calls (string representation)
        if hasattr(response, "content"):
            content_str = str(response.content)
            if content_str.startswith("[ToolCall"):
                # Parse tool calls from the string representation
                tool_calls = self._parse_tool_calls_from_string(content_str)

        # Also check for _tool_calls attribute (alternative format)
        if not tool_calls and hasattr(response, "_tool_calls") and response._tool_calls:
            tool_calls = [
                {"id": tc.id, "name": tc.name, "args": tc.arguments} for tc in response._tool_calls
            ]

        return tool_calls

    @staticmethod
    def _parse_tool_calls_from_string(content_str: str) -> list[dict[str, Any]]:
        """Parse tool calls from Mellea's string representation.

        Example input: "[ToolCall(function=Function(name='get_weather', arguments={'location': 'NYC'}))]"

        Args:
            content_str: String containing tool call representations

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []

        # Pattern to match ToolCall objects in the string
        # Matches: ToolCall(function=Function(name='tool_name', arguments={...}))
        pattern = r"ToolCall\(function=Function\(name='([^']+)',\s*arguments=(\{[^}]+\})\)\)"

        for match in re.finditer(pattern, content_str):
            tool_name = match.group(1)
            args_str = match.group(2)

            try:
                # Safely evaluate the arguments dictionary
                import ast

                args_dict = ast.literal_eval(args_str)

                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls)}",  # Generate a simple ID
                        "name": tool_name,
                        "args": args_dict,
                    }
                )
            except (ValueError, SyntaxError) as e:
                # If parsing fails, skip this tool call
                logger.warning("Failed to parse tool call arguments: %s", e)
                continue

        return tool_calls
