"""Message conversion utilities between CrewAI and Mellea formats."""

import ast
import re
from typing import Any

try:
    from mellea.stdlib.components import Message
except ImportError:
    # Fallback for type hints if mellea is not installed
    Message = Any  # type: ignore


def parse_tool_calls_from_text(text: str) -> list[Any] | None:
    """Parse tool calls from text that looks like tool call objects.

    WORKAROUND: Some models generate text that looks like tool calls instead of
    actual tool call objects. This function attempts to parse such text.

    Args:
        text: String that might contain tool call representations

    Returns:
        List of parsed tool call objects, or None if parsing fails
    """
    # Check if text looks like a list of ToolCall objects
    if not (text.strip().startswith("[") and "ToolCall" in text and "function=" in text):
        return None

    try:
        # Try to extract tool call information using regex
        # Pattern: ToolCall(function=Function(name='tool_name', arguments={...}))
        pattern = r"ToolCall\(function=Function\(name='([^']+)',\s*arguments=(\{[^}]+\})\)\)"
        matches = re.findall(pattern, text)

        if not matches:
            return None

        # Create tool call objects
        tool_calls = []
        for name, args_str in matches:
            try:
                # Parse arguments
                arguments = ast.literal_eval(args_str)

                # Create a simple tool call object
                class ToolCall:
                    def __init__(self, name: str, arguments: dict[str, Any]):
                        self.function = type(
                            "Function", (), {"name": name, "arguments": arguments}
                        )()

                tool_calls.append(ToolCall(name, arguments))
            except Exception:
                continue

        return tool_calls if tool_calls else None
    except Exception:
        return None


def crewai_to_mellea_messages(messages: str | list[dict[str, Any]]) -> list[Message]:
    """Convert CrewAI messages to Mellea format.

    Args:
        messages: CrewAI message format (string or list of message dicts)
                 Each dict should have 'role' and 'content' keys.

    Returns:
        List of Mellea Message objects

    Example:
        >>> # String input
        >>> crewai_to_mellea_messages("Hello, world!")
        [Message(role="user", content="Hello, world!")]

        >>> # List of dicts
        >>> crewai_to_mellea_messages([
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Hello"}
        ... ])
        [Message(role="system", content="You are helpful"),
         Message(role="user", content="Hello")]
    """
    if isinstance(messages, str):
        return [Message(role="user", content=messages)]

    mellea_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle different message types
        if role in ["system", "user", "assistant"]:
            mellea_messages.append(Message(role=role, content=str(content)))
        elif role == "tool":
            # Handle tool messages
            # Note: CrewAI may include tool_call_id, but Mellea's Message
            # may not support it yet. We preserve the content.
            mellea_messages.append(Message(role="tool", content=str(content)))
        else:
            # Fallback to user message for unknown roles
            mellea_messages.append(Message(role="user", content=str(content)))

    return mellea_messages


def mellea_to_crewai_response(response: Any) -> str | list[Any]:
    """Convert Mellea response to CrewAI format.

    Args:
        response: Mellea ModelOutputThunk or similar response object

    Returns:
        String content from the response, or list of tool calls if present

    Example:
        >>> response = session.chat("Hello")
        >>> mellea_to_crewai_response(response)
        "Hello! How can I help you today?"
    """
    # Extract content from Mellea response
    if hasattr(response, "content"):
        content = response.content
        # Check if content is a list (likely tool calls)
        if isinstance(content, list):
            return content
        # WORKAROUND: Try to parse tool calls from text
        # Some models generate text that looks like tool calls
        if isinstance(content, str):
            parsed_tool_calls = parse_tool_calls_from_text(content)
            if parsed_tool_calls:
                return parsed_tool_calls
        return str(content) if content is not None else ""
    elif hasattr(response, "value"):
        value = response.value
        # Check if value is a list (likely tool calls)
        if isinstance(value, list):
            return value
        # WORKAROUND: Try to parse tool calls from text
        if isinstance(value, str):
            parsed_tool_calls = parse_tool_calls_from_text(value)
            if parsed_tool_calls:
                return parsed_tool_calls
        return str(value) if value is not None else ""
    else:
        # Fallback: convert to string
        return str(response)
