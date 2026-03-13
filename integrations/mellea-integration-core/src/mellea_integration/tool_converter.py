"""Tool conversion utilities for Mellea integrations."""

import ast
import logging
import re
from typing import Any

try:
    from mellea.backends.tools import MelleaTool
except ImportError:
    # Fallback for type hints if mellea is not installed
    MelleaTool = Any  # type: ignore

logger = logging.getLogger(__name__)


class BaseToolConverter:
    """Base tool converter with common patterns.

    Provides shared functionality for converting between framework-specific
    tool formats and Mellea's MelleaTool format.
    """

    @staticmethod
    def extract_tool_schema(tool: Any) -> dict[str, Any]:
        """Extract JSON schema from a tool object.

        Args:
            tool: Framework-specific tool object

        Returns:
            JSON schema dictionary in OpenAI function calling format
        """
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": getattr(tool, "name", "unknown_tool"),
                "description": getattr(tool, "description", ""),
                "parameters": {},
            },
        }

        # Try to extract parameters schema
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                schema["function"]["parameters"] = tool.args_schema.schema()
            except Exception as e:
                # If schema extraction fails, use empty dict
                logger.warning(
                    "Failed to extract schema from tool '%s': %s",
                    schema["function"]["name"],
                    str(e),
                )

        return schema

    @staticmethod
    def get_tool_callable(tool: Any) -> Any:
        """Extract callable function from various tool formats.

        Tries multiple common attributes to find the executable function.

        Args:
            tool: Framework-specific tool object

        Returns:
            Callable function

        Raises:
            ValueError: If no callable can be extracted
        """
        # Try different attributes in order of preference
        for attr in ["func", "_run", "run", "__call__"]:
            if hasattr(tool, attr):
                func = getattr(tool, attr)
                if callable(func):
                    return func

        # If the tool itself is callable
        if callable(tool):
            return tool

        raise ValueError(f"Cannot extract callable from tool: {tool}")

    @staticmethod
    def _extract_balanced_braces(text: str, start_pos: int) -> str | None:
        """Extract a balanced dictionary string starting at start_pos.

        Uses brace counting to find the matching closing brace, handles nested dicts.

        Args:
            text: Full text to search in
            start_pos: Position of the opening brace

        Returns:
            The balanced dictionary string, or None if not found
        """
        if start_pos >= len(text) or text[start_pos] != "{":
            return None

        brace_count = 0
        in_string = False
        escape_next = False
        pos = start_pos

        while pos < len(text):
            char = text[pos]

            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_pos : pos + 1]

            pos += 1

        return None

    @staticmethod
    def parse_tool_calls_from_string(content_str: str) -> list[dict[str, Any]]:
        """Parse tool calls from Mellea's string representation.

        Some models return tool calls as string representations that need parsing.

        Args:
            content_str: String that may contain tool call representations
                        Example: "[ToolCall(function=Function(name='get_weather', arguments={'location': 'NYC'}))]"

        Returns:
            List of tool call dictionaries with 'id', 'name', and 'args'
        """
        tool_calls = []

        # Pattern to extract just the tool name from ToolCall
        # Matches: ToolCall(function=Function(name='tool_name', ...
        name_pattern = r"ToolCall\(function=Function\(name='([^']+)',\s*arguments="

        for match in re.finditer(name_pattern, content_str):
            tool_name = match.group(1)
            args_start = match.end()

            # Find the opening brace for arguments
            args_brace_pos = content_str.find("{", args_start)
            if args_brace_pos == -1:
                continue

            # Use brace counting to extract the complete dictionary
            args_str = BaseToolConverter._extract_balanced_braces(content_str, args_brace_pos)
            if not args_str:
                continue

            try:
                # Safely evaluate the arguments dictionary
                args_dict = ast.literal_eval(args_str)

                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls)}",  # Generate a simple ID
                        "name": tool_name,
                        "args": args_dict,
                    }
                )
            except (ValueError, SyntaxError):
                # If parsing fails, skip this tool call
                continue

        return tool_calls

    @staticmethod
    def extract_tool_calls_from_response(response: Any) -> list[dict[str, Any]]:
        """Extract tool calls from Mellea response.

        Handles multiple formats:
        - Direct tool_calls attribute
        - _tool_calls attribute
        - String representation that needs parsing

        Args:
            response: Mellea response object

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []

        # Check for direct tool_calls attribute
        if hasattr(response, "tool_calls") and response.tool_calls:
            if isinstance(response.tool_calls, list) and len(response.tool_calls) > 0:
                # Convert to standard format
                for tc in response.tool_calls:
                    tool_calls.append(
                        {
                            "id": getattr(tc, "id", f"call_{len(tool_calls)}"),
                            "name": getattr(tc, "name", "unknown"),
                            "args": getattr(tc, "arguments", {}),
                        }
                    )

        # Check for _tool_calls attribute (alternative format)
        elif hasattr(response, "_tool_calls") and response._tool_calls:
            if isinstance(response._tool_calls, list) and len(response._tool_calls) > 0:
                for tc in response._tool_calls:
                    tool_calls.append(
                        {
                            "id": getattr(tc, "id", f"call_{len(tool_calls)}"),
                            "name": getattr(tc, "name", "unknown"),
                            "args": getattr(tc, "arguments", {}),
                        }
                    )

        # Check if content is a string representation of tool calls
        elif hasattr(response, "content"):
            content_str = str(response.content)
            if content_str.startswith("[ToolCall"):
                tool_calls = BaseToolConverter.parse_tool_calls_from_string(content_str)

        return tool_calls

    def to_mellea(self, tools: list[Any]) -> list[Any]:
        """Convert framework tools to Mellea format.

        This method should be overridden by framework-specific converters.

        Args:
            tools: Framework-specific tool objects

        Returns:
            List of Mellea MelleaTool objects

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement to_mellea()")

    def parse_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Parse tool calls from Mellea response.

        Default implementation uses extract_tool_calls_from_response.
        Can be overridden for framework-specific parsing.

        Args:
            response: Mellea response that may contain tool calls

        Returns:
            List of tool call dictionaries with 'id', 'name', and 'args'
        """
        return self.extract_tool_calls_from_response(response)
