"""Common types and protocols for Mellea integrations."""

from typing import Any, Protocol, TypedDict


class ModelOptions(TypedDict, total=False):
    """Type definition for Mellea model options."""

    temperature: float
    max_tokens: int
    top_p: float
    tools: list[Any]


class GenerationResult(TypedDict):
    """Standard result format for generation operations."""

    content: str
    tool_calls: list[dict[str, Any]] | None
    metadata: dict[str, Any]


class MessageConverter(Protocol):
    """Protocol for message conversion between frameworks and Mellea."""

    def to_mellea(self, messages: Any) -> list[Any]:
        """Convert framework messages to Mellea Message format.

        Args:
            messages: Framework-specific message format

        Returns:
            List of Mellea Message objects
        """
        ...

    def from_mellea(self, response: Any) -> Any:
        """Convert Mellea response to framework format.

        Args:
            response: Mellea ModelOutputThunk or similar

        Returns:
            Framework-specific response format
        """
        ...


class ToolConverter(Protocol):
    """Protocol for tool conversion between frameworks and Mellea."""

    def to_mellea(self, tools: list[Any]) -> list[Any]:
        """Convert framework tools to Mellea MelleaTool format.

        Args:
            tools: Framework-specific tool objects

        Returns:
            List of Mellea MelleaTool objects
        """
        ...

    def parse_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Parse tool calls from Mellea response.

        Args:
            response: Mellea response that may contain tool calls

        Returns:
            List of tool call dictionaries with 'id', 'name', and 'args'
        """
        ...


