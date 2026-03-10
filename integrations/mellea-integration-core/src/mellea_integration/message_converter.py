"""Message conversion utilities for Mellea integrations."""

from typing import Any

try:
    from mellea.stdlib.components import Message
except ImportError:
    # Fallback for type hints if mellea is not installed
    Message = Any  # type: ignore


class BaseMessageConverter:
    """Base message converter with common utilities.

    Provides shared functionality for converting between framework-specific
    message formats and Mellea's Message format.
    """

    @staticmethod
    def extract_last_user_message(messages: list[Any]) -> str:
        """Extract the last user message content from a list of messages.

        Args:
            messages: List of Mellea Message objects

        Returns:
            Content string from the last user message

        Raises:
            ValueError: If no user message is found
        """
        for msg in reversed(messages):
            if hasattr(msg, "role") and msg.role == "user":
                return str(msg.content)
        raise ValueError("No user message found in message list")

    @staticmethod
    def create_mellea_message(role: str, content: str) -> Any:
        """Create a Mellea message with role validation.

        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content

        Returns:
            Mellea Message object
        """
        # Validate and normalize role
        valid_roles = ["system", "user", "assistant", "tool"]
        if role not in valid_roles:
            role = "user"  # Default fallback

        return Message(role=role, content=content)

    @staticmethod
    def normalize_content(content: Any) -> str:
        """Normalize message content to string format.

        Args:
            content: Message content (may be string, list, dict, etc.)

        Returns:
            Normalized string content
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (list, dict)):
            # For complex content, convert to string representation
            return str(content)
        return str(content)

    def to_mellea(self, messages: Any) -> list[Any]:
        """Convert framework messages to Mellea format.

        This method should be overridden by framework-specific converters.

        Args:
            messages: Framework-specific message format

        Returns:
            List of Mellea Message objects

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement to_mellea()")

    def from_mellea(self, response: Any) -> Any:
        """Convert Mellea response to framework format.

        This method should be overridden by framework-specific converters.

        Args:
            response: Mellea ModelOutputThunk or similar

        Returns:
            Framework-specific response format

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement from_mellea()")

    @staticmethod
    def extract_content_from_response(response: Any) -> str:
        """Extract text content from Mellea response.

        Args:
            response: Mellea response object

        Returns:
            Text content as string
        """
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            return str(content) if content is not None else ""
        elif hasattr(response, "value"):
            value = response.value
            if isinstance(value, str):
                return value
            return str(value) if value is not None else ""
        else:
            # If response has no content or value attribute, return empty string
            # rather than string representation of the object
            return ""


