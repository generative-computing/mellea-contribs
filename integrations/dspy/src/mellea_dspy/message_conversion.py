"""Message conversion utilities for DSPy integration."""

from typing import Any

from mellea_integration import BaseMessageConverter


class DSPyMessageConverter(BaseMessageConverter):
    """Convert between DSPy and Mellea message formats.

    DSPy uses a simple dictionary format for messages with 'role' and 'content' keys.
    This converter handles the conversion to/from Mellea's Message format.
    """

    def to_mellea(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Convert DSPy messages to Mellea format.

        Args:
            messages: List of DSPy message dictionaries with 'role' and 'content'

        Returns:
            List of Mellea Message objects
        """
        mellea_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = self.normalize_content(msg.get("content", ""))
            mellea_messages.append(self.create_mellea_message(role, content))
        return mellea_messages

    def from_mellea(self, response: Any) -> Any:
        """Convert Mellea response to DSPy-compatible format.

        DSPy expects an OpenAI-compatible response object with specific structure.
        This method creates a mock response that matches OpenAI's format.

        Args:
            response: Mellea response object (ModelOutputThunk or SamplingResult)

        Returns:
            OpenAI-compatible response object for DSPy
        """
        from types import SimpleNamespace

        # Extract content from Mellea response
        content = self.extract_content_from_response(response)

        # Create a mock response that matches OpenAI's structure
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content, role="assistant"),
            finish_reason="stop",
            index=0,
        )

        # Usage must be a dict-like object that can be converted with dict()
        # DSPy calls dict(response.usage) in base_lm.py line 71
        class UsageDict(dict):
            """Dict subclass that also supports attribute access."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__ = self

        usage = UsageDict(
            prompt_tokens=0,  # Mellea doesn't provide token counts
            completion_tokens=0,
            total_tokens=0,
        )

        response_obj = SimpleNamespace(
            id="mellea-" + str(hash(content))[:8],
            object="chat.completion",
            created=0,
            model="mellea",  # Will be overridden by the LM class
            choices=[choice],
            usage=usage,
        )

        return response_obj


# Made with Bob
