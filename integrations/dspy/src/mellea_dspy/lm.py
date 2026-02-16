"""DSPy LM integration for Mellea.

This module provides a DSPy-compatible LM class that wraps Mellea,
enabling DSPy applications to use Mellea's generative programming
capabilities through the standard DSPy interface.
"""

from typing import Any, Optional

import dspy
from mellea import MelleaSession


class MelleaLM(dspy.BaseLM):
    """DSPy LM that uses Mellea as the backend.

    This allows DSPy applications to use Mellea's generative programming
    capabilities, including requirements validation, sampling strategies,
    and multi-model support through the standard DSPy interface.

    Note: The 'model' parameter in the constructor is used only for metadata
    in response objects. The actual model used for generation is determined
    by the mellea_session configuration.

    Example:
        ```python
        from mellea import start_session
        from mellea_dspy import MelleaLM
        import dspy

        # Create Mellea session
        m = start_session()

        # Create DSPy LM
        lm = MelleaLM(mellea_session=m, model="mellea-ollama")

        # Configure DSPy to use Mellea
        dspy.configure(lm=lm)

        # Use with DSPy
        print(dspy.Predict("question -> answer")(question="What is Mellea?"))
        ```

    Args:
        mellea_session: Configured Mellea session
        model: Model identifier (default: "mellea")
        temperature: Temperature for generation (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 1000)
        **kwargs: Additional parameters passed to Mellea
    """

    def __init__(
        self,
        mellea_session: MelleaSession,
        model: str = "mellea",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ):
        """Initialize the Mellea LM.

        Args:
            mellea_session: Configured Mellea session
            model: Model identifier (NOTE: This is only used for metadata in responses.
                   The actual model used for generation is determined by the mellea_session
                   configuration, not this parameter.)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.mellea_session = mellea_session
        self.provider = "mellea"

    def _convert_messages_to_mellea_format(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert DSPy messages to Mellea format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Tuple of (prompt_string, mellea_messages)
        """
        if not messages:
            return None, []

        # Extract the last user message as the prompt
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break

        return last_user_message, messages

    def _create_openai_compatible_response(self, content: str, model: str) -> Any:
        """Create an OpenAI-compatible response object.

        Args:
            content: The generated text content
            model: The model identifier

        Returns:
            Mock response object compatible with OpenAI format
        """
        from types import SimpleNamespace

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

        response = SimpleNamespace(
            id="mellea-" + str(hash(content))[:8],
            object="chat.completion",
            created=0,
            model=model,
            choices=[choice],
            usage=usage,
        )

        return response

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass for the language model.

        This method implements the DSPy LM interface and returns a response
        in OpenAI-compatible format.

        Args:
            prompt: Optional prompt string (used if messages not provided)
            messages: Optional list of message dictionaries
            **kwargs: Additional generation parameters including:
                - requirements: List of requirement strings for validation
                - strategy: Sampling strategy for validated generation
                - model_options: Additional Mellea model options

        Returns:
            OpenAI-compatible response object
        """
        # Determine what to send to Mellea
        if messages:
            prompt_text, _ = self._convert_messages_to_mellea_format(messages)
        else:
            prompt_text = prompt

        if not prompt_text:
            raise ValueError("Either prompt or messages must be provided")

        # Extract Mellea-specific parameters
        requirements = kwargs.pop("requirements", None)
        strategy = kwargs.pop("strategy", None)
        model_options = kwargs.pop("model_options", {})

        # Merge kwargs into model_options (temperature, max_tokens, etc.)
        merged_options = {**self.kwargs, **model_options}

        # Remove DSPy-specific parameters that Mellea doesn't need
        merged_options.pop("cache", None)
        merged_options.pop("model_type", None)

        # Use instruct method if requirements or strategy are provided
        if requirements is not None or strategy is not None:
            response = self.mellea_session.instruct(
                prompt_text,
                requirements=requirements,
                strategy=strategy,
                model_options=merged_options,
            )
        else:
            # Use standard chat method
            response = self.mellea_session.chat(
                prompt_text, model_options=merged_options
            )

        # Extract content from Mellea response
        content = response.content if hasattr(response, "content") else str(response)

        # Create OpenAI-compatible response
        return self._create_openai_compatible_response(content, self.model)

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Async forward pass for the language model.

        This method implements the async DSPy LM interface and returns a response
        in OpenAI-compatible format.

        Args:
            prompt: Optional prompt string (used if messages not provided)
            messages: Optional list of message dictionaries
            **kwargs: Additional generation parameters including:
                - requirements: List of requirement strings for validation
                - strategy: Sampling strategy for validated generation
                - model_options: Additional Mellea model options

        Returns:
            OpenAI-compatible response object
        """
        # Determine what to send to Mellea
        if messages:
            prompt_text, _ = self._convert_messages_to_mellea_format(messages)
        else:
            prompt_text = prompt

        if not prompt_text:
            raise ValueError("Either prompt or messages must be provided")

        # Extract Mellea-specific parameters
        requirements = kwargs.pop("requirements", None)
        strategy = kwargs.pop("strategy", None)
        model_options = kwargs.pop("model_options", {})

        # Merge kwargs into model_options (temperature, max_tokens, etc.)
        merged_options = {**self.kwargs, **model_options}

        # Remove DSPy-specific parameters that Mellea doesn't need
        merged_options.pop("cache", None)
        merged_options.pop("model_type", None)

        # Use ainstruct method if requirements or strategy are provided
        if requirements is not None or strategy is not None:
            response = await self.mellea_session.ainstruct(
                prompt_text,
                requirements=requirements,
                strategy=strategy,
                model_options=merged_options,
            )
        else:
            # Use standard async chat method
            response = await self.mellea_session.achat(
                prompt_text, model_options=merged_options
            )

        # Extract content from Mellea response
        content = response.content if hasattr(response, "content") else str(response)

        # Create OpenAI-compatible response
        return self._create_openai_compatible_response(content, self.model)
