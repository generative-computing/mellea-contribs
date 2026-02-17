"""DSPy LM integration for Mellea.

This module provides a DSPy-compatible LM class that wraps Mellea,
enabling DSPy applications to use Mellea's generative programming
capabilities through the standard DSPy interface.
"""

from typing import Any

import dspy
from mellea import MelleaSession
from mellea_integration import MelleaIntegrationBase

from .message_conversion import DSPyMessageConverter


class MelleaLM(dspy.BaseLM, MelleaIntegrationBase):
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
        requirements: Optional list of requirements for validation
        strategy: Optional sampling strategy for validation
        **kwargs: Additional parameters passed to Mellea
    """

    def __init__(
        self,
        mellea_session: MelleaSession,
        model: str = "mellea",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        requirements: list[Any] | None = None,
        strategy: Any | None = None,
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
            requirements: Optional list of requirements for validation
            strategy: Optional sampling strategy for validation
            **kwargs: Additional parameters
        """
        # Initialize DSPy BaseLM
        dspy.BaseLM.__init__(
            self,
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Initialize MelleaIntegrationBase
        MelleaIntegrationBase.__init__(
            self,
            mellea_session=mellea_session,
            message_converter=DSPyMessageConverter(),
            requirements=requirements,
            strategy=strategy,
            **kwargs,
        )

        self.provider = "mellea"

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Generate response using Mellea (sync version).

        This is a helper method that wraps the forward() method for
        compatibility with the integration base pattern.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional generation parameters

        Returns:
            OpenAI-compatible response object
        """
        return self.forward(messages=messages, **kwargs)

    async def agenerate(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Generate response using Mellea (async version).

        This is a helper method that wraps the aforward() method for
        compatibility with the integration base pattern.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional generation parameters

        Returns:
            OpenAI-compatible response object
        """
        return await self.aforward(messages=messages, **kwargs)

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
            prompt_text, model_options, _ = self._prepare_generation(
                messages, None, **kwargs
            )
        else:
            prompt_text = prompt
            model_options = kwargs.get("model_options", {})

        if not prompt_text:
            raise ValueError("Either prompt or messages must be provided")

        # Extract Mellea-specific parameters
        requirements = kwargs.get("requirements")
        strategy = kwargs.get("strategy")

        # Merge kwargs into model_options (temperature, max_tokens, etc.)
        merged_options = {**self.kwargs, **model_options}

        # Remove DSPy-specific parameters that Mellea doesn't need
        merged_options.pop("cache", None)
        merged_options.pop("model_type", None)

        # Generate with Mellea using base class method
        response = self._generate_with_mellea(
            prompt_text,
            merged_options,
            tool_calls_enabled=False,
            requirements=requirements,
            strategy=strategy,
        )

        # Convert response using message converter
        openai_response = self.message_converter.from_mellea(response)

        # Update model name in response
        openai_response.model = self.model

        return openai_response

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
            prompt_text, model_options, _ = self._prepare_generation(
                messages, None, **kwargs
            )
        else:
            prompt_text = prompt
            model_options = kwargs.get("model_options", {})

        if not prompt_text:
            raise ValueError("Either prompt or messages must be provided")

        # Extract Mellea-specific parameters
        requirements = kwargs.get("requirements")
        strategy = kwargs.get("strategy")

        # Merge kwargs into model_options (temperature, max_tokens, etc.)
        merged_options = {**self.kwargs, **model_options}

        # Remove DSPy-specific parameters that Mellea doesn't need
        merged_options.pop("cache", None)
        merged_options.pop("model_type", None)

        # Generate with Mellea using base class async method
        response = await self._agenerate_with_mellea(
            prompt_text,
            merged_options,
            tool_calls_enabled=False,
            requirements=requirements,
            strategy=strategy,
        )

        # Convert response using message converter
        openai_response = self.message_converter.from_mellea(response)

        # Update model name in response
        openai_response.model = self.model

        return openai_response


# Made with Bob
