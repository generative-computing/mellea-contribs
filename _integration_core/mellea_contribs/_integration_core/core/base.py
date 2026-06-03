"""Base integration class for Mellea framework adapters."""

from abc import ABC, abstractmethod
from typing import Any

try:
    from mellea import MelleaSession
    from mellea.backends import ModelOption
    from mellea.stdlib.sampling import SamplingResult
except ImportError:
    # Fallback for type hints if mellea is not installed
    MelleaSession = Any  # type: ignore
    ModelOption = Any  # type: ignore
    SamplingResult = None  # type: ignore

from .types import MessageConverter, ToolConverter


class MelleaIntegrationBase(ABC):
    """Base class for Mellea framework integrations.

    Provides common functionality for integrating Mellea with various AI frameworks
    (LangChain, CrewAI, DSPy, etc.) while allowing framework-specific customization.

    This class handles:
    - Message conversion between framework and Mellea formats
    - Tool conversion and management
    - Mellea session interaction (chat vs instruct)
    - Requirements and sampling strategy support
    - Common generation patterns

    Attributes:
        mellea_session: Configured Mellea session for generation
        message_converter: Converter for framework-specific messages
        tool_converter: Optional converter for framework-specific tools
        _requirements: Optional list of requirements for validation
        _strategy: Optional sampling strategy for validation
        _kwargs: Additional configuration parameters
    """

    def __init__(
        self,
        mellea_session: Any,
        message_converter: MessageConverter,
        tool_converter: ToolConverter | None = None,
        requirements: list[Any] | None = None,
        strategy: Any | None = None,
        **kwargs: Any,
    ):
        """Initialize the Mellea integration base.

        Args:
            mellea_session: Configured Mellea session
            message_converter: Message converter for this framework
            tool_converter: Optional tool converter for this framework
            requirements: Optional list of requirements for validation
            strategy: Optional sampling strategy for validation
            **kwargs: Additional configuration parameters
        """
        self.mellea_session = mellea_session
        self.message_converter = message_converter
        self.tool_converter = tool_converter
        self._requirements = requirements
        self._strategy = strategy
        self._kwargs = kwargs

    def _prepare_generation(
        self,
        messages: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any], bool]:
        """Prepare inputs for Mellea generation.

        Converts framework-specific messages and tools to Mellea format,
        extracts the prompt, and prepares model options.

        Args:
            messages: Framework-specific message format
            tools: Optional list of framework-specific tools
            **kwargs: Additional parameters (may include model_options)

        Returns:
            Tuple of (prompt, model_options, tool_calls_enabled)
                - prompt: String prompt for generation
                - model_options: Dictionary of Mellea model options
                - tool_calls_enabled: Whether tool calling is enabled

        Raises:
            ValueError: If no messages are provided or no user message found
        """
        # Convert messages to Mellea format
        mellea_messages = self.message_converter.to_mellea(messages)

        # Extract last user message as prompt
        if not mellea_messages:
            raise ValueError("No messages provided for generation")

        prompt = self.message_converter.extract_last_user_message(mellea_messages)

        # Prepare model options
        model_options: dict[str, Any] = kwargs.get("model_options", {}).copy()

        # Handle tools if provided
        tool_calls_enabled = False
        if tools and self.tool_converter:
            mellea_tools = self.tool_converter.to_mellea(tools)
            if mellea_tools:
                model_options[ModelOption.TOOLS] = mellea_tools
                tool_calls_enabled = True

        return prompt, model_options, tool_calls_enabled

    def _generate_with_mellea(
        self,
        prompt: str,
        model_options: dict[str, Any],
        tool_calls_enabled: bool,
        requirements: list[Any] | None = None,
        strategy: Any | None = None,
        return_sampling_results: bool = False,
    ) -> Any:
        """Generate using appropriate Mellea method (sync).

        Chooses between chat() and instruct() based on whether
        requirements or strategy are provided.

        Args:
            prompt: String prompt for generation
            model_options: Dictionary of Mellea model options
            tool_calls_enabled: Whether tool calling is enabled
            requirements: Optional requirements for validation
            strategy: Optional sampling strategy
            return_sampling_results: Whether to return detailed sampling results

        Returns:
            Mellea response object (ModelOutputThunk or SamplingResult)
        """
        # Use provided requirements/strategy or fall back to instance defaults
        reqs = requirements if requirements is not None else self._requirements
        strat = strategy if strategy is not None else self._strategy

        # Use instruct method when requirements or strategy are provided
        if reqs is not None or strat is not None:
            # Note: instruct() doesn't have tool_calls parameter
            # Tools should be in model_options already
            return self.mellea_session.instruct(
                prompt,
                requirements=reqs,
                strategy=strat,
                model_options=model_options,
                return_sampling_results=return_sampling_results,
            )
        else:
            # Use standard chat method
            return self.mellea_session.chat(
                prompt,
                model_options=model_options,
                tool_calls=tool_calls_enabled,
            )

    async def _agenerate_with_mellea(
        self,
        prompt: str,
        model_options: dict[str, Any],
        tool_calls_enabled: bool,
        requirements: list[Any] | None = None,
        strategy: Any | None = None,
        return_sampling_results: bool = False,
    ) -> Any:
        """Generate using appropriate Mellea method (async).

        Async version of _generate_with_mellea. Chooses between
        achat() and ainstruct() based on requirements/strategy.

        Args:
            prompt: String prompt for generation
            model_options: Dictionary of Mellea model options
            tool_calls_enabled: Whether tool calling is enabled
            requirements: Optional requirements for validation
            strategy: Optional sampling strategy
            return_sampling_results: Whether to return detailed sampling results

        Returns:
            Mellea response object (ModelOutputThunk or SamplingResult)
        """
        # Use provided requirements/strategy or fall back to instance defaults
        reqs = requirements if requirements is not None else self._requirements
        strat = strategy if strategy is not None else self._strategy

        # Use ainstruct method when requirements or strategy are provided
        if reqs is not None or strat is not None:
            # Note: ainstruct() doesn't have tool_calls parameter
            # Tools should be in model_options already
            return await self.mellea_session.ainstruct(
                prompt,
                requirements=reqs,
                strategy=strat,
                model_options=model_options,
                return_sampling_results=return_sampling_results,
            )
        else:
            # Use standard async chat method
            return await self.mellea_session.achat(
                prompt,
                model_options=model_options,
                tool_calls=tool_calls_enabled,
            )

    def _handle_sampling_results(self, response: Any) -> Any:
        """Handle sampling results from instruct method.

        When return_sampling_results=True, Mellea returns a SamplingResult
        object. This method extracts the appropriate result.

        Args:
            response: Response from instruct/ainstruct (may be SamplingResult)

        Returns:
            The actual generation result (ModelOutputThunk)

        Raises:
            ValueError: If validation failed and no samples were generated
        """
        # Check if this is a SamplingResult object (duck-typing for test compatibility)
        if hasattr(response, "success") and hasattr(response, "result"):
            if response.success:
                # Use the successful result
                return response.result
            else:
                # Use the first sample if validation failed
                if hasattr(response, "sample_generations") and response.sample_generations:
                    # Get content from sample generation
                    sample = response.sample_generations[0]
                    # Handle both .content and .value attributes
                    if hasattr(sample, "content"):
                        return type("obj", (), {"content": sample.content})()
                    elif hasattr(sample, "value"):
                        return type("obj", (), {"content": sample.value})()
                    return sample
                else:
                    raise ValueError("No samples generated during validation")

        # Not a SamplingResult, return as-is
        return response

    @abstractmethod
    def generate(self, messages: Any, **kwargs: Any) -> Any:
        """Framework-specific synchronous generation method.

        This method must be implemented by subclasses to provide
        framework-specific generation logic.

        Args:
            messages: Framework-specific message format
            **kwargs: Additional generation parameters

        Returns:
            Framework-specific response format
        """
        pass

    @abstractmethod
    async def agenerate(self, messages: Any, **kwargs: Any) -> Any:
        """Framework-specific asynchronous generation method.

        This method must be implemented by subclasses to provide
        framework-specific async generation logic.

        Args:
            messages: Framework-specific message format
            **kwargs: Additional generation parameters

        Returns:
            Framework-specific response format
        """
        pass
