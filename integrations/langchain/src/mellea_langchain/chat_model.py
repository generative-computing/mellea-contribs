"""LangChain-compatible chat model that wraps Mellea."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from mellea_integration import MelleaIntegrationBase
from pydantic import Field

from .message_conversion import LangChainMessageConverter
from .tool_conversion import LangChainToolConverter

try:
    from mellea import MelleaSession
    from mellea.backends import ModelOption
    from mellea.core import Context
    from mellea.stdlib.requirements import check, req
    from mellea.stdlib.sampling import RejectionSamplingStrategy
except ImportError:
    # Fallback for type hints if mellea is not installed
    MelleaSession = Any  # type: ignore
    ModelOption = Any  # type: ignore
    Context = Any  # type: ignore
    req = None  # type: ignore
    check = None  # type: ignore
    RejectionSamplingStrategy = None  # type: ignore


class MelleaChatModel(BaseChatModel, MelleaIntegrationBase):
    """LangChain chat model that uses Mellea as the backend.

    This allows LangChain applications to use Mellea's generative
    programming capabilities through the standard LangChain interface.

    Example:
        ```python
        from mellea import start_session
        from mellea_langchain import MelleaChatModel

        # Create Mellea session
        m = start_session()

        # Create LangChain chat model
        chat_model = MelleaChatModel(mellea_session=m)

        # Use with LangChain
        from langchain_core.messages import HumanMessage
        response = chat_model.invoke([HumanMessage(content="Hello!")])
        ```
    """

    mellea_session: Any = Field(description="The Mellea session to use for generation")
    model_name: str = Field(default="mellea", description="Name to identify this model")
    streaming: bool = Field(default=False, description="Whether to stream responses by default")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra attributes for integration base

    def __init__(
        self,
        mellea_session: Any,
        model_name: str = "mellea",
        streaming: bool = False,
        requirements: list[Any] | None = None,
        strategy: Any | None = None,
        **kwargs: Any,
    ):
        """Initialize the Mellea chat model.

        Args:
            mellea_session: Configured Mellea session
            model_name: Name to identify this model
            streaming: Whether to stream by default
            requirements: Optional list of requirements for validation
            strategy: Optional sampling strategy for validation
            **kwargs: Additional LangChain model parameters
        """
        # Initialize BaseChatModel first (Pydantic model)
        BaseChatModel.__init__(
            self,
            mellea_session=mellea_session,
            model_name=model_name,
            streaming=streaming,
            **kwargs,
        )

        # Then initialize MelleaIntegrationBase attributes manually
        # (avoid calling __init__ which conflicts with Pydantic)
        self.message_converter = LangChainMessageConverter()
        self.tool_converter = LangChainToolConverter()
        self._requirements = requirements
        self._strategy = strategy
        self._kwargs = kwargs

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "MelleaChatModel":
        """Bind tools to this chat model.

        This method creates a new instance of the chat model with tools attached.
        The tools will be passed to Mellea during generation, enabling the model
        to call functions/tools as needed.

        Args:
            tools: List of LangChain tools (BaseTool) or Mellea tools (MelleaTool).
                   LangChain tools will be automatically converted to Mellea format.
            **kwargs: Additional binding parameters (e.g., tool_choice)

        Returns:
            New MelleaChatModel instance with tools bound

        Example:
            ```python
            from langchain_core.tools import tool

            @tool
            def web_search(query: str) -> str:
                '''Search the web.'''
                return f"Results for: {query}"

            # Bind tools to the model
            model_with_tools = chat_model.bind_tools([web_search])

            # Use with LangChain agents
            response = model_with_tools.invoke([
                HumanMessage(content="Search for Python tutorials")
            ])
            ```
        """
        # Create a new instance with the same configuration
        bound_model = self.__class__(
            mellea_session=self.mellea_session,
            model_name=self.model_name,
            streaming=self.streaming,
            requirements=self._requirements,
            strategy=self._strategy,
        )

        # Store tools and tool choice for later use
        bound_model._bound_tools = tools  # type: ignore
        bound_model._tool_choice = kwargs.get("tool_choice")  # type: ignore

        return bound_model

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "mellea"

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute tool calls and return results.

        Args:
            tool_calls: List of tool call dictionaries with 'id', 'name', and 'args'

        Returns:
            List of tool execution results
        """
        if not hasattr(self, "_bound_tools") or not self._bound_tools:
            return []

        # Create tools dictionary for easy lookup
        tools_dict = {tool.name: tool for tool in self._bound_tools}

        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "unknown")

            if tool_name in tools_dict:
                try:
                    # Execute the tool
                    result = tools_dict[tool_name].invoke(tool_args)
                    results.append(
                        {
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": str(result),
                            "success": True,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": f"Error executing {tool_name}: {e!s}",
                            "success": False,
                        }
                    )
            else:
                results.append(
                    {
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": f"Tool '{tool_name}' not found",
                        "success": False,
                    }
                )

        return results

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation.

        Args:
            messages: List of messages to generate from
            stop: Stop sequences (not currently supported)
            run_manager: Callback manager for this run
            **kwargs: Additional generation parameters

        Returns:
            ChatResult with generated message
        """
        # Get bound tools if available
        tools = getattr(self, "_bound_tools", None) or kwargs.get("tools")

        # Prepare generation using base class
        prompt, model_options, tool_calls_enabled = self._prepare_generation(
            messages, tools, **kwargs
        )

        # Extract requirements/strategy from kwargs or model_options
        # (can be passed directly or in model_options dict)
        requirements = kwargs.get("requirements") or model_options.pop("requirements", None)
        strategy = kwargs.get("strategy") or model_options.pop("strategy", None)
        return_sampling_results = kwargs.get("return_sampling_results", False) or model_options.pop(
            "return_sampling_results", False
        )

        # Generate with Mellea using base class method
        response = self._generate_with_mellea(
            prompt,
            model_options,
            tool_calls_enabled,
            requirements,
            strategy,
            return_sampling_results,
        )

        # Handle sampling results if needed
        if return_sampling_results:
            response = self._handle_sampling_results(response)

        # Convert response using message converter
        result = self.message_converter.from_mellea(response)

        # Execute tool calls if present
        if tool_calls_enabled and result.generations:
            ai_message = result.generations[0].message
            if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                # Execute the tools
                tool_results = self._execute_tool_calls(ai_message.tool_calls)

                # Store tool execution results in the AIMessage
                if tool_results:
                    from langchain_core.messages import AIMessage
                    from langchain_core.outputs import ChatGeneration

                    updated_message = AIMessage(
                        content=ai_message.content,
                        tool_calls=ai_message.tool_calls,
                        additional_kwargs={
                            **ai_message.additional_kwargs,
                            "tool_execution_results": tool_results,
                        },
                        response_metadata={
                            **ai_message.response_metadata,
                            "tool_execution_results": tool_results,
                        },
                        id=ai_message.id,
                    )

                    result.generations[0] = ChatGeneration(
                        message=updated_message,
                        generation_info=result.generations[0].generation_info,
                    )

        # Invoke callbacks
        if run_manager:
            run_manager.on_llm_end(result)

        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous generation using Mellea.

        Supports both standard chat and validated generation with requirements/strategies.
        When requirements or strategy are provided, uses Mellea's ainstruct method for
        validation and retry logic. Otherwise, uses standard achat method.

        Args:
            messages: List of messages to generate from
            stop: Stop sequences (not currently supported)
            run_manager: Async callback manager for this run
            **kwargs: Additional generation parameters including:
                - requirements: List of requirement strings or req()/check() objects
                - strategy: Sampling strategy (e.g., RejectionSamplingStrategy)
                - return_sampling_results: Boolean to get detailed validation info
                - model_options: Additional Mellea model options

        Returns:
            ChatResult with generated message
        """
        # Get bound tools if available
        tools = getattr(self, "_bound_tools", None) or kwargs.get("tools")

        # Prepare generation using base class
        prompt, model_options, tool_calls_enabled = self._prepare_generation(
            messages, tools, **kwargs
        )

        # Extract requirements/strategy from kwargs or model_options
        # (can be passed directly or in model_options dict)
        requirements = kwargs.get("requirements") or model_options.pop("requirements", None)
        strategy = kwargs.get("strategy") or model_options.pop("strategy", None)
        return_sampling_results = kwargs.get("return_sampling_results", False) or model_options.pop(
            "return_sampling_results", False
        )

        # Generate with Mellea using base class async method
        response = await self._agenerate_with_mellea(
            prompt,
            model_options,
            tool_calls_enabled,
            requirements,
            strategy,
            return_sampling_results,
        )

        # Handle sampling results if needed
        if return_sampling_results:
            response = self._handle_sampling_results(response)

        # Convert response using message converter
        result = self.message_converter.from_mellea(response)

        # Execute tool calls if present
        if tool_calls_enabled and result.generations:
            ai_message = result.generations[0].message
            if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                # Execute the tools
                tool_results = self._execute_tool_calls(ai_message.tool_calls)

                # Store tool execution results in the AIMessage
                if tool_results:
                    from langchain_core.messages import AIMessage
                    from langchain_core.outputs import ChatGeneration

                    updated_message = AIMessage(
                        content=ai_message.content,
                        tool_calls=ai_message.tool_calls,
                        additional_kwargs={
                            **ai_message.additional_kwargs,
                            "tool_execution_results": tool_results,
                        },
                        response_metadata={
                            **ai_message.response_metadata,
                            "tool_execution_results": tool_results,
                        },
                        id=ai_message.id,
                    )

                    result.generations[0] = ChatGeneration(
                        message=updated_message,
                        generation_info=result.generations[0].generation_info,
                    )

        # Invoke callbacks
        if run_manager:
            await run_manager.on_llm_end(result)

        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronous streaming.

        Args:
            messages: List of messages to generate from
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional parameters

        Yields:
            ChatGenerationChunk objects

        Note:
            Mellea's achat doesn't support streaming, so this returns
            the full response as a single chunk.
        """
        # Generate full response
        result = self._generate(messages, stop, run_manager, **kwargs)

        # Return as single chunk
        content = result.generations[0].message.content
        chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))

        if run_manager:
            run_manager.on_llm_new_token(content)

        yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronous streaming.

        Args:
            messages: List of messages to generate from
            stop: Stop sequences
            run_manager: Async callback manager
            **kwargs: Additional parameters

        Yields:
            ChatGenerationChunk objects

        Note:
            Mellea's achat doesn't support streaming, so this returns
            the full response as a single chunk.
        """
        # Generate full response
        result = await self._agenerate(messages, stop, run_manager, **kwargs)

        # Return as single chunk
        content = result.generations[0].message.content
        chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))

        if run_manager:
            await run_manager.on_llm_new_token(content)

        yield chunk


# Made with Bob
