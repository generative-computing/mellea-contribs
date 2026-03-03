"""CrewAI-compatible LLM implementation using Mellea."""

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from mellea_integration import MelleaIntegrationBase
from pydantic import BaseModel

from .message_conversion import CrewAIMessageConverter
from .tool_conversion import CrewAIToolConverter


@contextmanager
def llm_call_context() -> Generator[str, None, None]:
    """Context manager for LLM call tracking.

    Generates a unique call_id for tracking LLM calls across events.
    Compatible with both old and new versions of CrewAI.
    """
    call_id = str(uuid.uuid4())
    yield call_id


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage

try:
    from mellea import MelleaSession as _MelleaSession
    from mellea.backends import ModelOption
    from mellea.core import SamplingStrategy as _SamplingStrategy
except ImportError:
    # Fallback for type hints if mellea is not installed
    _MelleaSession = Any  # type: ignore
    ModelOption = Any  # type: ignore
    _SamplingStrategy = Any  # type: ignore


class MelleaLLM(BaseLLM, MelleaIntegrationBase):
    """CrewAI LLM implementation using Mellea as the backend.

    This allows CrewAI applications to use Mellea's generative programming
    capabilities including requirements, validation, and sampling strategies.

    Attributes:
        mellea_session: Configured Mellea session
        requirements: List of requirements for validation
        strategy: Sampling strategy for validation/retry
        return_sampling_results: Whether to return detailed validation info

    Example:
        ```python
        from mellea import start_session
        from mellea_crewai import MelleaLLM
        from crewai import Agent, Task, Crew

        # Create Mellea session
        m = start_session()

        # Create CrewAI LLM
        llm = MelleaLLM(mellea_session=m)

        # Use with CrewAI agents
        agent = Agent(
            role="Researcher",
            goal="Research topics thoroughly",
            backstory="You are an expert researcher",
            llm=llm
        )

        task = Task(
            description="Research AI trends",
            agent=agent
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        ```

    Example with requirements:
        ```python
        from mellea.stdlib.requirements import req, check
        from mellea.stdlib.sampling import RejectionSamplingStrategy

        llm = MelleaLLM(
            mellea_session=m,
            requirements=[
                req("The response should be professional"),
                req("Include specific examples"),
                check("Do not mention competitors")
            ],
            strategy=RejectionSamplingStrategy(loop_budget=5)
        )
        ```
    """

    def __init__(
        self,
        mellea_session: Any,
        model: str = "mellea",
        temperature: float | None = None,
        requirements: list[Any] | None = None,
        strategy: Any | None = None,
        return_sampling_results: bool = False,
        **kwargs: Any,
    ):
        """Initialize MelleaLLM.

        Args:
            mellea_session: Configured Mellea session
            model: Model identifier (for display purposes)
            temperature: Temperature setting (passed to Mellea)
            requirements: List of requirements for validation
            strategy: Sampling strategy for validation/retry
            return_sampling_results: Whether to return detailed validation info
            **kwargs: Additional parameters passed to BaseLLM
        """
        # Initialize CrewAI BaseLLM
        BaseLLM.__init__(self, model=model, temperature=temperature, **kwargs)

        # Initialize MelleaIntegrationBase
        MelleaIntegrationBase.__init__(
            self,
            mellea_session=mellea_session,
            message_converter=CrewAIMessageConverter(),
            tool_converter=CrewAIToolConverter(),
            requirements=requirements,
            strategy=strategy,
            **kwargs,
        )

        self._return_sampling_results = return_sampling_results

    def call(
        self,
        messages: str | list["LLMMessage"],
        tools: list[dict[str, "BaseTool"]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: "Task | None" = None,
        from_agent: "Agent | None" = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Synchronous LLM call using Mellea.

        Args:
            messages: Input messages for the LLM
            tools: Optional list of tool schemas for function calling
            callbacks: Optional list of callback functions
            available_functions: Optional dict mapping function names to callables
            from_task: Optional task caller
            from_agent: Optional agent caller
            response_model: Optional response model for structured output

        Returns:
            String response from the LLM or tool execution result

        Raises:
            ValueError: If the messages format is invalid
            RuntimeError: If the LLM request fails
        """
        # Establish call context for event tracking
        with llm_call_context():
            # Emit call started event
            self._emit_call_started_event(
                messages, tools, callbacks, available_functions, from_task, from_agent
            )

            try:
                # WORKAROUND: Extract tools from from_task or from_agent if tools parameter is None
                # This is a temporary solution until we understand CrewAI's tool passing mechanism better
                if not tools:
                    if from_task and hasattr(from_task, "tools") and from_task.tools:
                        tools = from_task.tools
                    elif from_agent and hasattr(from_agent, "tools") and from_agent.tools:
                        tools = from_agent.tools

                # Prepare generation using base class (handles message/tool conversion)
                prompt, model_options, tool_calls_enabled = self._prepare_generation(
                    messages, tools, model_options={}
                )

                # Add temperature if set
                if self.temperature is not None:
                    model_options[ModelOption.TEMPERATURE] = self.temperature

                # Generate with Mellea using base class method
                response = self._generate_with_mellea(
                    prompt,
                    model_options,
                    tool_calls_enabled,
                    requirements=self._requirements,
                    strategy=self._strategy,
                    return_sampling_results=self._return_sampling_results,
                )

                # Handle sampling results if needed
                if self._return_sampling_results:
                    response = self._handle_sampling_results(response)

                # Convert response using message converter
                result = self.message_converter.from_mellea(response)

                # Check if result contains tool calls (after conversion)
                # mellea_to_crewai_response should have parsed the tool calls
                tool_calls = None

                # Check various places for tool calls
                # Only consider it a tool call if it's a non-empty list
                if hasattr(response, "tool_calls") and response.tool_calls:
                    if isinstance(response.tool_calls, list) and len(response.tool_calls) > 0:
                        tool_calls = response.tool_calls
                elif hasattr(response, "_tool_calls") and response._tool_calls:
                    if isinstance(response._tool_calls, list) and len(response._tool_calls) > 0:
                        tool_calls = response._tool_calls
                elif isinstance(result, list) and len(result) > 0:
                    # Check if result is a list of ToolCall objects
                    if hasattr(result[0], "function"):
                        tool_calls = result

                # Apply stop words only if result is a string
                if isinstance(result, str):
                    result = self._apply_stop_words(result)

                # Handle tool calls if present (following OpenAI/Anthropic/Gemini pattern)
                if tool_calls:
                    # Build available_functions from task/agent tools if not provided
                    # WORKAROUND: CrewAI doesn't always pass available_functions, so we build it
                    if not available_functions:
                        available_functions = {}
                        tools_source = None
                        if from_task and hasattr(from_task, "tools") and from_task.tools:
                            tools_source = from_task.tools
                        elif from_agent and hasattr(from_agent, "tools") and from_agent.tools:
                            tools_source = from_agent.tools

                        if tools_source:
                            for tool in tools_source:
                                if hasattr(tool, "name"):
                                    # Try different ways to get the callable
                                    if hasattr(tool, "func"):
                                        available_functions[tool.name] = tool.func
                                    elif hasattr(tool, "_run"):
                                        available_functions[tool.name] = tool._run
                                    elif hasattr(tool, "run"):
                                        available_functions[tool.name] = tool.run
                                    elif callable(tool):
                                        available_functions[tool.name] = tool

                    # If still no available_functions, return tool calls for executor to handle
                    if not available_functions:
                        # Convert to list if not already
                        tool_calls_list = (
                            tool_calls if isinstance(tool_calls, list) else [tool_calls]
                        )
                        self._emit_call_completed_event(
                            response=tool_calls_list,
                            call_type=LLMCallType.TOOL_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=messages,
                        )
                        return tool_calls_list

                    # Execute tools internally
                    for tool_call in tool_calls:
                        # Extract name and arguments from tool_call
                        # ToolCall has a function attribute with name and arguments
                        if hasattr(tool_call, "function"):
                            tool_name = tool_call.function.name
                            tool_args = tool_call.function.arguments
                        else:
                            tool_name = (
                                tool_call.name if hasattr(tool_call, "name") else str(tool_call)
                            )
                            tool_args = (
                                tool_call.arguments if hasattr(tool_call, "arguments") else {}
                            )

                        tool_result = self._handle_tool_execution(
                            tool_name,
                            tool_args,
                            available_functions,
                            from_task,
                            from_agent,
                        )
                        if tool_result is not None:
                            return tool_result

                # Track token usage if available
                if hasattr(response, "usage") and response.usage:
                    # Convert usage object to dict if needed
                    usage_dict = (
                        response.usage
                        if isinstance(response.usage, dict)
                        else {
                            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                            "total_tokens": getattr(response.usage, "total_tokens", 0),
                        }
                    )
                    self._track_token_usage_internal(usage_dict)
                else:
                    # Track request even if usage info is not available
                    # This allows counting successful requests from backends that don't provide usage
                    self._track_token_usage_internal(
                        {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
                    )

                # Emit call completed event
                self._emit_call_completed_event(
                    result, LLMCallType.LLM_CALL, from_task, from_agent, messages
                )

                return result

            except Exception as e:
                # Emit call failed event
                self._emit_call_failed_event(str(e), from_task, from_agent)
                raise

    async def acall(
        self,
        messages: str | list["LLMMessage"],
        tools: list[dict[str, "BaseTool"]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: "Task | None" = None,
        from_agent: "Agent | None" = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Asynchronous LLM call using Mellea.

        Args:
            messages: Input messages for the LLM
            tools: Optional list of tool schemas for function calling
            callbacks: Optional list of callback functions
            available_functions: Optional dict mapping function names to callables
            from_task: Optional task caller
            from_agent: Optional agent caller
            response_model: Optional response model for structured output

        Returns:
            String response from the LLM or tool execution result

        Raises:
            ValueError: If the messages format is invalid
            RuntimeError: If the LLM request fails
        """
        # Establish call context for event tracking
        with llm_call_context():
            # Emit call started event
            self._emit_call_started_event(
                messages, tools, callbacks, available_functions, from_task, from_agent
            )

            try:
                # WORKAROUND: Extract tools from from_task or from_agent if tools parameter is None
                # This is a temporary solution until we understand CrewAI's tool passing mechanism better
                if not tools:
                    if from_task and hasattr(from_task, "tools") and from_task.tools:
                        tools = from_task.tools
                    elif from_agent and hasattr(from_agent, "tools") and from_agent.tools:
                        tools = from_agent.tools

                # Prepare generation using base class (handles message/tool conversion)
                prompt, model_options, tool_calls_enabled = self._prepare_generation(
                    messages, tools, model_options={}
                )

                # Add temperature if set
                if self.temperature is not None:
                    model_options[ModelOption.TEMPERATURE] = self.temperature

                # Generate with Mellea using base class async method
                response = await self._agenerate_with_mellea(
                    prompt,
                    model_options,
                    tool_calls_enabled,
                    requirements=self._requirements,
                    strategy=self._strategy,
                    return_sampling_results=self._return_sampling_results,
                )

                # Handle sampling results if needed
                if self._return_sampling_results:
                    response = self._handle_sampling_results(response)

                # Convert response using message converter
                result = self.message_converter.from_mellea(response)

                # Check if result contains tool calls
                tool_calls = None
                # Only consider it a tool call if it's a non-empty list
                if hasattr(response, "tool_calls") and response.tool_calls:
                    if isinstance(response.tool_calls, list) and len(response.tool_calls) > 0:
                        tool_calls = response.tool_calls
                elif hasattr(response, "_tool_calls") and response._tool_calls:
                    if isinstance(response._tool_calls, list) and len(response._tool_calls) > 0:
                        tool_calls = response._tool_calls
                elif isinstance(result, list) and len(result) > 0:
                    # Check if result is a list of ToolCall objects
                    if hasattr(result[0], "function"):
                        tool_calls = result

                # Apply stop words only if result is a string
                if isinstance(result, str):
                    result = self._apply_stop_words(result)

                # Handle tool calls if present (following OpenAI/Anthropic/Gemini pattern)
                if tool_calls:
                    # Build available_functions from task/agent tools if not provided
                    # WORKAROUND: CrewAI doesn't always pass available_functions, so we build it
                    if not available_functions:
                        available_functions = {}
                        tools_source = None
                        if from_task and hasattr(from_task, "tools") and from_task.tools:
                            tools_source = from_task.tools
                        elif from_agent and hasattr(from_agent, "tools") and from_agent.tools:
                            tools_source = from_agent.tools

                        if tools_source:
                            for tool in tools_source:
                                if hasattr(tool, "name"):
                                    # Try different ways to get the callable
                                    if hasattr(tool, "func"):
                                        available_functions[tool.name] = tool.func
                                    elif hasattr(tool, "_run"):
                                        available_functions[tool.name] = tool._run
                                    elif hasattr(tool, "run"):
                                        available_functions[tool.name] = tool.run
                                    elif callable(tool):
                                        available_functions[tool.name] = tool

                    # If still no available_functions, return tool calls for executor to handle
                    if not available_functions:
                        # Convert to list if not already
                        tool_calls_list = (
                            tool_calls if isinstance(tool_calls, list) else [tool_calls]
                        )
                        self._emit_call_completed_event(
                            response=tool_calls_list,
                            call_type=LLMCallType.TOOL_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=messages,
                        )
                        return tool_calls_list

                    # Execute tools internally
                    for tool_call in tool_calls:
                        # Extract name and arguments from tool_call
                        # ToolCall has a function attribute with name and arguments
                        if hasattr(tool_call, "function"):
                            tool_name = tool_call.function.name
                            tool_args = tool_call.function.arguments
                        else:
                            tool_name = (
                                tool_call.name if hasattr(tool_call, "name") else str(tool_call)
                            )
                            tool_args = (
                                tool_call.arguments if hasattr(tool_call, "arguments") else {}
                            )

                        tool_result = self._handle_tool_execution(
                            tool_name,
                            tool_args,
                            available_functions,
                            from_task,
                            from_agent,
                        )
                        if tool_result is not None:
                            return tool_result

                # Track token usage if available
                if hasattr(response, "usage") and response.usage:
                    # Convert usage object to dict if needed
                    usage_dict = (
                        response.usage
                        if isinstance(response.usage, dict)
                        else {
                            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                            "total_tokens": getattr(response.usage, "total_tokens", 0),
                        }
                    )
                    self._track_token_usage_internal(usage_dict)
                else:
                    # Track request even if usage info is not available
                    # This allows counting successful requests from backends that don't provide usage
                    self._track_token_usage_internal(
                        {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
                    )

                # Emit call completed event
                self._emit_call_completed_event(
                    result, LLMCallType.LLM_CALL, from_task, from_agent, messages
                )

                return result

            except Exception as e:
                # Emit call failed event
                self._emit_call_failed_event(str(e), from_task, from_agent)
                raise

    def generate(self, messages: Any, **kwargs: Any) -> Any:
        """Framework-specific synchronous generation method.

        This is required by MelleaIntegrationBase but not used directly.
        CrewAI uses call() instead.
        """
        return self.call(messages, **kwargs)

    async def agenerate(self, messages: Any, **kwargs: Any) -> Any:
        """Framework-specific asynchronous generation method.

        This is required by MelleaIntegrationBase but not used directly.
        CrewAI uses acall() instead.
        """
        return await self.acall(messages, **kwargs)

    def supports_stop_words(self) -> bool:
        """Check if the LLM supports stop words.

        Returns:
            True if stop words are configured (Mellea supports stop words via post-processing)
        """
        return self._supports_stop_words_implementation()

    def get_context_window_size(self) -> int:
        """Get the context window size for the LLM.

        Returns:
            The number of tokens the model can handle
        """
        # Try to get from Mellea backend
        if hasattr(self.mellea_session, "backend"):
            backend = self.mellea_session.backend
            if hasattr(backend, "get_context_window_size"):
                return backend.get_context_window_size()

        # Default fallback
        return super().get_context_window_size()

    def supports_multimodal(self) -> bool:
        """Check if the LLM supports multimodal inputs.

        Returns:
            True if the backend supports multimodal inputs
        """
        # Try to get from Mellea backend
        if hasattr(self.mellea_session, "backend"):
            backend = self.mellea_session.backend
            if hasattr(backend, "supports_multimodal"):
                return backend.supports_multimodal()

        return False
