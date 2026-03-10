"""Example demonstrating tool/function calling with Mellea Chat Model.

This example shows how to:
1. Define LangChain tools
2. Bind tools to the Mellea chat model
3. Use the model with tools for function calling

Note: This example shows the model returning tool calls. For automatic tool
execution, use LangGraph's react agent pattern or create custom tool execution logic.
See the test_agent_integration.py file for working examples.
"""

import sys
from pathlib import Path

# Add parent directory to path to import mellea_langchain
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from mellea import start_session

from mellea_langchain import MelleaChatModel


# Define some example tools
@tool
def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The unit system (fahrenheit or celsius)
    """
    # This is a mock implementation
    return f"The weather in {location} is 72 degrees {unit} and sunny."


@tool
def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers.

    Args:
        a: First number
        b: Second number
    """
    return a + b


@tool
def web_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query
    """
    # This is a mock implementation
    return f"Search results for '{query}': [Mock results would appear here]"


def main():
    """Run tool calling examples."""
    print("=" * 60)
    print("Mellea Chat Model - Tool Calling Examples")
    print("=" * 60)

    # Create Mellea session
    m = start_session()

    # Create chat model
    chat_model = MelleaChatModel(mellea_session=m)

    # Example 1: Bind a single tool
    print("\n" + "=" * 60)
    print("Example 1: Weather Tool")
    print("=" * 60)

    model_with_weather = chat_model.bind_tools([get_current_weather])

    response = model_with_weather.invoke(
        [HumanMessage(content="What's the weather like in San Francisco?")]
    )

    print(f"\nResponse: {response.content}")

    # Check if tool calls were made
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\nTool calls requested: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  - Tool: {tool_call.get('name', 'unknown')}")
            print(f"    Args: {tool_call.get('args', {})}")
        print("\nNote: Tool calls are returned but not executed.")
        print("See test_agent_integration.py for automatic tool execution examples.")

    # Example 2: Bind multiple tools
    print("\n" + "=" * 60)
    print("Example 2: Multiple Tools")
    print("=" * 60)

    model_with_tools = chat_model.bind_tools([get_current_weather, calculate_sum, web_search])

    response = model_with_tools.invoke([HumanMessage(content="What is 25 + 17?")])

    print(f"\nResponse: {response.content}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\nTool calls requested: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  - Tool: {tool_call.get('name', 'unknown')}")
            print(f"    Args: {tool_call.get('args', {})}")
        print("\nNote: Tool calls are returned but not executed.")
        print("See test_agent_integration.py for automatic tool execution examples.")

    # Example 3: Complex query that might use multiple tools
    print("\n" + "=" * 60)
    print("Example 3: Complex Query")
    print("=" * 60)

    response = model_with_tools.invoke(
        [HumanMessage(content="Search for Python tutorials and tell me the weather in New York")]
    )

    print(f"\nResponse: {response.content}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\nTool calls requested: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  - Tool: {tool_call.get('name', 'unknown')}")
            print(f"    Args: {tool_call.get('args', {})}")
        print("\nNote: Tool calls are returned but not executed.")
        print("See test_agent_integration.py for automatic tool execution examples.")

    print("\n" + "=" * 60)
    print("Tool calling examples completed!")
    print("=" * 60)
    print("\nNote: This example shows the model returning tool calls.")
    print("For automatic tool execution, use LangGraph's react agent pattern")
    print("or create custom tool execution logic. Tool calling support")
    print("depends on the underlying model's capabilities.")


if __name__ == "__main__":
    main()
