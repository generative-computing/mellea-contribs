"""Convert CrewAI tools to Mellea format."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crewai.tools.base_tool import BaseTool
    from mellea.backends.tools import MelleaTool

CREWAI_AVAILABLE = False
BaseTool = None
MelleaTool = None

try:
    from crewai.tools.base_tool import BaseTool as _BaseTool  # type: ignore
    from mellea.backends.tools import MelleaTool as _MelleaTool  # type: ignore

    BaseTool = _BaseTool
    MelleaTool = _MelleaTool
    CREWAI_AVAILABLE = True
except ImportError:
    pass


def crewai_to_mellea_tools(tools: list[Any]) -> list[Any]:
    """Convert CrewAI tools to Mellea tool format.

    Args:
        tools: List of CrewAI tools (BaseTool instances or dicts containing tools)

    Returns:
        List of Mellea-compatible tool objects

    Example:
        >>> from crewai.tools import Tool
        >>> crewai_tool = Tool(name="search", description="Search the web", ...)
        >>> mellea_tools = crewai_to_mellea_tools([crewai_tool])

    Note:
        This function handles both direct BaseTool instances and
        dictionaries that contain tools (as CrewAI sometimes passes).
    """
    if not CREWAI_AVAILABLE:
        raise ImportError("CrewAI is not installed. Please install it with: pip install crewai")

    mellea_tools = []

    for tool_item in tools:
        # Extract tool from dict if necessary
        if isinstance(tool_item, dict):
            tool = tool_item.get("tool") or tool_item.get("function") or tool_item
        else:
            tool = tool_item

        # Check if it's already a Mellea tool
        if MelleaTool and isinstance(tool, MelleaTool):
            mellea_tools.append(tool)
            continue

        # Convert CrewAI tool to Mellea format
        if BaseTool and isinstance(tool, BaseTool):
            # Extract tool properties
            name = getattr(tool, "name", "unknown_tool")
            description = getattr(tool, "description", "")

            # Get the function to call
            if hasattr(tool, "_run"):
                function = tool._run
            elif hasattr(tool, "run"):
                function = tool.run
            elif callable(tool):
                function = tool
            else:
                raise ValueError(f"Tool {name} does not have a callable run method")

            # Get parameters schema if available
            parameters = {}
            if hasattr(tool, "args_schema") and tool.args_schema:
                try:
                    parameters = tool.args_schema.schema()
                except Exception:
                    # If schema extraction fails, use empty dict
                    parameters = {}

            # Create JSON tool schema for Mellea
            as_json_tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }

            # Create Mellea tool with correct constructor signature
            mellea_tool = MelleaTool(  # type: ignore
                name=name,
                tool_call=function,
                as_json_tool=as_json_tool,
            )
            mellea_tools.append(mellea_tool)
        else:
            # If we can't identify the tool type, try to pass it through
            # This allows for custom tool formats
            mellea_tools.append(tool)

    return mellea_tools
