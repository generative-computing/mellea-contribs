"""Mellea MCP integration — discover MCP server tools and use them in Mellea agents."""

from .connections import http_connection, sse_connection, stdio_connection
from .tools import MCPToolSpec, discover_mcp_tools

__all__ = [
    "MCPToolSpec",
    "discover_mcp_tools",
    "http_connection",
    "sse_connection",
    "stdio_connection",
]
