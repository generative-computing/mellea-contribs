"""MCP tool discovery and MelleaTool wrapping."""

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mellea.backends import MelleaTool


class MCPToolSpec:
    """Metadata for a single tool from an MCP server.

    Holds everything needed to inspect or instantiate a :class:`MelleaTool`
    without keeping a live session open.

    Attributes:
        name (str): Tool name as registered on the server.
        description (str): Human-readable description from the server.
        input_schema (dict): OpenAI-compatible parameters schema dict.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        connection: dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._connection = connection

    def as_mellea_tool(self) -> MelleaTool:
        """Create a callable :class:`MelleaTool` from this spec.

        The returned tool opens a fresh MCP session per call, which avoids
        cross-event-loop conflicts with Mellea's background event loop model.
        For ``stdio`` transport this means a new process is spawned on every
        tool invocation; prefer ``streamable_http`` or ``sse`` for performance-sensitive use.

        Returns:
            A ``MelleaTool`` instance ready to pass via ``ModelOption.TOOLS``.
        """
        return MelleaTool(
            self.name,
            _make_sync_call(self._connection, self.name),
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.input_schema,
                },
            },
        )

    def __repr__(self) -> str:
        return f"MCPToolSpec(name={self.name!r})"


async def discover_mcp_tools(connection: dict[str, Any]) -> list[MCPToolSpec]:
    """Discover all tools on an MCP server and return their metadata.

    Opens a single session, calls ``list_tools()``, then closes. No tools
    are instantiated — callers can inspect and filter before calling
    :meth:`MCPToolSpec.as_mellea_tool`.

    Args:
        connection: Transport config dict. Use the connection helpers rather than
            building this directly: :func:`mellea_mcp.http_connection`,
            :func:`mellea_mcp.sse_connection`, :func:`mellea_mcp.stdio_connection`.

    Returns:
        List of :class:`MCPToolSpec` objects, one per tool on the server.
    """
    async with _open_session(connection) as session:
        result = await session.list_tools()
        return [
            MCPToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if tool.inputSchema is not None else {},
                connection=connection,
            )
            for tool in result.tools
        ]


@asynccontextmanager
async def _open_session(connection: dict[str, Any]):
    """Open a fresh MCP ClientSession for the given connection config."""
    transport = connection.get("transport", "streamable_http")

    if transport == "streamable_http":
        async with httpx.AsyncClient(headers=connection.get("headers", {})) as http_client:
            async with streamable_http_client(
                connection["url"],
                http_client=http_client,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    elif transport == "sse":
        async with sse_client(
            url=connection["url"],
            headers=connection.get("headers", {}),
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    elif transport == "stdio":
        params = StdioServerParameters(
            command=connection["command"],
            args=connection.get("args", []),
            env=connection.get("env"),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    else:
        raise ValueError(f"Unknown MCP transport: {transport!r}")


async def _execute_tool(connection: dict[str, Any], tool_name: str, kwargs: dict[str, Any]) -> str:
    async with _open_session(connection) as session:
        result = await session.call_tool(tool_name, arguments=kwargs)
        if result.content:
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif hasattr(block, "resource") and hasattr(block.resource, "text"):
                    # EmbeddedResource wrapping a TextResourceContents
                    parts.append(block.resource.text)
                elif hasattr(block, "mimeType"):
                    # ImageContent — binary, not usable as text
                    parts.append(f"[binary: {block.mimeType}]")
                elif hasattr(block, "resource"):
                    # EmbeddedResource wrapping a BlobResourceContents
                    mime = getattr(block.resource, "mimeType", "unknown")
                    parts.append(f"[binary: {mime}]")
            return "\n".join(parts) if parts else ""
        return ""


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _make_sync_call(connection: dict[str, Any], tool_name: str) -> Any:
    def sync_call(**kwargs: Any) -> str:
        # Strip None values: MCP servers expect absent fields, not explicit nulls.
        clean = {k: v for k, v in kwargs.items() if v is not None}
        # Run the async MCP call in a dedicated worker thread so asyncio.run()
        # never competes with Mellea's background _EventLoopHandler loop.
        return _executor.submit(
            asyncio.run,
            _execute_tool(connection, tool_name, clean),
        ).result()

    return sync_call
