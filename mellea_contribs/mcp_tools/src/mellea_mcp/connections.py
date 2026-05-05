"""Helpers for building MCP connection config dicts."""

from typing import Any


def http_connection(
    url: str,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    connect_timeout: float = 30.0,
    read_timeout: float = 300.0,
) -> dict[str, Any]:
    """Build a Streamable HTTP connection config.

    Args:
        url: MCP server URL.
        api_key: Sets ``Authorization: Bearer <api_key>``.
        headers: Additional headers, merged after ``api_key``.
        connect_timeout: Seconds to wait for TCP connection (default 30).
        read_timeout: Seconds to wait for a response (default 300).

    Returns:
        Connection dict ready to pass to :func:`discover_mcp_tools`.
    """
    h: dict[str, str] = {}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    if headers:
        h.update(headers)
    return {
        "transport": "streamable_http",
        "url": url,
        "headers": h,
        "connect_timeout": connect_timeout,
        "read_timeout": read_timeout,
    }


def sse_connection(
    url: str,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    connect_timeout: float = 30.0,
    read_timeout: float = 300.0,
) -> dict[str, Any]:
    """Build an SSE connection config.

    Args:
        url: MCP server URL.
        api_key: Sets ``Authorization: Bearer <api_key>``.
        headers: Additional headers, merged after ``api_key``.
        connect_timeout: Seconds to wait for TCP connection (default 30).
        read_timeout: Seconds to wait for a response (default 300).

    Returns:
        Connection dict ready to pass to :func:`discover_mcp_tools`.
    """
    h: dict[str, str] = {}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    if headers:
        h.update(headers)
    return {
        "transport": "sse",
        "url": url,
        "headers": h,
        "connect_timeout": connect_timeout,
        "read_timeout": read_timeout,
    }


def stdio_connection(
    command: str,
    *,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Build a stdio connection config.

    Args:
        command: Executable to run (e.g. ``"gh"``).
        args: Command-line arguments (e.g. ``["mcp", "serve"]``).
        env: Environment variables for the subprocess.
        timeout: Total seconds allowed for a tool call to complete (default 300).

    Returns:
        Connection dict ready to pass to :func:`discover_mcp_tools`.
    """
    conn: dict[str, Any] = {
        "transport": "stdio",
        "command": command,
        "read_timeout": timeout,
    }
    if args:
        conn["args"] = args
    if env:
        conn["env"] = env
    return conn
