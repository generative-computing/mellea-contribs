"""Tests for MCP tool discovery and MelleaTool wrapping."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from mellea_mcp.connections import http_connection


@pytest.fixture
def connection():
    return http_connection("https://example.com/mcp")


def _make_mcp_tool(name: str, description: str = "", schema: dict | None = None):
    """Build a minimal stand-in for mcp.types.Tool."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _make_session(*tools):
    """Build a mock MCP ClientSession with list_tools and call_tool."""
    session = AsyncMock()
    list_result = SimpleNamespace(tools=list(tools))
    session.list_tools = AsyncMock(return_value=list_result)
    return session


def _mock_open_session(session):
    """Return a patch for _open_session that yields the given mock session."""

    @asynccontextmanager
    async def _fake(*args, **kwargs):
        yield session

    return patch("mellea_mcp.tools._open_session", new=_fake)


class MockMelleaTool:
    """Minimal stand-in for MelleaTool."""

    def __init__(self, name, tool_call, as_json_tool):
        self.name = name
        self._call_func = tool_call
        self._as_json_tool = as_json_tool


class TestDiscoverMcpTools:
    @pytest.mark.asyncio
    async def test_returns_specs_for_each_tool(self, connection):
        session = _make_session(
            _make_mcp_tool("get_me", "Return the current user"),
            _make_mcp_tool("search_pull_requests", "Search PRs"),
        )
        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        assert len(specs) == 2
        assert [s.name for s in specs] == ["get_me", "search_pull_requests"]

    @pytest.mark.asyncio
    async def test_empty_server_returns_empty_list(self, connection):
        session = _make_session()
        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        assert specs == []

    @pytest.mark.asyncio
    async def test_spec_fields_populated(self, connection):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        session = _make_session(_make_mcp_tool("search", "Search things", schema))
        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        s = specs[0]
        assert s.name == "search"
        assert s.description == "Search things"
        assert s.input_schema == schema
        assert s._connection == connection

    @pytest.mark.asyncio
    async def test_none_input_schema_becomes_empty_dict(self, connection):
        tool = SimpleNamespace(name="x", description="", inputSchema=None)
        session = AsyncMock()
        session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[tool]))
        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        assert specs[0].input_schema == {}


class TestAsMelleaTool:
    @pytest.mark.asyncio
    async def test_produces_mellea_tool_with_correct_name(self, connection):
        session = _make_session(_make_mcp_tool("get_me", "Get current user"))
        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()

        assert tool.name == "get_me"

    @pytest.mark.asyncio
    async def test_json_schema_structure(self, connection):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        session = _make_session(_make_mcp_tool("search", "Search", schema))
        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()

        fn = tool._as_json_tool["function"]
        assert fn["name"] == "search"
        assert fn["description"] == "Search"
        assert fn["parameters"] == schema


def _call_result(*blocks, is_error=False):
    return CallToolResult(content=list(blocks), isError=is_error)


class TestSyncWrapper:
    @pytest.mark.asyncio
    async def test_extracts_text_from_content_blocks(self, connection):
        call_result = _call_result(TextContent(type="text", text="hello world"))
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session), patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()
            output = tool._call_func(q="test")

        assert output == "hello world"

    @pytest.mark.asyncio
    async def test_joins_multiple_content_blocks(self, connection):
        call_result = _call_result(
            TextContent(type="text", text="line1"),
            TextContent(type="text", text="line2"),
        )
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session), patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()
            output = tool._call_func()

        assert output == "line1\nline2"

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_string(self, connection):
        call_result = _call_result()
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session), patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()
            output = tool._call_func()

        assert output == ""

    @pytest.mark.asyncio
    async def test_none_kwargs_stripped(self, connection):
        """Kwargs with None values are not forwarded to call_tool."""
        received: list[dict] = []

        async def _capture(tool_name, *, arguments):
            received.append(arguments)
            return _call_result()

        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = _capture

        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session), patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()
            tool._call_func(q="test", page=None)

        assert received == [{"q": "test"}]

    @pytest.mark.asyncio
    async def test_embedded_resource_text_extracted(self, connection):
        resource = TextResourceContents(uri="file://doc.txt", text="resource text")
        call_result = _call_result(EmbeddedResource(type="resource", resource=resource))
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session), patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()
            output = tool._call_func()

        assert output == "resource text"

    @pytest.mark.asyncio
    async def test_image_content_returns_binary_descriptor(self, connection):
        call_result = _call_result(ImageContent(type="image", data="abc123", mimeType="image/png"))
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session), patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()
            output = tool._call_func()

        assert output == "[binary: image/png]"

    @pytest.mark.asyncio
    async def test_blob_resource_returns_binary_descriptor(self, connection):
        resource = BlobResourceContents(
            uri="file://data.pdf", blob="abc123", mimeType="application/pdf"
        )
        call_result = _call_result(EmbeddedResource(type="resource", resource=resource))
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session), patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()
            output = tool._call_func()

        assert output == "[binary: application/pdf]"

    @pytest.mark.asyncio
    async def test_is_error_returns_error_string(self, connection):
        call_result = _call_result(TextContent(type="text", text="not found"), is_error=True)
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session), patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()
            output = tool._call_func()

        assert output == "[tool error] not found"

    @pytest.mark.asyncio
    async def test_is_error_no_content_returns_fallback(self, connection):
        call_result = _call_result(is_error=True)
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            from mellea_mcp.tools import discover_mcp_tools

            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session), patch("mellea_mcp.tools.MelleaTool", MockMelleaTool):
            tool = specs[0].as_mellea_tool()
            output = tool._call_func()

        assert output == "[tool error] tool call failed"
