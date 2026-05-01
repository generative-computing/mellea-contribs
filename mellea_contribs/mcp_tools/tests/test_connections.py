"""Tests for MCP connection config helpers."""

from mellea_mcp.connections import http_connection, sse_connection, stdio_connection


class TestHttpConnection:
    def test_sets_transport(self):
        assert http_connection("https://example.com")["transport"] == "streamable_http"

    def test_sets_url(self):
        assert http_connection("https://example.com")["url"] == "https://example.com"

    def test_token_sets_auth_header(self):
        conn = http_connection("https://example.com", api_key="abc")
        assert conn["headers"]["Authorization"] == "Bearer abc"

    def test_no_token_empty_headers(self):
        assert http_connection("https://example.com")["headers"] == {}

    def test_extra_headers_merged(self):
        conn = http_connection("https://example.com", api_key="abc", headers={"X-Custom": "val"})
        assert conn["headers"]["Authorization"] == "Bearer abc"
        assert conn["headers"]["X-Custom"] == "val"

    def test_headers_only(self):
        conn = http_connection("https://example.com", headers={"X-Key": "v"})
        assert conn["headers"] == {"X-Key": "v"}


class TestSseConnection:
    def test_sets_transport(self):
        assert sse_connection("https://example.com")["transport"] == "sse"

    def test_token_sets_auth_header(self):
        conn = sse_connection("https://example.com", api_key="tok")
        assert conn["headers"]["Authorization"] == "Bearer tok"


class TestStdioConnection:
    def test_sets_transport(self):
        assert stdio_connection("gh")["transport"] == "stdio"

    def test_sets_command(self):
        assert stdio_connection("gh")["command"] == "gh"

    def test_args_included_when_provided(self):
        conn = stdio_connection("gh", args=["mcp", "serve"])
        assert conn["args"] == ["mcp", "serve"]

    def test_args_omitted_when_not_provided(self):
        assert "args" not in stdio_connection("gh")

    def test_env_included_when_provided(self):
        conn = stdio_connection("gh", env={"TOKEN": "x"})
        assert conn["env"] == {"TOKEN": "x"}

    def test_env_omitted_when_not_provided(self):
        assert "env" not in stdio_connection("gh")
