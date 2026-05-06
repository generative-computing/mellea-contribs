# Mellea MCP Integration

MCP tool discovery and integration for Mellea agents.

Bridges [MCP](https://modelcontextprotocol.io/) server tools into Mellea's native tool-calling
system. Connect to any MCP server and use its tools directly inside a Mellea agent — no changes
to Mellea core required.

## How it works

```
MCP Server → mcp.ClientSession → MCPToolSpec → MelleaTool (sync wrapper) → Mellea agent
```

Each tool call opens its own short-lived MCP session, so no session lifetime management is
required by the caller. The sync wrapper runs the async MCP call in a dedicated thread to
stay compatible with Mellea's internal event-loop model.

## Installation

```bash
uv pip install "git+https://github.com/generative-computing/mellea-contribs.git#subdirectory=mellea_contribs/mcp_tools"
```

To develop locally:

```bash
cd mellea_contribs/mcp_tools
uv sync
```

## Quick start

```python
import asyncio
from mellea import start_session
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea_mcp import discover_mcp_tools, http_connection

async def main():
    connection = http_connection("https://api.githubcopilot.com/mcp/", api_key="<api-key>")

    # Discover what the server offers, then pick only what you need
    specs = await discover_mcp_tools(connection)
    tools = [s.as_mellea_tool() for s in specs if s.name in {"get_me", "search_pull_requests"}]

    m = start_session()
    result, _ = await react(
        goal="Find my open pull requests and list each with its title, number, and repository.",
        context=ChatContext(),
        backend=m.backend,
        tools=tools,
    )
    print(result.value)

asyncio.run(main())
```

## API

### `discover_mcp_tools(connection) -> list[MCPToolSpec]`

Discover all tools on an MCP server. Opens a single session, fetches tool metadata, then
closes. No `MelleaTool` objects are created yet — inspect and filter before calling
`as_mellea_tool()`.

### `MCPToolSpec.as_mellea_tool() -> MelleaTool`

Instantiate a callable `MelleaTool` from a spec. The returned tool is ready to pass via
`ModelOption.TOOLS` or to `react()`.

### Connection helpers

```python
from mellea_mcp import http_connection, sse_connection, stdio_connection

http_connection(url, *, api_key=None, headers=None)   # Streamable HTTP
sse_connection(url, *, api_key=None, headers=None)    # SSE
stdio_connection(command, *, args=None, env=None)   # stdio subprocess
```

`api_key` sets `Authorization: Bearer <key>`. Use `headers` for anything else.

## Example

### `github_activity_summary.py`

Uses the hosted GitHub MCP server to summarise recent pull requests via Mellea's `react()` loop.

```bash
cd mellea_contribs/mcp_tools
export GITHUB_TOKEN=<token with repo + read:user scopes>
uv run python examples/github_activity_summary.py
```

## Running tests

```bash
cd mellea_contribs/mcp_tools
uv sync --all-extras
uv run python -m pytest tests/ -v
```

## Project structure

```
mcp_tools/
├── README.md
├── pyproject.toml
├── examples/
│   └── github_activity_summary.py
├── src/
│   └── mellea_mcp/
│       ├── __init__.py
│       ├── connections.py
│       └── tools.py
└── tests/
    ├── __init__.py
    └── test_tools.py
```

## License

Apache License 2.0 — see the [LICENSE](../../LICENSE) file for details.
