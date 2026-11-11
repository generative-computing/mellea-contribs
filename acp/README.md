# mellea-contribs-acp

Mellea backend that drives `m.instruct()` and `m.generate()` through any
[Agent Client Protocol (ACP)](https://agentclientprotocol.com) stdio agent.

Tested with **Bob** (`bob acp`) and **Mistral Vibe** (`vibe-acp`); other
ACP-compatible stdio agents (see the
[ACP agent list](https://agentclientprotocol.com/overview/agents)) should work via `command=`.

## How it works

```
MelleaSession  →  ACPBackend  →  spawn_agent_process(<command>)  →  ACP agent
```

On first use, `ACPBackend` spawns one agent subprocess and shares it. Every
generation opens its own fresh ACP session, renders the Mellea context into a
plain-text prompt, and sends it via `prompt()`. Streamed `AgentMessageChunk`
responses are buffered and resolved through a `ModelOutputThunk` (so Mellea
hooks and async streaming semantics apply). The session is closed afterwards,
so the agent keeps no hidden history: the Mellea context is the only memory, and
rejection-sampling retries do not pollute each other. Concurrent calls run in
parallel.

## Install

```bash
pip install -e .
```

## Usage

```python
from mellea.stdlib.session import MelleaSession
from mellea_contribs.acp import ACPBackend

# Bob (default) — requires `bob` on PATH with ACP support
m = MelleaSession(ACPBackend())

# Mistral Vibe — install the CLI from https://github.com/mistralai/mistral-vibe,
# then run `vibe-acp --setup`
m = MelleaSession(ACPBackend(command=["vibe-acp"]))

# Bob with extra flags (`--trust` trusts the workspace; see https://bob.ibm.com/docs/shell/features/acp)
m = MelleaSession(ACPBackend(command=["bob", "acp", "--trust"]))

result = m.instruct(
    "Write an email to invite all interns to the office party.",
    requirements=["be formal", "Use 'Dear Interns' as greeting."],
)
print(result)
```

## Limitations

- `model_options` and `tool_calls` are ignored (a warning is logged); the agent controls its own model and tools.
- `format=` is best-effort: the JSON schema is appended to the prompt and the first JSON value in the reply is extracted (agents often add prose, code fences or persona prefixes). ACP has no constrained decoding, so pair it with a validating sampling strategy.
- `@generative` works (it relies on `format=`), so the same best-effort caveat applies: use a `RejectionSamplingStrategy` for retries on malformed replies. Each `precondition_requirements` entry costs one extra agent session.
- Tool-permission requests from the agent are rejected (`reject_once`, else cancelled). The backend runs headless. Whether a tool call reaches this callback is up to the agent: with an agent's default settings, expect prompts that need tools to fail (Bob, for example, replies that it has no tool access). Agents usually have their own setting that lets tool calls run without asking (for Bob, `bob acp --auto-approve`). With it, the agent's own safety rules decide what runs, not this backend, so check the agent's docs before enabling it. See `108_agentic_tool_use.py`.
  **This is not a sandbox**: agents decide for themselves what needs permission. In testing, Bob created a file in the working directory without asking. Treat the agent as having full access to `cwd`.
- `cwd` (default: a fresh empty temporary directory, removed on `close()`/`aclose()`) is the directory the agent works in. Pass `ACPBackend(cwd=os.getcwd())` or any path to let the agent work on real files.
- `env` (default: a copy of the current environment) sets the agent's environment variables. The default lets agents read API keys such as `BOB_API_KEY`. Pass `ACPBackend(env={...})` to give the agent only those variables plus a minimal set (`PATH`, `HOME`, ...).
- `timeout` (default: 300 seconds) is the maximum time for agent startup (spawn + `initialize`) and, separately, for each generation. On expiry a `TimeoutError` is raised. Pass `timeout=None` to disable.
- `on_update(session_id, update)` (optional, sync or async) is called for every session update the agent sends: message and thought chunks, tool calls and their results, plans. `update` is the `acp.schema` object, and `session_id` identifies one generation, so concurrent calls can be told apart. Use it to show progress or log tool calls (see `108_agentic_tool_use.py`). Exceptions in the callback are logged and ignored. The backend does not otherwise keep tool calls or their output.
- If the agent subprocess dies, the call in flight fails with `ConnectionError` (it is not retried, since the agent may already have acted on the prompt) and the next call starts a fresh agent.
- Responses with stop reason `cancelled` or `refusal`, and empty replies, raise `RuntimeError`.
- `generate_from_raw` runs each action in its own session, concurrently, and ignores the context.
- The agent's own rules and persona (e.g. a custom reply prefix configured in Bob) are applied to its replies.
- Tested with `mellea>=0.8.0`, Python 3.11, 3.12 and 3.13.

## Examples

| File | Description |
|---|---|
| `101_email_with_requirements.py` | `instruct` with string requirements (Bob) |
| `102_plain_generate.py` | Plain `instruct`, no requirements (Bob) |
| `103_multi_turn_chat.py` | Multi-turn `m.chat()` with `ChatContext` (Bob) |
| `104_custom_validation.py` | `req()` with `simple_validate()` lambda (Bob) |
| `105_vibe_plain_generate.py` | Plain `instruct` using Mistral Vibe |
| `106_structured_output.py` | `format=` with a Pydantic model + validating requirement (Bob) |
| `107_generative.py` | `@generative` functions, with and without a sampling strategy (Bob) |
| `108_agentic_tool_use.py` | Agent uses its own tools (`--trust --auto-approve`); a requirement checks the reply against ground truth; `on_update` prints the tool calls |
| `109_observe_updates.py` | `on_update` timeline of everything the agent reports (tool calls, thoughts, usage); takes the agent command as argument (Bob, Vibe) |
| `110_validate_with_tool_calls.py` | Requirements that read tool calls and outputs collected by `on_update`; a hidden `check()` that fails on purpose |

```bash
uv run python examples/102_plain_generate.py
uv run python examples/105_vibe_plain_generate.py
```
