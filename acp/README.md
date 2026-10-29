# mellea-contribs-acp

Mellea backend that drives `m.instruct()` and `m.generate()` through any
[Agent Client Protocol (ACP)](https://agentclientprotocol.com) stdio agent.

Works out of the box with **Bob** (`bob acp`) and **Mistral Vibe** (`vibe-acp`),
or any other ACP-compatible stdio agent.

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

# Mistral Vibe — requires `pip install vibe-acp` and `vibe-acp --setup`
m = MelleaSession(ACPBackend(command=["vibe-acp"]))

# Bob with extra flags
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
- Tool-permission requests from the agent are rejected (`reject_once`, else cancelled). The backend runs headless.
  **This is not a sandbox**: agents decide for themselves what needs permission. In testing, Bob created a file in the working directory without asking. Treat the agent as having full access to `cwd`.
- `cwd` (default: the current directory) is the directory the agent works in. Pass `ACPBackend(cwd="/some/scratch/dir")` to confine it to a scratch directory.
- Responses with stop reason `cancelled` or `refusal`, and empty replies, raise `RuntimeError`.
- `generate_from_raw` runs each action in its own session, concurrently, and ignores the context.
- The agent's own rules and persona (e.g. a custom reply prefix configured in Bob) are applied to its replies.
- Tested with `mellea>=0.8.0`, Python 3.11, 3.12 and 3.13.

## Examples

| File | Description |
|---|---|
| `101_email_with_requirements.py` | `instruct` with string requirements (Bob) |
| `102_plain_generate.py` | Plain `instruct`, no requirements (Bob) |
| `103_multi_turn_chat.py` | Multi-turn — Bob remembers context |
| `104_custom_validation.py` | `req()` with `simple_validate()` lambda (Bob) |
| `105_vibe_plain_generate.py` | Plain `instruct` using Mistral Vibe |
| `106_structured_output.py` | `format=` with a Pydantic model + validating requirement (Bob) |

```bash
uv run python examples/102_plain_generate.py
uv run python examples/105_vibe_plain_generate.py
```
