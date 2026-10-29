# mellea-contribs-acp

Mellea backend that drives `m.instruct()` and `m.generate()` through any
[Agent Client Protocol (ACP)](https://agentclientprotocol.com) stdio agent.

Works out of the box with **Bob** (`bob acp`) and **Mistral Vibe** (`vibe-acp`),
or any other ACP-compatible stdio agent.

## How it works

```
MelleaSession  →  ACPBackend  →  spawn_agent_process(<command>)  →  ACP agent
```

On the first call, `ACPBackend` spawns the agent subprocess and opens an ACP
session. Every subsequent `_generate_from_context` call renders the Mellea
context into a plain-text prompt and sends it via `prompt()`. Streamed
`AgentMessageChunk` responses are buffered and returned as a `ModelOutputThunk`.

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

## Examples

| File | Description |
|---|---|
| `101_email_with_requirements.py` | `instruct` with string requirements (Bob) |
| `102_plain_generate.py` | Plain `instruct`, no requirements (Bob) |
| `103_multi_turn_chat.py` | Multi-turn — Bob remembers context |
| `104_custom_validation.py` | `req()` with `simple_validate()` lambda (Bob) |
| `105_vibe_plain_generate.py` | Plain `instruct` using Mistral Vibe |

```bash
uv run python examples/102_plain_generate.py
uv run python examples/105_vibe_plain_generate.py
```
