"""Watch everything an ACP agent reports while it works, using ``on_update``.

``on_update(session_id, update)`` is called for every ``session/update`` the
agent sends. This example prints a compact timeline (tool calls, thoughts,
token usage, ...) and a summary of how many updates of each type arrived. What
you see depends on the agent: Bob reports tool calls and a session title; Mistral
Vibe also streams thoughts and token usage.

Tool-call ids are opaque (long encoded strings), so a title is remembered from
the ``ToolCallStart`` and reused when the later ``ToolCallProgress`` arrives.

Usage:
    uv run python examples/109_observe_updates.py              # Bob
    uv run python examples/109_observe_updates.py vibe-acp     # any ACP command

Prerequisites:
    - the agent must be installed and authenticated; for Bob, ``BOB_API_KEY``
"""

import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    SessionInfoUpdate,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)
from mellea.stdlib.session import MelleaSession

from mellea_contribs.acp import ACPBackend

# ``--trust --auto-approve`` lets Bob run tools without asking (see README).
command = sys.argv[1:] or ["bob", "acp", "--trust", "--auto-approve"]

counts: Counter[str] = Counter()
titles: dict[str, str] = {}


def short(text: str, width: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def observe(session_id: str, update: Any) -> None:
    counts[type(update).__name__] += 1
    tag = f"[{session_id[:8]}]"
    if isinstance(update, ToolCallStart):
        titles[update.tool_call_id] = update.title
        print(f"{tag} tool   {update.kind}: {short(update.title)}")
    elif isinstance(update, ToolCallProgress):
        if update.status in ("completed", "failed"):
            title = titles.get(update.tool_call_id, "?")
            print(f"{tag} tool   {update.status}: {short(title)}")
    elif isinstance(update, AgentThoughtChunk):
        pass  # many tiny chunks; only counted
    elif isinstance(update, UsageUpdate):
        print(f"{tag} usage  {update.used}/{update.size} tokens in context")
    elif isinstance(update, SessionInfoUpdate):
        print(f"{tag} title  {short(update.title or '')}")
    elif isinstance(update, AgentMessageChunk):
        print(f"{tag} reply  {short(getattr(update.content, 'text', ''))}")


with tempfile.TemporaryDirectory(prefix="mellea-acp-demo-") as workdir:
    Path(workdir, "a.txt").write_text("one\ntwo\nthree\n")
    Path(workdir, "b.txt").write_text("x\ny\n")

    backend = ACPBackend(command=command, cwd=workdir, on_update=observe)
    m = MelleaSession(backend)
    try:
        reply = m.instruct(
            "Use your tools to count the lines of each .txt file in the current "
            "directory, then reply with one 'name: N' line per file."
        )
    finally:
        backend.close()

print(f"\n***** reply *****\n{reply}")
print("\n***** updates received *****")
for name, n in counts.most_common():
    print(f"{n:4d}  {name}")
