"""Mellea requirements that look at what the agent did, not just what it said.

``on_update`` lets a requirement see the agent's tool calls. Here the callback
records, per ACP session, every tool call and its output text. Two requirements
then read that record:

1. Grounded reply: every number in the reply appears in some tool's output.
   The agent's own tools give the ground truth, so a made-up count is rejected.
2. No shell: no tool of kind ``execute`` was used. The prompt below tells the
   agent to use the shell, so this is expected to FAIL. It shows how a check on
   agent behaviour fails sampling and how the failure is reported. It is declared with ``check()`` so the agent is not told about it:
   a ``req()`` is rendered into the prompt, and the agent would simply comply.

Each rejection-sampling attempt is a fresh ACP session, i.e. a full agent re-run,
so the loop budget is 1: the example stops after the first (failing) attempt.

Matching a validated output to its tool calls: every result carries the id of
the ACP session that produced it in ``thunk._meta["acp_session_id"]``, which is
also the ``session_id`` passed to ``on_update``. The validators below are written
as ``validation_fn(ctx)`` (not ``simple_validate``) so they can read the output
from ``ctx.last_output()`` and look up exactly its tool calls. This stays correct
with concurrent calls.

Usage:
    uv run python examples/110_validate_with_tool_calls.py

Prerequisites:
    - ``bob`` must be installed and on PATH, and authenticated (e.g. ``BOB_API_KEY``)
"""

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acp.schema import TextContentBlock, ToolCallProgress, ToolCallStart
from mellea.core import Context, ValidationResult
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.session import MelleaSession

from mellea_contribs.acp import ACPBackend


@dataclass
class ToolCall:
    title: str
    kind: str | None
    output: str = ""


@dataclass
class Recorder:
    """Collects tool calls per ACP session."""

    sessions: dict[str, dict[str, ToolCall]] = field(default_factory=dict)

    def __call__(self, session_id: str, update: Any) -> None:
        calls = self.sessions.setdefault(session_id, {})
        if isinstance(update, ToolCallStart):
            calls[update.tool_call_id] = ToolCall(update.title, update.kind)
        elif isinstance(update, ToolCallProgress) and update.tool_call_id in calls:
            # Agents differ in ``raw_output``; the text ``content`` is portable.
            for item in update.content or []:
                block = getattr(item, "content", None)
                if isinstance(block, TextContentBlock):
                    calls[update.tool_call_id].output += block.text

    def for_output(self, ctx: Context) -> list[ToolCall]:
        """Tool calls of the session that produced the last output in ``ctx``."""
        output = ctx.last_output()
        session_id = output._meta.get("acp_session_id") if output else None
        return list(self.sessions.get(session_id, {}).values())


recorder = Recorder()


def numbers_come_from_tools(ctx: Context) -> ValidationResult:
    reply = str(ctx.last_output())
    tool_text = " ".join(c.output for c in recorder.for_output(ctx))
    seen = set(re.findall(r"\d+", tool_text))
    missing = sorted(set(re.findall(r"\d+", reply)) - seen)
    if missing:
        reason = f"numbers {missing} do not appear in any tool output"
        return ValidationResult(False, reason=reason)
    return ValidationResult(
        True, reason="all numbers in the reply appear in tool output"
    )


def no_shell(ctx: Context) -> ValidationResult:
    shell = [c.title for c in recorder.for_output(ctx) if c.kind == "execute"]
    if shell:
        return ValidationResult(False, reason=f"the agent used the shell: {shell[0]}")
    return ValidationResult(True, reason="no shell commands were used")


with tempfile.TemporaryDirectory(prefix="mellea-acp-demo-") as workdir:
    Path(workdir, "a.txt").write_text("one\ntwo\nthree\n")
    Path(workdir, "b.txt").write_text("x\ny\n")

    backend = ACPBackend(
        command=["bob", "acp", "--trust", "--auto-approve"],
        cwd=workdir,
        on_update=recorder,
    )
    m = MelleaSession(backend)

    try:
        result = m.instruct(
            "Use the shell (wc -l) to count the lines of each .txt file in the "
            "current directory. Reply with one 'name.txt: N' line per file.",
            requirements=[
                req(
                    "Numbers in the reply come from tool output",
                    validation_fn=numbers_come_from_tools,
                ),
                check(
                    "The agent does not use the shell",
                    validation_fn=no_shell,
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=1),
            return_sampling_results=True,
        )
    finally:
        backend.close()

print(f"\nsampling success: {result.success}")
for attempt, validations in enumerate(result.sample_validations, start=1):
    print(f"\nattempt {attempt}:")
    for requirement, outcome in validations:
        status = "PASS" if outcome.as_bool() else "FAIL"
        print(f"  {status}  {requirement.description}: {outcome.reason}")
