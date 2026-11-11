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

Caveat: ``session_id`` is not linked to a Mellea call or sampling attempt. This
example runs one generation at a time, so the validator simply uses the most
recent session. With concurrent calls you would need another way to correlate.

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
from mellea.stdlib.requirements import check, req, simple_validate
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
    """Collects tool calls per ACP session, in the order sessions were opened."""

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

    def latest(self) -> list[ToolCall]:
        """Tool calls of the most recent session (valid for sequential use)."""
        if not self.sessions:
            return []
        return list(list(self.sessions.values())[-1].values())


recorder = Recorder()


def numbers_come_from_tools(reply: str) -> tuple[bool, str]:
    tool_text = " ".join(c.output for c in recorder.latest())
    seen = set(re.findall(r"\d+", tool_text))
    missing = sorted(set(re.findall(r"\d+", reply)) - seen)
    if missing:
        return False, f"numbers {missing} do not appear in any tool output"
    return True, "all numbers in the reply appear in tool output"


def no_shell(reply: str) -> tuple[bool, str]:
    shell = [c.title for c in recorder.latest() if c.kind == "execute"]
    if shell:
        return False, f"the agent used the shell: {shell[0]}"
    return True, "no shell commands were used"


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
                    validation_fn=simple_validate(numbers_come_from_tools),
                ),
                check(
                    "The agent does not use the shell",
                    validation_fn=simple_validate(no_shell),
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
