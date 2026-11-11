"""An agent using its own tools, checked by Mellea requirements.

This is what ACPBackend offers that a plain LLM backend does not: the agent
reads files with its own tools, and a Mellea requirement plus rejection sampling
wraps the whole agent run. The requirement compares the reply against ground
truth computed in Python, so a hallucinated answer is rejected and retried.
Each retry is a fresh agent session, i.e. a full re-run. The ``on_update``
observer prints the agent's tool calls as they happen.

The agent runs in a scratch directory that this script fills with a few files.
``--trust --auto-approve`` lets Bob run tools without asking; Bob's own safety
rules still decide what is allowed. See https://bob.ibm.com/docs/shell/features/acp
and only use these flags in a directory you are happy to expose.

Usage:
    uv run python examples/108_agentic_tool_use.py

Prerequisites:
    - ``bob`` must be installed and on PATH, and authenticated (e.g. ``BOB_API_KEY``)
"""

import re
import tempfile
from pathlib import Path
from typing import Any

from acp.schema import ToolCallProgress, ToolCallStart
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.session import MelleaSession

from mellea_contribs.acp import ACPBackend

FILES = {"apples.txt": 3, "pears.txt": 7, "plums.txt": 12}


def lines_match_truth(reply: str) -> bool:
    """True if the reply gives the right line count for every file."""
    found = {
        name: int(count) for name, count in re.findall(r"(\w+\.txt)\W+(\d+)", reply)
    }
    return found == FILES


tool_titles: dict[str, str] = {}


def show_tool_calls(session_id: str, update: Any) -> None:
    """Print each tool call and its outcome (``on_update`` observer).

    A call is announced by a ``ToolCallStart`` carrying its title; later
    ``ToolCallProgress`` updates refer to it by id only, so remember the title.
    """
    if isinstance(update, ToolCallStart):
        tool_titles[update.tool_call_id] = update.title
        print(f"[{session_id[:8]}] tool: {update.title}")
    elif isinstance(update, ToolCallProgress) and update.status in (
        "completed",
        "failed",
    ):
        title = tool_titles.pop(update.tool_call_id, update.tool_call_id)
        print(f"[{session_id[:8]}]   {update.status}: {title}")


with tempfile.TemporaryDirectory(prefix="mellea-acp-demo-") as workdir:
    for name, n_lines in FILES.items():
        Path(workdir, name).write_text("".join(f"line {i}\n" for i in range(n_lines)))

    backend = ACPBackend(
        command=["bob", "acp", "--trust", "--auto-approve"],
        cwd=workdir,
        on_update=show_tool_calls,
    )
    m = MelleaSession(backend)

    result = m.instruct(
        "Count the lines in each .txt file in the current directory. "
        "Reply with exactly one line per file, in the form 'name.txt: N'.",
        requirements=[
            req(
                "Every .txt file is listed with its exact line count",
                validation_fn=simple_validate(lines_match_truth),
            )
        ],
        strategy=RejectionSamplingStrategy(loop_budget=3),
        return_sampling_results=True,
    )

    if result.success:
        print(f"***** reply (verified against the files) *****\n{result.result}")
    else:
        print(f"***** reply (NOT verified) *****\n{result.sample_generations[0].value}")

    backend.close()
