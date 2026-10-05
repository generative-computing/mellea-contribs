"""Typed decision functions with @decisive.

Run it::

    python examples/decision_functions.py           # scripted judge
    python examples/decision_functions.py --real    # real GLiNER2

What this shows
---------------
- ``@decisive`` turns a type-annotated stub into a judge call.  The return
  annotation *is* the schema: a ``Literal`` becomes a closed choice, a ``bool``
  becomes a thresholded truth score.
- ``.verdict()`` exposes the confidence alongside the answer, so the caller
  decides what to do with an unsure decision.
- Unsupported return annotations fail at **decoration** time, so a mis-typed
  stub cannot reach production.

.. warning::
    GLiNER2 confidence is uncalibrated (softmax over label logits).  Every
    threshold below is a placeholder; tune on your own labelled data.
"""

from __future__ import annotations

import sys
from typing import Literal

from mellea_contribs.systemone import (
    FakeJudge,
    UnsupportedReturnType,
    Verdict,
    decisive,
)

#: Below this confidence, a caller would route the decision for review.
REVIEW_BELOW = 0.80


# ---------------------------------------------------------------------------
# The decision functions.  No bodies — the annotations carry the contract.
# ---------------------------------------------------------------------------


@decisive
def triage(ticket: str) -> Literal["billing", "bug", "feature"]:
    """Route a support ticket to the team that owns it."""


@decisive
def is_urgent(ticket: str) -> bool:
    """Whether this ticket describes an active outage needing immediate action."""


@decisive(rubric=["low", "medium", "high"])
def severity(ticket: str) -> Literal["low", "medium", "high"]:
    """How severe the described problem is for the customer."""


# ---------------------------------------------------------------------------
# Scripted judge, so the example runs with no model download.
# ---------------------------------------------------------------------------


def _v(value, confidence, provider="fake"):
    return Verdict(
        value=value,
        confidence=confidence,
        probabilities=None,
        provider=provider,
        latency_ms=1.0,
    )


def scripted_judge() -> FakeJudge:
    """A judge whose answers make each point below visible."""
    return FakeJudge(
        choose_verdicts=[
            _v("billing", 0.94),  # section 1: triage
            _v("billing", 0.94),  # section 2, clear ticket: confident
            _v("bug", 0.41),  # section 2, ambiguous ticket: unsure
        ],
        truth_verdicts=[_v(0.88, 0.88)],  # is_urgent -> True
        rate_verdicts=[_v("high", 0.79)],  # severity
    )


# ---------------------------------------------------------------------------
# Walkthrough
# ---------------------------------------------------------------------------

TICKET = "I was charged twice for the same invoice this month."
CONFUSING = "It does the thing but not the other thing, please advise."


def main() -> int:
    """Run the walkthrough and return a process exit code."""
    real = "--real" in sys.argv
    if real:
        from mellea_contribs.systemone import Gliner2Judge

        judge = Gliner2Judge()
    else:
        judge = scripted_judge()

    print("=" * 68)
    print("1. @decisive — the return annotation is the schema")
    print("=" * 68)
    print(f"ticket: {TICKET}")

    v = triage.verdict(judge=judge, ticket=TICKET)
    print(f"  triage    -> {v.value!r:12} confidence={v.confidence}  via {v.provider}")

    urgent = is_urgent.verdict(judge=judge, ticket=TICKET)
    print(
        f"  is_urgent -> {urgent.value!s:12} confidence={urgent.confidence}"
        "   (bool = truth score thresholded at 0.5)"
    )

    sev = severity.verdict(judge=judge, ticket=TICKET)
    print(
        f"  severity  -> {sev.value!r:12} confidence={sev.confidence}  (rate, ordinal)"
    )

    print()
    print("=" * 68)
    print("2. .verdict() — the confidence travels with the answer")
    print("=" * 68)
    for label, text in (("clear", TICKET), ("ambiguous", CONFUSING)):
        result = triage.verdict(judge=judge, ticket=text)
        confident = result.confidence is not None and result.confidence >= REVIEW_BELOW
        action = "accept" if confident else "send for review"
        print(
            f"  {label:10} -> {result.value!r:12} confidence={result.confidence}  ({action})"
        )

    print()
    print("=" * 68)
    print("3. Bad annotations fail at decoration, not in production")
    print("=" * 68)
    try:

        @decisive
        def summarise(doc: str) -> str:
            """Free prose is a generation task, not a decision."""

    except UnsupportedReturnType as exc:
        print(f"  @decisive -> str raised UnsupportedReturnType:\n    {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
