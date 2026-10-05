"""Run every example against a real GLiNER2 checkpoint and report results.

This is the integration smoke test that validates systemone works end-to-end
with real model weights — the seam the unit tests deliberately avoid.

Usage::

    # Inside a bsub GPU job or a machine with torch + GPU:
    python examples/run_all_real.py

    # Optionally specify a checkpoint:
    python examples/run_all_real.py --checkpoint fastino/gliner2.5-small-v1

Exit code is 0 if every check passes, 1 if any check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Literal

from mellea_contribs.systemone import (
    Gliner2Judge,
    decisive,
)
from mellea_contribs.systemone.core.judge import (
    ChoiceQ,
    Judge,
    RateQ,
    TruthQ,
)

PASS = 0
FAIL = 1


class Result:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.detail = ""
        self.elapsed_ms = 0.0

    def ok(self, detail: str = "", elapsed_ms: float = 0.0):
        self.passed = True
        self.detail = detail
        self.elapsed_ms = elapsed_ms
        return self

    def fail(self, detail: str, elapsed_ms: float = 0.0):
        self.passed = False
        self.detail = detail
        self.elapsed_ms = elapsed_ms
        return self


def section(title: str):
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")


def report(result: Result):
    tag = "PASS" if result.passed else "FAIL"
    ms = f"{result.elapsed_ms:.0f}ms" if result.elapsed_ms else ""
    print(f"  [{tag}] {result.name} {ms}")
    if result.detail:
        for line in result.detail.splitlines():
            print(f"         {line}")


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_conformance(judge: Gliner2Judge) -> Result:
    r = Result("Protocol conformance (Judge)")
    if not isinstance(judge, Judge):
        return r.fail("Gliner2Judge does not satisfy Judge protocol")
    return r.ok("Implements Judge")


# ---------------------------------------------------------------------------
# 2. Judge.choose
# ---------------------------------------------------------------------------

def test_choose(judge: Gliner2Judge) -> Result:
    r = Result("Judge.choose — pick from options")
    t0 = time.perf_counter()
    v = judge.choose(
        state="The Eiffel Tower is in Paris, France.",
        options={
            "supports": "the text confirms the claim",
            "contradicts": "the text refutes the claim",
            "says_nothing": "the text is silent on the claim",
        },
        instructions="Claim: the Eiffel Tower is in Paris.",
    )
    elapsed = (time.perf_counter() - t0) * 1000
    if v.value not in {"supports", "contradicts", "says_nothing"}:
        return r.fail(f"value={v.value!r} not in options", elapsed)
    if v.provider != "gliner2":
        return r.fail(f"provider={v.provider!r}", elapsed)
    return r.ok(
        f"value={v.value!r}  confidence={v.confidence}  latency={v.latency_ms:.0f}ms",
        elapsed,
    )


# ---------------------------------------------------------------------------
# 3. Judge.truth — with direction check
# ---------------------------------------------------------------------------

def test_truth(judge: Gliner2Judge) -> Result:
    r = Result("Judge.truth — boolean scoring with direction check")
    state = "The Eiffel Tower is in Paris, France."

    t0 = time.perf_counter()
    true_v = judge.truth(state=state, instructions="Is the Eiffel Tower in Paris?")
    false_v = judge.truth(state=state, instructions="Is the Eiffel Tower in Tokyo?")
    elapsed = (time.perf_counter() - t0) * 1000

    detail_lines = [
        f"true claim:  value={true_v.value}  confidence={true_v.confidence}",
        f"false claim: value={false_v.value}  confidence={false_v.confidence}",
    ]

    if true_v.value is None and false_v.value is None:
        return r.fail("Both verdicts returned None — model produced no output", elapsed)

    if true_v.value is not None and false_v.value is not None:
        if true_v.value <= false_v.value:
            detail_lines.append(
                "DIRECTION WRONG: true claim should score higher than false claim"
            )
            return r.fail("\n".join(detail_lines), elapsed)
        detail_lines.append("Direction correct: true_score > false_score")

    return r.ok("\n".join(detail_lines), elapsed)


# ---------------------------------------------------------------------------
# 4. Judge.rate
# ---------------------------------------------------------------------------

def test_rate(judge: Gliner2Judge) -> Result:
    r = Result("Judge.rate — ordinal rating")
    rubric = ["1", "2", "3", "4", "5"]
    t0 = time.perf_counter()
    v = judge.rate(
        state="A thorough, well-sourced summary with clear structure.",
        rubric=rubric,
        instructions="Rate the quality of this summary.",
    )
    elapsed = (time.perf_counter() - t0) * 1000
    if v.value not in rubric:
        return r.fail(f"value={v.value!r} not in rubric {rubric}", elapsed)
    return r.ok(f"value={v.value!r}  confidence={v.confidence}", elapsed)


# ---------------------------------------------------------------------------
# 5. Judge.batch
# ---------------------------------------------------------------------------

def test_batch(judge: Gliner2Judge) -> Result:
    r = Result("Judge.batch — multiple questions, one forward pass")
    questions = {
        "team": ChoiceQ(
            options={"billing": "a payment issue", "bug": "a software defect"},
            instructions="Which team should handle this?",
        ),
        "urgent": TruthQ(instructions="Is the customer asking for urgent help?"),
        "severity": RateQ(
            rubric=["low", "medium", "high"],
            instructions="Rate severity.",
        ),
    }

    t0 = time.perf_counter()
    results = judge.batch(
        state="I was charged twice this month. Please fix this immediately.",
        questions=questions,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    if set(results) != {"team", "urgent", "severity"}:
        return r.fail(f"Missing keys: got {set(results)}", elapsed)

    lines = []
    for key, v in results.items():
        lines.append(f"{key}: value={v.value!r}  confidence={v.confidence}")

    if results["team"].value not in {"billing", "bug", None}:
        return r.fail(f"team value={results['team'].value!r} unexpected\n" + "\n".join(lines), elapsed)

    return r.ok("\n".join(lines), elapsed)


# ---------------------------------------------------------------------------
# 6. @decisive
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


def test_decisive(judge: Gliner2Judge) -> Result:
    r = Result("@decisive — typed decision functions")
    ticket = "I was charged twice for the same invoice this month."

    t0 = time.perf_counter()
    tri = triage.verdict(judge=judge, ticket=ticket)
    urg = is_urgent.verdict(judge=judge, ticket=ticket)
    sev = severity.verdict(judge=judge, ticket=ticket)
    elapsed = (time.perf_counter() - t0) * 1000

    lines = [
        f"triage:    value={tri.value!r}  confidence={tri.confidence}",
        f"is_urgent: value={urg.value!r}  confidence={urg.confidence}",
        f"severity:  value={sev.value!r}  confidence={sev.confidence}",
    ]

    if tri.value not in {"billing", "bug", "feature", None}:
        return r.fail(f"triage returned unexpected value: {tri.value!r}\n" + "\n".join(lines), elapsed)

    if not isinstance(urg.value, bool) and urg.value is not None:
        return r.fail(f"is_urgent returned non-bool: {urg.value!r}\n" + "\n".join(lines), elapsed)

    if sev.value not in {"low", "medium", "high", None}:
        return r.fail(f"severity returned unexpected value: {sev.value!r}\n" + "\n".join(lines), elapsed)

    return r.ok("\n".join(lines), elapsed)


# ---------------------------------------------------------------------------
# 7. Run the shipped examples with --real
# ---------------------------------------------------------------------------

def test_example_scripts() -> list[Result]:
    """Run each shipped example with --real and check exit code."""
    import subprocess

    venv_python = sys.executable
    examples_dir = __file__.replace("run_all_real.py", "")
    scripts = [
        "decision_functions.py",
    ]
    results = []
    for script in scripts:
        r = Result(f"Example script: {script} --real")
        path = f"{examples_dir}{script}"
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [venv_python, path, "--real"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if proc.returncode == 0:
                r.ok(f"exit code 0\n{proc.stdout[-500:]}" if proc.stdout else "exit code 0", elapsed)
            else:
                r.fail(
                    f"exit code {proc.returncode}\n"
                    f"stdout: {proc.stdout[-300:]}\n"
                    f"stderr: {proc.stderr[-300:]}",
                    elapsed,
                )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            r.fail(f"{exc}", elapsed)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_protocol_conformance,
    test_choose,
    test_truth,
    test_rate,
    test_batch,
    test_decisive,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Integration smoke test for systemone + real GLiNER2")
    parser.add_argument("--checkpoint", default="fastino/gliner2.5-small-v1",
                        help="GLiNER2 checkpoint to load (default: small-v1)")
    parser.add_argument("--skip-examples", action="store_true",
                        help="Skip running the example scripts as subprocesses")
    args = parser.parse_args()

    print(f"Loading GLiNER2 checkpoint: {args.checkpoint}")
    t0 = time.perf_counter()
    try:
        judge = Gliner2Judge(checkpoint=args.checkpoint)
    except Exception as exc:
        print(f"FATAL: could not load checkpoint: {exc}")
        traceback.print_exc()
        return FAIL
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Checkpoint loaded in {load_ms:.0f}ms")

    results: list[Result] = []

    section("Core protocol tests")
    for test_fn in ALL_TESTS:
        try:
            r = test_fn(judge)
        except Exception as exc:
            r = Result(test_fn.__name__)
            r.fail(f"EXCEPTION: {exc}\n{traceback.format_exc()[-500:]}")
        report(r)
        results.append(r)

    if not args.skip_examples:
        section("Example scripts (--real)")
        for r in test_example_scripts():
            report(r)
            results.append(r)

    section("Summary")
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print(f"  {passed} passed, {failed} failed, {len(results)} total")
    total_ms = sum(r.elapsed_ms for r in results)
    print(f"  Total inference time: {total_ms:.0f}ms")

    if failed:
        print("\n  FAILED tests:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}")
        return FAIL

    print("\n  All tests passed!")
    return PASS


if __name__ == "__main__":
    raise SystemExit(main())
