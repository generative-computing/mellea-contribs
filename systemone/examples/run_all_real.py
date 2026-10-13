"""Smoke-test every Judge primitive and @decisive against a real provider.

This is the integration check the unit tests deliberately avoid: real model,
real API, no stubs.

Usage::

    python examples/run_all_real.py                                  # Ollama tev1
    python examples/run_all_real.py --model clef-flash --host http://gpu:11434
    python examples/run_all_real.py --provider gliner2                # local weights, needs [local]
    python examples/run_all_real.py --provider jev                    # hosted, needs TYPESAFE_API_KEY

Exit code is 0 if every check passes, 1 if any check fails.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from enum import StrEnum
from typing import Literal

from mellea_contribs.systemone import (
    ChoiceQ,
    Judge,
    RateQ,
    TruthQ,
    decisive,
)

EXAMPLES_DIR = pathlib.Path(__file__).parent

#: Example scripts that run on Ollama with no extra input.
OLLAMA_EXAMPLES = ["quickstart.py", "moderation_pipeline.py"]


class CheckFailed(Exception):
    """Raised by a check when the provider's answer is wrong or malformed."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


# ---------------------------------------------------------------------------
# Checks.  Each returns a one-line summary on success or raises CheckFailed.
# ---------------------------------------------------------------------------


def check_protocol(judge: Judge) -> str:
    expect(isinstance(judge, Judge), f"{type(judge).__name__} does not satisfy Judge")
    return f"{type(judge).__name__} implements Judge"


def check_choose(judge: Judge) -> str:
    options = {
        "order_status": "asking where an order is",
        "refund": "asking for money back",
        "product_question": "asking about a product's features",
    }
    v = judge.choose(
        state="The blender arrived cracked. I want my money back.",
        options=options,
        instructions="Which intent does this customer message express?",
    )
    expect(v.value in options, f"value={v.value!r} not in options")
    expect(
        v.provider == judge.name, f"provider={v.provider!r}, expected {judge.name!r}"
    )
    return f"value={v.value!r} confidence={v.confidence}"


def check_truth_direction(judge: Judge) -> str:
    # A true claim must score higher than a false one about the same text.
    # This catches an inverted yes/no mapping in a provider.
    context = "The X200 blender is rated for 110-120 V outlets only."
    yes = judge.truth(
        state=context, instructions="Is the X200 rated for 110 V outlets?"
    )
    no = judge.truth(state=context, instructions="Is the X200 rated for 220 V outlets?")
    expect(yes.value is not None and no.value is not None, "truth returned None")
    expect(yes.value > no.value, f"true claim {yes.value} <= false claim {no.value}")
    return f"true={yes.value:.2f} > false={no.value:.2f}"


def check_rate(judge: Judge) -> str:
    rubric = ["negative", "mixed", "positive"]
    v = judge.rate(
        state="Stopped working after a week. Support never replied.",
        rubric=rubric,
        instructions="Overall sentiment of this product review.",
    )
    expect(v.value in rubric, f"value={v.value!r} not in rubric")
    return f"value={v.value!r} confidence={v.confidence}"


def check_batch(judge: Judge) -> str:
    questions = {
        "category": ChoiceQ(
            options={"ok": "ordinary discussion", "spam": "advertising or scams"},
            instructions="Which category does this forum post belong to?",
        ),
        "has_link": TruthQ(instructions="Does the post ask readers to visit a link?"),
        "severity": RateQ(
            rubric=["none", "low", "high"], instructions="How harmful is this post?"
        ),
    }
    out = judge.batch(
        state="EARN $5000/WEEK FROM HOME!!! click bit.ly/xx-cash now",
        questions=questions,
    )
    expect(set(out) == set(questions), f"keys={sorted(out)}")
    expect(
        out["category"].value in {"ok", "spam"}, f"category={out['category'].value!r}"
    )
    return "  ".join(f"{k}={v.value!r}" for k, v in out.items())


class Grounding(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_MENTIONED = "not_mentioned"


@decisive
def route_intent(message: str) -> Literal["order_status", "refund", "product_question"]:
    """Which intent best describes this customer chat message."""


@decisive
def check_grounding(context: str, answer: str) -> Grounding:
    """Whether the answer is supported by, contradicted by, or absent from the context."""


@decisive(threshold=0.7)
def contains_pii(text: str) -> bool:
    """Whether the text contains personal data such as an email, phone number, or address."""


@decisive
def toxicity(comment: str) -> float:
    """Whether this comment is abusive, harassing, or hateful."""


@decisive(rubric=["negative", "mixed", "positive"])
def review_sentiment(review: str) -> Literal["negative", "mixed", "positive"]:
    """Overall sentiment of this product review."""


def check_decisive(judge: Judge) -> str:
    intent = route_intent(judge, message="Where is my package?")
    grounding = check_grounding(
        judge,
        context="The X200 has a 900 W motor.",
        answer="The X200 has a 900 W motor.",
    )
    pii = contains_pii(judge, text="Call me on 555-0134.")
    tox = toxicity(judge, comment="Thanks, this fixed it!")
    sentiment = review_sentiment(judge, review="Crushes ice in seconds. Worth it.")

    expect(
        intent in {"order_status", "refund", "product_question"},
        f"route_intent={intent!r}",
    )
    expect(
        isinstance(grounding, Grounding),
        f"check_grounding={grounding!r} is not a Grounding",
    )
    expect(isinstance(pii, bool), f"contains_pii={pii!r} is not a bool")
    expect(
        isinstance(tox, float) and 0.0 <= tox <= 1.0, f"toxicity={tox!r} not in [0, 1]"
    )
    expect(
        sentiment in {"negative", "mixed", "positive"},
        f"review_sentiment={sentiment!r}",
    )
    return (
        f"intent={intent!r} grounding={grounding.value!r} pii={pii} "
        f"toxicity={tox:.2f} sentiment={sentiment!r}"
    )


CHECKS: list[Callable[[Judge], str]] = [
    check_protocol,
    check_choose,
    check_truth_direction,
    check_rate,
    check_batch,
    check_decisive,
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def build_judge(args: argparse.Namespace) -> Judge:
    if args.provider == "ollama":
        from mellea_contribs.systemone import OllamaJudge

        return OllamaJudge(host=args.host, model=args.model)
    if args.provider == "gliner2":
        from mellea_contribs.systemone import Gliner2Judge

        return Gliner2Judge(checkpoint=args.checkpoint)
    from mellea_contribs.systemone import JevJudge

    return JevJudge(model=args.model)


def run(name: str, fn: Callable[[], str]) -> bool:
    t0 = time.perf_counter()
    try:
        detail, passed = fn(), True
    except CheckFailed as exc:
        detail, passed = str(exc), False
    except Exception as exc:  # noqa: BLE001 - a smoke test reports every failure
        detail, passed = (
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-500:]}",
            False,
        )
    ms = (time.perf_counter() - t0) * 1000
    print(f"  [{'PASS' if passed else 'FAIL'}] {name} ({ms:.0f}ms)")
    for line in detail.splitlines():
        print(f"         {line}")
    return passed


def run_example(script: str, args: argparse.Namespace) -> str:
    cmd = [sys.executable, str(EXAMPLES_DIR / script)]
    if args.host:
        cmd += ["--host", args.host]
    if args.model:
        cmd += ["--model", args.model]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    expect(proc.returncode == 0, f"exit {proc.returncode}\n{proc.stderr[-400:]}")
    return "exit 0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider", choices=["ollama", "gliner2", "jev"], default="ollama"
    )
    parser.add_argument("--host", default=None, help="Ollama URL (ollama only)")
    parser.add_argument(
        "--model", default=None, help="Model name (ollama/jev; default: provider's own)"
    )
    parser.add_argument(
        "--checkpoint",
        default="fastino/gliner2.5-small-v1",
        help="GLiNER2 checkpoint (gliner2 only)",
    )
    parser.add_argument(
        "--skip-examples", action="store_true", help="Do not run the example scripts"
    )
    args = parser.parse_args()

    print(f"Provider: {args.provider}")
    t0 = time.perf_counter()
    try:
        judge = build_judge(args)
    except Exception as exc:  # noqa: BLE001 - a smoke test reports every failure
        print(f"FATAL: could not build judge: {exc}")
        return 1
    print(f"Judge ready in {(time.perf_counter() - t0) * 1000:.0f}ms\n")

    results = [run(fn.__name__, lambda fn=fn: fn(judge)) for fn in CHECKS]

    # The example scripts are written for Ollama, so only run them there.
    if args.provider == "ollama" and not args.skip_examples:
        print()
        results += [
            run(script, lambda s=script: run_example(s, args))
            for script in OLLAMA_EXAMPLES
        ]

    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
