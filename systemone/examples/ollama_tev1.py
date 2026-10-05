"""Decision functions against a local Ollama tev1 model.

Run it::

    # Start Ollama with tev1 first:
    #   ollama pull tev1
    #   ollama serve

    python examples/ollama_tev1.py                           # localhost:11434
    python examples/ollama_tev1.py --host gpu-node:11434     # remote machine

What this shows
---------------
- ``OllamaJudge`` wraps the TypeSafe SDK pointed at Ollama's
  ``/v1/systemone`` endpoint — no env-var hacking needed.
- ``@decisive`` and ``Judge.batch`` work identically across GLiNER2, hosted
  Jev, and local Ollama/tev1 — the provider is injected, the decorator doesn't
  change.
- Clef models (multimodal) accept images via the ``Image`` annotation on
  ``@decisive`` parameters — images are base64-encoded and passed in a single
  API call alongside the questions.

Setup::

    pip install "mellea-contribs-systemone[ollama]"   # typesafe-sdk
    ollama pull tev1                                   # or tev1:0.8b for smaller
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Annotated, Literal

from mellea_contribs.systemone import OllamaJudge, decisive
from mellea_contribs.systemone.core.errors import JudgeUnavailable
from mellea_contribs.systemone.core.judge import ChoiceQ, Image, RateQ, TruthQ


# ---------------------------------------------------------------------------
# Decision stubs — identical to what you'd write for GLiNER2 or hosted Jev
# ---------------------------------------------------------------------------


@decisive
def triage(ticket: str) -> Literal["billing", "bug", "feature", "account"]:
    """Route a support ticket to the team that owns it."""


@decisive
def is_urgent(ticket: str) -> bool:
    """Whether this ticket needs immediate action."""


@decisive(rubric=["low", "medium", "high"])
def severity(ticket: str) -> Literal["low", "medium", "high"]:
    """How severe the customer's problem is."""


# ---------------------------------------------------------------------------
# Image-aware stubs — Clef multimodal models only
# ---------------------------------------------------------------------------


@decisive
def classify_photo(
    caption: str,
    photo: Annotated[bytes, Image()],
) -> Literal["indoor", "outdoor", "diagram", "screenshot"]:
    """Classify a photo into one of four scene categories."""


@decisive
def has_text(photo: Annotated[bytes, Image()]) -> bool:
    """Whether the image contains readable text."""


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

TICKETS = [
    (
        "billing",
        "I was charged twice for my subscription this month and the refund "
        "button returns a 500 error. I need this resolved before Friday.",
    ),
    (
        "bug",
        "The export button on the dashboard has been broken since the last "
        "update. It just spins and nothing downloads.",
    ),
    (
        "ambiguous",
        "It does the thing but not the other thing. Can someone look at this?",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama tev1 decision demo")
    parser.add_argument("--host", default="localhost:11434", help="Ollama host:port")
    parser.add_argument("--model", default="tev1", help="tev1, tev1:0.8b, clef, clef-flash")
    parser.add_argument("--image", type=pathlib.Path, help="Image file for Clef demo (section 3)")
    args = parser.parse_args()

    print(f"Connecting to Ollama at {args.host} with model {args.model}")
    print()

    try:
        judge = OllamaJudge(host=f"http://{args.host}", model=args.model)
    except JudgeUnavailable as exc:
        print(f"ERROR: {exc}")
        return 1

    # ---- 1. Single decisions with @decisive ----
    print("=" * 68)
    print("1. @decisive — same decorators, different provider")
    print("=" * 68)

    for label, ticket in TICKETS:
        v = triage.verdict(judge=judge, ticket=ticket)
        print(f"  [{label}] -> team={v.value!r}  confidence={v.confidence:.2f}  "
              f"latency={v.latency_ms:.0f}ms")

    print()
    v = is_urgent.verdict(judge=judge, ticket=TICKETS[0][1])
    print(f"  urgent?   -> {v.value}  (truth score: {v.confidence:.3f})")

    v = severity.verdict(judge=judge, ticket=TICKETS[0][1])
    print(f"  severity  -> {v.value!r}  confidence={v.confidence:.2f}")

    # ---- 2. Batched questions — one round trip ----
    print()
    print("=" * 68)
    print("2. Judge.batch — three questions, one API call")
    print("=" * 68)

    verdicts = judge.batch(
        state=TICKETS[0][1],
        questions={
            "team": ChoiceQ(
                options={
                    "billing": "a payment or refund problem",
                    "bug": "a software defect",
                    "feature": "a feature request",
                },
                instructions="Which team should handle this?",
            ),
            "urgent": TruthQ(instructions="Is this customer asking for urgent help?"),
            "severity": RateQ(
                rubric=["low", "medium", "high"],
                instructions="Rate the severity.",
            ),
        },
    )

    for key, v in verdicts.items():
        conf = f"{v.confidence:.2f}" if v.confidence is not None else "n/a"
        print(f"  {key:10s} = {str(v.value):8s}  confidence={conf}  "
              f"latency={v.latency_ms:.0f}ms")

    # ---- 3. Image support (Clef only) ----
    print()
    print("=" * 68)
    print("3. Image annotation with @decisive (Clef models only)")
    print("=" * 68)

    if args.image and args.image.exists():
        image_bytes = args.image.read_bytes()
        print(f"  Image: {args.image} ({len(image_bytes)} bytes)")

        v = classify_photo.verdict(
            judge=judge, caption="uploaded photo", photo=image_bytes
        )
        print(f"  classify_photo -> {v.value!r}  confidence={v.confidence:.2f}")

        v = has_text.verdict(judge=judge, photo=image_bytes)
        print(f"  has_text       -> {v.value}  (truth score: {v.confidence:.3f})")
    else:
        print("  Skipped — pass --image path/to/photo.jpg with a Clef model.")
        print("  Example: python examples/ollama_tev1.py --model clef-flash --image photo.jpg")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
