"""Typed decision functions with ``@decisive``, on a local Ollama model.

Run it::

    ollama pull tev1 && ollama serve
    python examples/quickstart.py
    python examples/quickstart.py --model tev1:0.8b --host http://gpu-node:11434

What this shows
---------------
Each stub below is a different everyday classification task.  The return
annotation picks the judge primitive, so the stub body stays empty:

=================================  ====================  ==================
Stub                               Return annotation     Judge primitive
=================================  ====================  ==================
``route_intent``                   ``Literal[...]``      ``choose``
``check_grounding``                ``StrEnum``           ``choose``
``contains_pii``                   ``bool``              ``truth`` + cutoff
``toxicity``                       ``float``             ``truth`` (raw)
``review_sentiment``               ``Literal`` + rubric  ``rate``
=================================  ====================  ==================

Every call can return a :class:`Verdict`, so the caller decides what a
low-confidence answer means: accept it, send it for review, or fall back to
an LLM.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
from typing import Literal

from mellea_contribs.systemone import OllamaJudge, decisive

# ---------------------------------------------------------------------------
# Decision functions.  No bodies: the signature and docstring are the prompt.
# ---------------------------------------------------------------------------


@decisive
def route_intent(
    message: str,
) -> Literal["order_status", "refund", "product_question", "complaint", "other"]:
    """Which intent best describes this customer chat message."""


class Grounding(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_MENTIONED = "not_mentioned"


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


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

MESSAGES = [
    "Where is my package? Tracking hasn't updated since Monday.",
    "The blender arrived cracked. I want my money back.",
    "Does the X200 work with 220V outlets?",
    "hmm",
]

CONTEXT = (
    "The Model X200 blender has a 1.5 L jar, a 900 W motor, and a two-year "
    "warranty. It is rated for 110-120 V outlets only."
)

#: Below this confidence the demo flags the answer for review.  Tune it on
#: your own labelled data; the right value depends on the model and the task.
REVIEW_BELOW = 0.75


def conf(confidence: float | None) -> str:
    """Format a confidence; some providers omit it for some answer types."""
    return " n/a" if confidence is None else f"{confidence:.2f}"


def flag(confidence: float | None) -> str:
    return "" if (confidence or 0.0) >= REVIEW_BELOW else "  <- review"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama URL (default: $OLLAMA_HOST or localhost:11434)",
    )
    parser.add_argument(
        "--model", default="tev1", help="tev1, tev1:0.8b, nimble, clef, clef-flash"
    )
    args = parser.parse_args()

    judge = OllamaJudge(host=args.host, model=args.model)

    print("Intent routing (Literal -> choose)")
    for msg in MESSAGES:
        v = route_intent.verdict(judge, message=msg)
        print(f"  {v.value:<17} {conf(v.confidence)}  {msg!r}{flag(v.confidence)}")

    print("\nRAG grounding check (StrEnum -> choose)")
    for answer in (
        "The X200 has a 900 W motor.",
        "The X200 works on 220 V outlets.",
        "The X200 is dishwasher safe.",
    ):
        v = check_grounding.verdict(judge, context=CONTEXT, answer=answer)
        print(
            f"  {v.value.value:<17} {conf(v.confidence)}  {answer!r}{flag(v.confidence)}"
        )

    print("\nPII check (bool, threshold=0.7)")
    for text in (
        "Ship it to Jane Doe, 42 Elm St, Springfield. Call 555-0134.",
        "Ship it to the address on my account.",
    ):
        v = contains_pii.verdict(judge, text=text)
        # For a bool stub, confidence holds the raw truth score behind the cutoff.
        print(f"  {v.value!s:<17} {conf(v.confidence)}  {text!r}")

    print("\nToxicity score (float -> raw truth score)")
    for comment in (
        "Thanks, this fixed it for me!",
        "You are an idiot and should quit.",
    ):
        score = toxicity(judge, comment=comment)
        print(f"  {score:<17.2f}       {comment!r}")

    print("\nReview sentiment (Literal + rubric -> rate)")
    for review in (
        "Loud, but crushes ice in seconds. Worth it.",
        "Stopped working after a week. Support never replied.",
    ):
        v = review_sentiment.verdict(judge, review=review)
        print(f"  {v.value:<17} {conf(v.confidence)}  {review!r}{flag(v.confidence)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
