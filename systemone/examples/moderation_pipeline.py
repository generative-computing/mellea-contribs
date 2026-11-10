"""A content-moderation pipeline: several decisions per post, one round trip.

Run it::

    ollama pull tev1 && ollama serve
    python examples/moderation_pipeline.py

What this shows
---------------
- ``Judge.batch`` asks every question about one post in a single API call.
  Calling three ``@decisive`` stubs would cost three calls instead.
- Confidence drives the action.  Clear cases are handled automatically, and
  only the unsure ones go to a person (or to a larger ``@generative`` model).
"""

from __future__ import annotations

import argparse

from mellea_contribs.systemone import ChoiceQ, OllamaJudge, RateQ, TruthQ, Verdict

QUESTIONS = {
    "category": ChoiceQ(
        options={
            "ok": "ordinary, on-topic discussion",
            "spam": "advertising, scams, or link farming",
            "harassment": "insults or attacks aimed at a person",
            "self_harm": "a person talking about hurting themselves",
        },
        instructions="Which category does this forum post belong to?",
    ),
    "has_link": TruthQ(
        instructions="Does the post ask readers to visit an external link?"
    ),
    "severity": RateQ(
        rubric=["none", "low", "high"],
        instructions="How much harm could this post cause if left up?",
    ),
}

POSTS = [
    "Has anyone tried the new firmware? My battery life doubled.",
    "EARN $5000/WEEK FROM HOME!!! click bit.ly/xx-cash now",
    "You clearly have no idea what you're talking about, as usual.",
    "I don't really see the point of anything lately.",
]

#: Act automatically only above this confidence.  Placeholder value: tune it
#: on labelled posts from your own forum.
AUTO_ACT_ABOVE = 0.85


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def decide(answers: dict[str, Verdict]) -> str:
    """Turn the batch answers for one post into a moderation action."""
    category = answers["category"]
    # A possible self-harm post goes to people however unsure the model is.
    if category.value == "self_harm":
        return "escalate to safety team"
    if (category.confidence or 0.0) < AUTO_ACT_ABOVE:
        return "human review (unsure)"
    if category.value == "ok":
        return "approve"
    if answers["severity"].value == "high":
        return "remove"
    return "hide pending review"


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

    for post in POSTS:
        answers = judge.batch(state=post, questions=QUESTIONS)
        category = answers["category"]
        print(f"{post!r}")
        print(
            f"  category={category.value} ({fmt(category.confidence)})  "
            f"has_link={fmt(answers['has_link'].value)}  "
            f"severity={answers['severity'].value}  "
            f"latency={category.latency_ms:.0f}ms"
        )
        print(f"  -> {decide(answers)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
