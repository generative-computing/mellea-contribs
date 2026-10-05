"""Unit tests for Phase 4: JevJudge.

Hermetic — a ``StubClient`` stands in for ``TypeSafeClient``, so these run with
no API key and no network.  Because ``typesafe-sdk`` is a public PyPI package,
the real question *types* (``Noul``/``Choice``/``Score``) are used and
type-checked here; only live calls are out of scope (see ``tests/e2e/``).

The Jev response shape these tests pin (verified against typesafe-sdk 0.7.1):

- ``SystemOneResponse.answers`` is one flat ``dict[str, Answer]``.
- ``NoulAnswer`` has ``noul: float`` and **no** confidence field — the noul
  *is* the calibrated value.
- ``ChoiceAnswer`` has ``choice: str``, ``confidence: float``,
  ``probabilities: dict[str, float]``.
- ``ScoreAnswer`` has ``score: float``, ``confidence``, ``probabilities``
  keyed by **int**, and ``legend: dict[int, str]``.

Test sections
-------------
1. Construction — SDK absence, injected client, name tag.
2. Capability boundary — implements Judge.
3. choose — Choice question, ChoiceAnswer mapping.
4. truth — Noul question, noul-as-confidence.
5. rate — Score question, float+legend back to a rubric label.
6. batch — one system_one call for N questions.
7. Error translation — SDK errors become JudgeUnavailable.
"""

from __future__ import annotations

from typing import Any

import pytest

from mellea_contribs.systemone.core.errors import JudgeUnavailable
from mellea_contribs.systemone.core.judge import (
    ChoiceQ,
    Judge,
    RateQ,
    TruthQ,
    Verdict,
)

pytestmark = pytest.mark.unit

typesafe_sdk = pytest.importorskip("typesafe_sdk", reason="needs the [jev] extra")


# ---------------------------------------------------------------------------
# Stubs built from the real SDK response models
# ---------------------------------------------------------------------------


def _noul(value: float) -> Any:
    return typesafe_sdk.NoulAnswer(noul=value)


def _choice(
    value: str, confidence: float, probabilities: dict[str, float] | None = None
) -> Any:
    return typesafe_sdk.ChoiceAnswer(
        choice=value,
        confidence=confidence,
        probabilities=probabilities or {value: confidence},
    )


def _score(value: float, confidence: float, legend: dict[int, str]) -> Any:
    return typesafe_sdk.ScoreAnswer(
        score=value,
        confidence=confidence,
        legend=legend,
        probabilities={k: 1.0 / len(legend) for k in legend},
    )


class StubClient:
    """Records ``system_one`` calls and returns a canned response."""

    def __init__(
        self, answers: dict[str, Any] | None = None, raises: Exception | None = None
    ):
        self._answers = answers or {}
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def system_one(self, state: Any, questions: Any, **kw: Any) -> Any:
        self.calls.append({"state": state, "questions": questions, **kw})
        if self._raises is not None:
            raise self._raises
        return typesafe_sdk.SystemOneResponse(
            model="jev-1",
            usage=typesafe_sdk.Usage(input_tokens=10, output_tokens=0),
            answers=self._answers,
        )


def _judge(answers: dict[str, Any] | None = None, raises: Exception | None = None):
    from mellea_contribs.systemone.backends.jev import JevJudge

    return JevJudge(client=StubClient(answers, raises))


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


def test_name_is_jev() -> None:
    assert _judge().name == "jev"


def test_accepts_injected_client() -> None:
    from mellea_contribs.systemone.backends.jev import JevJudge

    stub = StubClient()
    assert JevJudge(client=stub).client is stub


def test_missing_api_key_raises_judge_unavailable(monkeypatch) -> None:
    """No key and no client is a clear configuration error, not a crash later."""
    from mellea_contribs.systemone.backends.jev import JevJudge

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JudgeUnavailable, match="TYPESAFE_API_KEY"):
        JevJudge()


# ---------------------------------------------------------------------------
# 2. Capability boundary
# ---------------------------------------------------------------------------


def test_is_a_judge() -> None:
    assert isinstance(_judge(), Judge)


# ---------------------------------------------------------------------------
# 3. choose
# ---------------------------------------------------------------------------


def test_choose_maps_choice_answer() -> None:
    judge = _judge(
        {"_q": _choice("supports", 0.93, {"supports": 0.93, "contradicts": 0.07})}
    )
    v = judge.choose(
        state="The sky is blue.",
        options={"supports": "backs the claim", "contradicts": "refutes it"},
        instructions="How does the text relate?",
    )
    assert isinstance(v, Verdict)
    assert v.value == "supports"
    assert v.confidence == pytest.approx(0.93)
    assert v.probabilities == {"supports": 0.93, "contradicts": 0.07}
    assert v.provider == "jev"
    assert v.latency_ms >= 0


def test_choose_sends_a_choice_question_with_criteria() -> None:
    judge = _judge({"_q": _choice("a", 0.5)})
    judge.choose(
        state="x", options={"a": "means A", "b": "means B"}, instructions="pick"
    )
    questions = judge.client.calls[0]["questions"]
    question = next(iter(questions.values()))
    assert isinstance(question, typesafe_sdk.Choice)
    assert dict(question.criteria) == {"a": "means A", "b": "means B"}
    assert question.instructions == "pick"



# ---------------------------------------------------------------------------
# 4. truth — the noul IS the calibrated value
# ---------------------------------------------------------------------------


def test_truth_confidence_is_the_noul_itself() -> None:
    """NoulAnswer has no separate confidence field; the noul is calibrated."""
    judge = _judge({"_q": _noul(0.87)})
    v = judge.truth(state="x", instructions="true?")
    assert v.confidence == pytest.approx(0.87)



# ---------------------------------------------------------------------------
# 5. rate — float score + int-keyed legend back to a rubric label
# ---------------------------------------------------------------------------


RUBRIC = ["poor", "fair", "good"]
LEGEND = {0: "poor", 1: "fair", 2: "good"}


def test_rate_returns_a_rubric_label_not_a_float() -> None:
    """Verdict[str] is the protocol's contract; ScoreAnswer.score is a float."""
    judge = _judge({"_q": _score(1.0, 0.8, LEGEND)})
    v = judge.rate(state="x", rubric=RUBRIC, instructions="rate")
    assert v.value in RUBRIC
    assert v.value == "fair"


def test_rate_rounds_to_the_nearest_rubric_entry() -> None:
    judge = _judge({"_q": _score(1.7, 0.7, LEGEND)})
    v = judge.rate(state="x", rubric=RUBRIC, instructions="rate")
    assert v.value == "good"


def test_rate_clamps_out_of_range_scores() -> None:
    """A score outside the legend must not raise IndexError."""
    judge = _judge({"_q": _score(99.0, 0.6, LEGEND)})
    v = judge.rate(state="x", rubric=RUBRIC, instructions="rate")
    assert v.value == "good"


def test_rate_probabilities_are_relabelled_from_int_keys() -> None:
    """Jev keys score probabilities by index; callers think in rubric labels."""
    judge = _judge({"_q": _score(1.0, 0.8, LEGEND)})
    v = judge.rate(state="x", rubric=RUBRIC, instructions="rate")
    assert v.probabilities is not None
    assert set(v.probabilities) == set(RUBRIC)


# ---------------------------------------------------------------------------
# 6. batch — one round trip
# ---------------------------------------------------------------------------


def test_batch_issues_one_system_one_call() -> None:
    judge = _judge(
        {
            "team": _choice("billing", 0.91),
            "urgent": _noul(0.7),
            "severity": _score(2.0, 0.6, LEGEND),
        }
    )
    results = judge.batch(
        state="I was charged twice, please help",
        questions={
            "team": ChoiceQ(
                options={"billing": None, "bug": None}, instructions="which team"
            ),
            "urgent": TruthQ(instructions="Is this urgent?"),
            "severity": RateQ(rubric=RUBRIC, instructions="how severe"),
        },
    )
    assert len(judge.client.calls) == 1
    assert results["team"].value == "billing"
    assert results["urgent"].value == pytest.approx(0.7)
    assert results["severity"].value == "good"


def test_batch_preserves_caller_keys() -> None:
    judge = _judge({"my_key": _noul(0.5)})
    results = judge.batch(state="x", questions={"my_key": TruthQ(instructions="true?")})
    assert set(results) == {"my_key"}


def test_batch_empty_makes_no_call() -> None:
    judge = _judge()
    assert judge.batch(state="x", questions={}) == {}
    assert judge.client.calls == []


def test_batch_unknown_question_type_raises() -> None:
    judge = _judge()
    with pytest.raises(TypeError, match="unknown question type"):
        judge.batch(state="x", questions={"q": "nope"})  # type: ignore[dict-item]


def test_batch_missing_answer_yields_none_verdict() -> None:
    """A key the API did not answer must not KeyError or silently vanish."""
    judge = _judge({})  # asked one question, got no answers
    results = judge.batch(state="x", questions={"q": TruthQ(instructions="true?")})
    assert results["q"].value is None
    assert results["q"].confidence is None


def test_batch_accepts_dict_state() -> None:
    """Jev takes JSON content natively, so a dict state needs no flattening."""
    judge = _judge({"q": _noul(0.5)})
    judge.batch(
        state={"ticket": "hi", "tier": "gold"},
        questions={"q": TruthQ(instructions="?")},
    )
    assert judge.client.calls[0]["state"] == {"ticket": "hi", "tier": "gold"}


# ---------------------------------------------------------------------------
# 7. Error translation
# ---------------------------------------------------------------------------


def _api_error(cls: type, message: str):
    """Build a real SDK API error; they require status/body/headers."""
    import httpx2

    return cls(
        status=401, body={"error": message}, headers=httpx2.Headers(), message=message
    )


def test_auth_error_becomes_judge_unavailable() -> None:
    judge = _judge(
        raises=_api_error(typesafe_sdk.TypeSafeAuthenticationError, "bad key")
    )
    with pytest.raises(JudgeUnavailable, match="TYPESAFE_API_KEY|authenticat"):
        judge.truth(state="x", instructions="true?")


def test_other_sdk_errors_propagate() -> None:
    """A rate limit is a transient condition the caller may want to retry."""
    judge = _judge(raises=_api_error(typesafe_sdk.TypeSafeRateLimitError, "slow down"))
    with pytest.raises(typesafe_sdk.TypeSafeRateLimitError):
        judge.truth(state="x", instructions="true?")
