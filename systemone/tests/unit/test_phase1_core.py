"""Unit tests for Phase 1: core protocols, FakeJudge, and error types.

All tests in this file are marked ``unit`` and run with no network access,
no torch, and no provider SDK installed.  Only ``FakeJudge`` is used.

Test sections
-------------
1. Verdict dataclass — immutability, generic typing.
2. Question tagged union — isinstance dispatch.
3. FakeJudge.choose — scripted queue, call recording, queue-empty error.
4. FakeJudge.truth — same shape.
5. FakeJudge.batch — mixed-type questions drain correct per-type queues.
6. Protocol conformance — isinstance checks against Judge.
7. Error types — JudgeUnavailable, UnsupportedReturnType are raise-able.
"""

from __future__ import annotations

import pytest

from mellea_contribs.systemone.backends.fake import FakeJudge
from mellea_contribs.systemone.core.errors import (
    JudgeUnavailable,
    UnsupportedReturnType,
)
from mellea_contribs.systemone.core.judge import (
    ChoiceQ,
    Judge,
    RateQ,
    TruthQ,
    Verdict,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Verdict
# ---------------------------------------------------------------------------


def test_verdict_is_frozen() -> None:
    v: Verdict[str] = Verdict(
        value="ok",
        confidence=0.9,
        probabilities={"ok": 0.9, "bad": 0.1},
        provider="fake",
        latency_ms=1.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        v.value = "changed"  # type: ignore[misc]


def test_verdict_none_confidence_allowed() -> None:
    v: Verdict[float] = Verdict(
        value=0.7,
        confidence=None,
        probabilities=None,
        provider="fake",
        latency_ms=0.5,
    )
    assert v.confidence is None
    assert v.probabilities is None


def test_verdict_generic_type() -> None:
    v_str: Verdict[str] = Verdict(
        value="a", confidence=1.0, probabilities=None, provider="fake", latency_ms=0.0
    )
    v_float: Verdict[float] = Verdict(
        value=0.42, confidence=None, probabilities=None, provider="fake", latency_ms=0.0
    )
    assert isinstance(v_str.value, str)
    assert isinstance(v_float.value, float)


# ---------------------------------------------------------------------------
# 2. Question tagged union
# ---------------------------------------------------------------------------


def test_choice_q_fields() -> None:
    q = ChoiceQ(options={"a": "option a", "b": None}, instructions="Pick one")
    assert isinstance(q, ChoiceQ)
    assert q.options == {"a": "option a", "b": None}


def test_truth_q_fields() -> None:
    q = TruthQ(instructions="Is the claim supported?")
    assert isinstance(q, TruthQ)


def test_rate_q_fields() -> None:
    q = RateQ(rubric=["1", "2", "3", "4", "5"], instructions="Rate quality")
    assert isinstance(q, RateQ)
    assert q.rubric == ["1", "2", "3", "4", "5"]


# ---------------------------------------------------------------------------
# 3. FakeJudge.choose
# ---------------------------------------------------------------------------


def _make_choose_verdict(value: str, confidence: float = 0.9) -> Verdict[str]:
    return Verdict(
        value=value,
        confidence=confidence,
        probabilities={value: confidence},
        provider="fake",
        latency_ms=1.0,
    )


def test_fake_choose_returns_scripted_sequence() -> None:
    judge = FakeJudge(
        choose_verdicts=[
            _make_choose_verdict("supports", 0.92),
            _make_choose_verdict("contradicts", 0.88),
        ]
    )
    v1 = judge.choose(
        state="text",
        options={"supports": None, "contradicts": None},
        instructions="assess",
    )
    v2 = judge.choose(
        state="text",
        options={"supports": None, "contradicts": None},
        instructions="assess",
    )
    assert v1.value == "supports"
    assert v1.confidence == 0.92
    assert v2.value == "contradicts"
    assert v2.confidence == 0.88


def test_fake_choose_records_calls() -> None:
    judge = FakeJudge(choose_verdicts=[_make_choose_verdict("a")])
    judge.choose(state="doc", options={"a": "desc a"}, instructions="pick")
    assert len(judge.calls) == 1
    assert judge.calls[0].method == "choose"
    assert judge.calls[0].kwargs["state"] == "doc"
    assert judge.calls[0].kwargs["instructions"] == "pick"


def test_fake_choose_empty_queue_raises() -> None:
    judge = FakeJudge(choose_verdicts=[])
    with pytest.raises(ValueError, match="no more scripted verdicts"):
        judge.choose(state="x", options={}, instructions="pick")


# ---------------------------------------------------------------------------
# 4. FakeJudge.truth
# ---------------------------------------------------------------------------


def _make_truth_verdict(value: float) -> Verdict[float]:
    return Verdict(
        value=value,
        confidence=value,
        probabilities=None,
        provider="fake",
        latency_ms=1.0,
    )




# ---------------------------------------------------------------------------
# 5. FakeJudge.batch — mixed questions drain correct per-type queues
# ---------------------------------------------------------------------------


def test_fake_batch_mixed_questions() -> None:
    judge = FakeJudge(
        choose_verdicts=[_make_choose_verdict("billing", 0.91)],
        truth_verdicts=[_make_truth_verdict(0.05)],
    )
    questions = {
        "team": ChoiceQ(
            options={"billing": None, "bug": None}, instructions="pick team"
        ),
        "is_urgent": TruthQ(instructions="Is this urgent?"),
    }
    results = judge.batch(state="ticket text", questions=questions)
    assert results["team"].value == "billing"
    assert results["is_urgent"].value == pytest.approx(0.05)
    # One batch call recorded, not two individual calls
    assert len([c for c in judge.calls if c.method == "batch"]) == 1


def test_fake_batch_records_single_call() -> None:
    judge = FakeJudge(
        choose_verdicts=[_make_choose_verdict("a")],
        truth_verdicts=[_make_truth_verdict(0.9)],
    )
    judge.batch(
        state="x",
        questions={
            "q1": ChoiceQ(options={"a": None}, instructions="pick"),
            "q2": TruthQ(instructions="true?"),
        },
    )
    assert len(judge.calls) == 1
    assert judge.calls[0].method == "batch"


def test_fake_batch_empty_choose_queue_raises() -> None:
    judge = FakeJudge(choose_verdicts=[])
    with pytest.raises(ValueError, match="choose verdicts"):
        judge.batch(
            state="x",
            questions={"q": ChoiceQ(options={"a": None}, instructions="pick")},
        )


def test_fake_batch_unknown_question_type_raises() -> None:
    judge = FakeJudge()
    with pytest.raises(TypeError, match="unknown question type"):
        judge.batch(state="x", questions={"q": "not-a-question-type"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. Protocol conformance
# ---------------------------------------------------------------------------


def test_fake_judge_is_judge() -> None:
    judge = FakeJudge()
    assert isinstance(judge, Judge)


def test_fake_judge_name() -> None:
    judge = FakeJudge()
    assert judge.name == "fake"


# ---------------------------------------------------------------------------
# 7. Error types
# ---------------------------------------------------------------------------


def test_judge_unavailable_is_runtime_error() -> None:
    exc = JudgeUnavailable("missing SDK")
    assert isinstance(exc, RuntimeError)
    assert "missing SDK" in str(exc)


def test_unsupported_return_type_is_type_error() -> None:
    exc = UnsupportedReturnType("list[str] not supported")
    assert isinstance(exc, TypeError)
    assert "list[str]" in str(exc)


