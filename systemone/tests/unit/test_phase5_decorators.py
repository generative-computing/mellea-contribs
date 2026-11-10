"""Unit tests for Phase 5: @decisive.

Hermetic — ``FakeJudge`` only.

The central behaviour under test is that **unsupported return annotations fail
at decoration time, not at call time**.  That is a deliberate contract: the type
restriction is documented, and an import-time failure makes it impossible to
ship code that only breaks in production.

Test sections
-------------
1. @decisive over Literal — maps to choose, returns the bare value.
2. @decisive over StrEnum.
3. @decisive over bool — maps to truth with a threshold.
4. @decisive over float — maps to truth, returns the raw score.
5. @decisive with rubric= — maps to rate.
6. .verdict() — exposes the full Verdict.
7. Instruction building — name, docstring and args reach the judge.
8. Decoration-time type enforcement.
9. A minimal Judge-only provider works with @decisive.
10. Metadata preservation — __name__, __doc__, signature.
"""

from __future__ import annotations

import enum
from typing import Literal

import pytest

from mellea_contribs.systemone.core.errors import UnsupportedReturnType
from mellea_contribs.systemone.core.judge import Verdict
from mellea_contribs.systemone.stdlib.components.decisive import decisive

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake(**kw):
    from mellea_contribs.systemone.backends.fake import FakeJudge

    return FakeJudge(**kw)


def _v(value, confidence=0.9, probabilities=None):
    return Verdict(
        value=value,
        confidence=confidence,
        probabilities=probabilities,
        provider="fake",
        latency_ms=1.0,
    )


class JudgeOnly:
    """A minimal hand-written Judge — stands in for JevJudge."""

    name = "judge-only"

    def choose(self, state, options, instructions, *, images=None):
        return _v(next(iter(options)))

    def truth(self, state, instructions, *, images=None):
        return _v(0.9)

    def rate(self, state, rubric, instructions, *, images=None):
        return _v(rubric[0])

    def batch(self, state, questions, *, images=None):
        return {}


# ---------------------------------------------------------------------------
# 1. @decisive over Literal
# ---------------------------------------------------------------------------


@decisive
def triage(ticket: str) -> Literal["billing", "bug", "feature"]:
    """Route a support ticket to the owning team."""


def test_literal_returns_the_bare_value() -> None:
    judge = _fake(choose_verdicts=[_v("billing")])
    assert triage(judge=judge, ticket="I was charged twice") == "billing"


def test_literal_maps_to_choose() -> None:
    judge = _fake(choose_verdicts=[_v("bug")])
    triage(judge=judge, ticket="it crashed")
    assert [c.method for c in judge.calls] == ["choose"]


def test_literal_options_come_from_the_annotation() -> None:
    judge = _fake(choose_verdicts=[_v("billing")])
    triage(judge=judge, ticket="x")
    assert set(judge.calls[0].kwargs["options"]) == {"billing", "bug", "feature"}


def test_judge_may_be_passed_positionally() -> None:
    """Mirrors @generative, which takes the session as the first positional."""
    judge = _fake(choose_verdicts=[_v("billing")])
    assert triage(judge, ticket="x") == "billing"


# ---------------------------------------------------------------------------
# 2. @decisive over StrEnum
# ---------------------------------------------------------------------------


class Team(enum.StrEnum):
    BILLING = "billing"
    BUG = "bug"


@decisive
def route(ticket: str) -> Team:
    """Route a ticket."""


def test_strenum_returns_an_enum_member() -> None:
    judge = _fake(choose_verdicts=[_v("billing")])
    result = route(judge=judge, ticket="x")
    assert result is Team.BILLING
    assert isinstance(result, Team)


def test_strenum_options_are_the_member_values() -> None:
    judge = _fake(choose_verdicts=[_v("bug")])
    route(judge=judge, ticket="x")
    assert set(judge.calls[0].kwargs["options"]) == {"billing", "bug"}


# ---------------------------------------------------------------------------
# 3. @decisive over bool
# ---------------------------------------------------------------------------


@decisive
def is_urgent(ticket: str) -> bool:
    """Whether the ticket needs urgent attention."""


def test_bool_above_threshold_is_true() -> None:
    judge = _fake(truth_verdicts=[_v(0.91, 0.91)])
    assert is_urgent(judge=judge, ticket="help now") is True


def test_bool_below_threshold_is_false() -> None:
    judge = _fake(truth_verdicts=[_v(0.12, 0.12)])
    assert is_urgent(judge=judge, ticket="whenever") is False


def test_bool_threshold_is_configurable() -> None:
    @decisive(threshold=0.95)
    def strict(ticket: str) -> bool:
        """Needs very high confidence."""

    judge = _fake(truth_verdicts=[_v(0.9, 0.9)])
    assert strict(judge=judge, ticket="x") is False


# ---------------------------------------------------------------------------
# 4. @decisive over float
# ---------------------------------------------------------------------------


@decisive
def urgency(ticket: str) -> float:
    """How urgent the ticket is, from 0 to 1."""


def test_float_returns_the_raw_score() -> None:
    judge = _fake(truth_verdicts=[_v(0.73, 0.73)])
    assert urgency(judge=judge, ticket="x") == pytest.approx(0.73)


# ---------------------------------------------------------------------------
# 5. @decisive with rubric= -> rate
# ---------------------------------------------------------------------------


@decisive(rubric=["low", "medium", "high"])
def severity(ticket: str) -> Literal["low", "medium", "high"]:
    """How severe the problem is."""


def test_rubric_maps_to_rate() -> None:
    judge = _fake(rate_verdicts=[_v("medium")])
    assert severity(judge=judge, ticket="x") == "medium"
    assert [c.method for c in judge.calls] == ["rate"]


def test_rubric_order_is_preserved() -> None:
    judge = _fake(rate_verdicts=[_v("low")])
    severity(judge=judge, ticket="x")
    assert judge.calls[0].kwargs["rubric"] == ["low", "medium", "high"]


def test_rubric_must_match_the_literal_values() -> None:
    """A rubric disagreeing with the annotation cannot produce a valid value."""
    with pytest.raises(UnsupportedReturnType, match="rubric"):

        @decisive(rubric=["a", "b"])
        def mismatched(x: str) -> Literal["p", "q"]:
            """Mismatched."""


# ---------------------------------------------------------------------------
# 6. .verdict()
# ---------------------------------------------------------------------------


def test_verdict_exposes_confidence_and_provider() -> None:
    judge = _fake(choose_verdicts=[_v("billing", 0.93)])
    v = triage.verdict(judge=judge, ticket="x")
    assert isinstance(v, Verdict)
    assert v.value == "billing"
    assert v.confidence == pytest.approx(0.93)
    assert v.provider == "fake"


def test_verdict_for_bool_keeps_the_raw_score_and_coerced_value() -> None:
    """The bool is the decision; the float is the evidence. Both are useful."""
    judge = _fake(truth_verdicts=[_v(0.42, 0.42)])
    v = is_urgent.verdict(judge=judge, ticket="x")
    assert v.value is False
    assert v.confidence == pytest.approx(0.42)



# ---------------------------------------------------------------------------
# 7. Instruction building
# ---------------------------------------------------------------------------


def test_instructions_include_the_docstring() -> None:
    judge = _fake(choose_verdicts=[_v("billing")])
    triage(judge=judge, ticket="x")
    assert "owning team" in judge.calls[0].kwargs["instructions"]


def test_instructions_include_the_function_name() -> None:
    judge = _fake(choose_verdicts=[_v("billing")])
    triage(judge=judge, ticket="x")
    assert "triage" in judge.calls[0].kwargs["instructions"]


def test_arguments_reach_the_judge_as_state() -> None:
    judge = _fake(choose_verdicts=[_v("billing")])
    triage(judge=judge, ticket="I was charged twice")
    assert "I was charged twice" in str(judge.calls[0].kwargs["state"])


def test_multiple_arguments_are_all_present() -> None:
    @decisive
    def compare(left: str, right: str) -> Literal["left", "right"]:
        """Which one is better."""

    judge = _fake(choose_verdicts=[_v("left")])
    compare(judge=judge, left="alpha", right="beta")
    state = str(judge.calls[0].kwargs["state"])
    assert "alpha" in state and "beta" in state


def test_missing_required_argument_raises_type_error() -> None:
    """The decorated function keeps its signature's arity contract."""
    judge = _fake(choose_verdicts=[_v("billing")])
    with pytest.raises(TypeError):
        triage(judge=judge)


# ---------------------------------------------------------------------------
# 8. Decoration-time type enforcement
# ---------------------------------------------------------------------------


def test_list_of_str_is_rejected_at_decoration() -> None:
    with pytest.raises(UnsupportedReturnType, match=r"list\[str\]|list"):

        @decisive
        def bad(x: str) -> list[str]:
            """Not a closed decision."""


def test_bare_str_is_rejected_and_points_at_generative() -> None:
    with pytest.raises(UnsupportedReturnType, match="generative"):

        @decisive
        def prose(x: str) -> str:
            """Free text is a generation task."""


def test_missing_annotation_is_rejected() -> None:
    with pytest.raises(UnsupportedReturnType, match="annotation|return"):

        @decisive
        def unannotated(x: str):
            """No return annotation."""


def test_non_string_literal_is_rejected() -> None:
    """GLiNER2 and Jev both label with strings; int labels cannot round-trip."""
    with pytest.raises(UnsupportedReturnType):

        @decisive
        def numeric(x: str) -> Literal[1, 2, 3]:
            """Integer labels."""


def test_error_message_names_the_offending_annotation() -> None:
    with pytest.raises(UnsupportedReturnType) as exc:

        @decisive
        def bad(x: str) -> dict[str, int]:
            """Nope."""

    assert "dict" in str(exc.value)


# ---------------------------------------------------------------------------
# 9. Judge-only provider
# ---------------------------------------------------------------------------


def test_decisive_accepts_a_judge_only_provider() -> None:
    assert triage(judge=JudgeOnly(), ticket="x") == "billing"


# ---------------------------------------------------------------------------
# 10. Metadata preservation
# ---------------------------------------------------------------------------


def test_name_is_preserved() -> None:
    assert triage.__name__ == "triage"


def test_docstring_is_preserved() -> None:
    assert triage.__doc__ is not None
    assert "owning team" in triage.__doc__


