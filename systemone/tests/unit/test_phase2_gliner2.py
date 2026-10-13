"""Unit tests for Phase 2: Gliner2Judge.

Hermetic — no checkpoint download, no network, no real forward pass.  A
``StubModel`` stands in for the GLiNER2 extractor object, letting these tests
assert on exactly which underlying GLiNER2 method the adapter calls and how it
maps the raw return shape onto :class:`Verdict`.

Real-checkpoint coverage lives in ``tests/integration/`` behind the
``integration`` marker.

Test sections
-------------
1. Construction — injected model, lazy import error, name tag.
2. choose   — maps to ``classify_text``, single-label.
3. truth    — maps to ``classify_text``, yes/no -> float.
4. rate     — maps to ``classify_text`` over the rubric.
5. batch    — one composed ``extract`` call, not N calls.
6. Protocol conformance — satisfies Judge.
7. Confidence semantics — uncalibrated probabilities surfaced as-is.
"""

from __future__ import annotations

from typing import Any

import pytest

from mellea_contribs.systemone.core.judge import (
    ChoiceQ,
    Judge,
    RateQ,
    TruthQ,
    Verdict,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stub standing in for a loaded GLiNER2 extractor
# ---------------------------------------------------------------------------


class StubModel:
    """Records calls and returns canned GLiNER2-shaped results.

    Return shapes mirror the real library with ``include_confidence=True``:

    - ``classify_text`` -> ``{task: {"label": str, "confidence": float}}``
    """

    def __init__(self, **canned: Any) -> None:
        self.canned = canned
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def classify_text(self, text: str, tasks: dict, **kw: Any) -> dict:
        self.calls.append(("classify_text", {"text": text, "tasks": tasks, **kw}))
        return self.canned["classify_text"]

    def extract(self, text: str, schema: Any, **kw: Any) -> dict:
        self.calls.append(("extract", {"text": text, "schema": schema, **kw}))
        return self.canned["extract"]

    def create_schema(self) -> Any:
        self.calls.append(("create_schema", {}))
        return self.canned["create_schema"]


def _judge(**canned: Any):
    from mellea_contribs.systemone.backends.gliner2 import Gliner2Judge

    return Gliner2Judge(model=StubModel(**canned))


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


def test_name_is_gliner2() -> None:
    j = _judge()
    assert j.name == "gliner2"


def test_accepts_injected_model_without_download() -> None:
    """An injected model must bypass from_pretrained entirely."""
    stub = StubModel()
    from mellea_contribs.systemone.backends.gliner2 import Gliner2Judge

    j = Gliner2Judge(model=stub)
    assert j.model is stub


def test_exposes_default_checkpoint_constant() -> None:
    from mellea_contribs.systemone.backends.gliner2 import DEFAULT_CHECKPOINT

    assert "gliner2" in DEFAULT_CHECKPOINT


# ---------------------------------------------------------------------------
# 2. choose
# ---------------------------------------------------------------------------


def test_choose_calls_classify_text_and_maps_verdict() -> None:
    j = _judge(classify_text={"_choice": {"label": "supports", "confidence": 0.93}})
    v = j.choose(
        state="The sky is blue.",
        options={"supports": "backs the claim", "contradicts": "refutes it"},
        instructions="How does the text relate to the claim?",
    )
    assert isinstance(v, Verdict)
    assert v.value == "supports"
    assert v.confidence == pytest.approx(0.93)
    assert v.provider == "gliner2"
    assert v.latency_ms >= 0

    method, kwargs = j.model.calls[0]
    assert method == "classify_text"
    # Option keys must reach the model as labels.
    labels = next(iter(kwargs["tasks"].values()))
    assert set(labels) == {"supports", "contradicts"} or set(labels.keys()) == {  # type: ignore[union-attr]
        "supports",
        "contradicts",
    }


def test_choose_requests_confidence_from_model() -> None:
    j = _judge(classify_text={"_choice": {"label": "a", "confidence": 0.5}})
    j.choose(state="x", options={"a": None, "b": None}, instructions="pick")
    _, kwargs = j.model.calls[0]
    assert kwargs.get("include_confidence") is True


def test_choose_passes_descriptions_when_given() -> None:
    """Option descriptions raise accuracy, so they must not be dropped."""
    j = _judge(classify_text={"_choice": {"label": "a", "confidence": 0.7}})
    j.choose(state="x", options={"a": "means A", "b": "means B"}, instructions="pick")
    _, kwargs = j.model.calls[0]
    labels = next(iter(kwargs["tasks"].values()))
    assert isinstance(labels, dict)
    assert labels["a"] == "means A"


# ---------------------------------------------------------------------------
# 3. truth
# ---------------------------------------------------------------------------


def test_truth_yes_maps_to_high_float() -> None:
    j = _judge(classify_text={"_truth": {"label": "yes", "confidence": 0.9}})
    v = j.truth(state="The sky is blue.", instructions="Is the sky blue?")
    assert v.value == pytest.approx(0.9)
    assert v.provider == "gliner2"


def test_truth_no_maps_to_low_float() -> None:
    """A confident 'no' must invert to a low truth score, not stay at 0.9."""
    j = _judge(classify_text={"_truth": {"label": "no", "confidence": 0.9}})
    v = j.truth(state="The sky is green.", instructions="Is the sky blue?")
    assert v.value == pytest.approx(0.1)


def test_truth_probabilities_carry_both_poles() -> None:
    j = _judge(classify_text={"_truth": {"label": "yes", "confidence": 0.8}})
    v = j.truth(state="x", instructions="true?")
    assert v.probabilities is not None
    assert v.probabilities["yes"] == pytest.approx(0.8)
    assert v.probabilities["no"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 4. rate
# ---------------------------------------------------------------------------


def test_rate_returns_rubric_label() -> None:
    j = _judge(classify_text={"_rate": {"label": "4", "confidence": 0.61}})
    v = j.rate(
        state="a decent summary",
        rubric=["1", "2", "3", "4", "5"],
        instructions="rate faithfulness",
    )
    assert v.value == "4"
    assert v.confidence == pytest.approx(0.61)


def test_rate_sends_rubric_as_labels() -> None:
    j = _judge(classify_text={"_rate": {"label": "low", "confidence": 0.5}})
    j.rate(state="x", rubric=["low", "medium", "high"], instructions="rate")
    _, kwargs = j.model.calls[0]
    labels = next(iter(kwargs["tasks"].values()))
    assert list(labels) == ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# 5. batch — must be ONE call, not N
# ---------------------------------------------------------------------------


class StubSchema:
    """Minimal stand-in for gliner2's Schema builder (fluent interface)."""

    def __init__(self) -> None:
        self.classifications: list[tuple[str, Any]] = []

    def classification(self, task: str, labels: Any, **kw: Any) -> StubSchema:
        self.classifications.append((task, labels))
        return self


def test_batch_issues_single_model_call() -> None:
    schema = StubSchema()
    j = _judge(
        create_schema=schema,
        extract={
            "team": {"label": "billing", "confidence": 0.91},
            "urgent": {"label": "yes", "confidence": 0.7},
        },
    )
    results = j.batch(
        state="I was charged twice, please help ASAP",
        questions={
            "team": ChoiceQ(
                options={"billing": None, "bug": None}, instructions="which team"
            ),
            "urgent": TruthQ(instructions="Is this urgent?"),
        },
    )
    extract_calls = [c for c in j.model.calls if c[0] == "extract"]
    assert len(extract_calls) == 1, "batch must compose one forward pass"
    assert not [c for c in j.model.calls if c[0] == "classify_text"]

    assert results["team"].value == "billing"
    assert results["urgent"].value == pytest.approx(0.7)


def test_batch_registers_every_question_on_schema() -> None:
    schema = StubSchema()
    j = _judge(
        create_schema=schema,
        extract={
            "a": {"label": "x", "confidence": 0.5},
            "b": {"label": "1", "confidence": 0.5},
        },
    )
    j.batch(
        state="s",
        questions={
            "a": ChoiceQ(options={"x": None, "y": None}, instructions="i"),
            "b": RateQ(rubric=["1", "2"], instructions="i"),
        },
    )
    assert {task for task, _ in schema.classifications} == {"a", "b"}


def test_batch_empty_questions_makes_no_call() -> None:
    j = _judge(create_schema=StubSchema(), extract={})
    assert j.batch(state="s", questions={}) == {}
    assert not [c for c in j.model.calls if c[0] == "extract"]


# ---------------------------------------------------------------------------
# 6. Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_judge_protocol() -> None:
    assert isinstance(_judge(), Judge)


# ---------------------------------------------------------------------------
# 7. Confidence semantics
# ---------------------------------------------------------------------------


def test_choose_surfaces_probabilities_when_model_gives_them() -> None:
    j = _judge(
        classify_text={
            "_choice": {
                "label": "a",
                "confidence": 0.6,
                "probabilities": {"a": 0.6, "b": 0.4},
            }
        }
    )
    v = j.choose(state="x", options={"a": None, "b": None}, instructions="pick")
    assert v.probabilities == {"a": 0.6, "b": 0.4}


def test_missing_confidence_is_none_not_zero() -> None:
    """Absent confidence must not be silently coerced to 0.0."""
    j = _judge(classify_text={"_choice": {"label": "a"}})
    v = j.choose(state="x", options={"a": None}, instructions="pick")
    assert v.confidence is None
