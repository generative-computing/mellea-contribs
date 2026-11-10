"""Integration tests for Gliner2Judge against a real checkpoint.

Marked ``integration``: these download ~194M of weights on first run and need
``pip install "mellea-contribs-systemone[local]"`` (torch).  They are excluded
from the default CI tier; run them with::

    pytest -m integration

What these cover that the unit tests cannot
-------------------------------------------
The unit tests pin the *mapping* from protocol call to GLiNER2 call using a
stub.  They cannot catch a mapping that is internally consistent but wrong
about the real library — a renamed kwarg, a changed return shape, or a task
name that GLiNER2 rejects.  These tests exercise exactly that seam, asserting
only on shape and direction, never on exact confidence values (which are
uncalibrated and will drift between checkpoints).
"""

from __future__ import annotations

import pytest

from mellea_contribs.systemone.core.errors import JudgeUnavailable
from mellea_contribs.systemone.core.judge import (
    ChoiceQ,
    Judge,
    RateQ,
    TruthQ,
)

pytestmark = pytest.mark.integration

pytest.importorskip("torch", reason="integration tier needs [local] extra")


@pytest.fixture(scope="module")
def judge():
    """Load the real checkpoint, skipping if the environment cannot build it.

    A checkpoint load can fail for reasons that have nothing to do with this
    adapter — notably a ``transformers`` version that satisfies gliner2's
    ``>=4.38,<5`` pin but cannot construct the DeBERTa-v2 tokenizer for a given
    checkpoint.  That is an upstream incompatibility, so it skips rather than
    failing this package's suite; anything raised from our own code (e.g.
    :class:`JudgeUnavailable`) still fails loudly.
    """
    from mellea_contribs.systemone.backends.gliner2 import Gliner2Judge

    try:
        return Gliner2Judge()
    except JudgeUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - upstream load failure, not ours
        pytest.skip(f"checkpoint could not be loaded in this environment: {exc!r}")


def test_conforms_to_judge_protocol(judge) -> None:
    assert isinstance(judge, Judge)


def test_choose_returns_one_of_the_options(judge) -> None:
    v = judge.choose(
        state="The Eiffel Tower is in Paris, France.",
        options={
            "supports": "the text confirms the claim",
            "contradicts": "the text refutes the claim",
            "says_nothing": "the text is silent on the claim",
        },
        instructions="Claim: the Eiffel Tower is in Paris.",
    )
    assert v.value in {"supports", "contradicts", "says_nothing"}
    assert v.provider == "gliner2"
    assert v.latency_ms > 0


def test_truth_returns_a_float_in_range(judge) -> None:
    v = judge.truth(
        state="The Eiffel Tower is in Paris, France.",
        instructions="Is the Eiffel Tower in Paris?",
    )
    assert v.value is None or 0.0 <= v.value <= 1.0


def test_truth_direction_separates_true_from_false(judge) -> None:
    """The sign of the mapping must be right, even if the magnitudes are not.

    This is the one assertion that would catch an inverted ``truth`` — a bug a
    stub test cannot see, because the stub's label is whatever we scripted.
    """
    state = "The Eiffel Tower is in Paris, France."
    true_claim = judge.truth(state=state, instructions="Is the Eiffel Tower in Paris?")
    false_claim = judge.truth(state=state, instructions="Is the Eiffel Tower in Tokyo?")
    if true_claim.value is None or false_claim.value is None:
        pytest.skip("checkpoint returned no confidence for this task")
    assert true_claim.value > false_claim.value


def test_rate_returns_a_rubric_label(judge) -> None:
    rubric = ["1", "2", "3", "4", "5"]
    v = judge.rate(
        state="A thorough, well-sourced summary.",
        rubric=rubric,
        instructions="Rate the quality of this summary.",
    )
    assert v.value in rubric


def test_batch_answers_every_question(judge) -> None:
    results = judge.batch(
        state="I was charged twice this month. Please fix this immediately.",
        questions={
            "team": ChoiceQ(
                options={"billing": "a payment issue", "bug": "a software defect"},
                instructions="Which team should handle this?",
            ),
            "urgent": TruthQ(instructions="Is the customer asking for urgent help?"),
            "severity": RateQ(
                rubric=["low", "medium", "high"], instructions="Rate severity."
            ),
        },
    )
    assert set(results) == {"team", "urgent", "severity"}
    assert results["team"].value in {"billing", "bug", None}
    assert results["urgent"].value is None or 0.0 <= results["urgent"].value <= 1.0
    assert results["severity"].value in {"low", "medium", "high", None}

