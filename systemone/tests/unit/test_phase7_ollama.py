"""Unit tests for Phase 7: OllamaJudge + Image annotation.

Hermetic — a ``StubClient`` stands in for ``TypeSafeClient``, so these run with
no Ollama server and no network.

Test sections
-------------
1. Construction — host/model defaults, name tag, env var, injected client.
2. Capability boundary — implements Judge.
3. Text-only (no images) — delegates to JevJudge, same question mapping.
4. Image support — base64 encoding, extra_body passthrough.
5. Provider tagging — verdicts say ``"ollama"``, not ``"jev"``.
6. Image annotation with @decisive — Image-annotated params routed to images kwarg.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any, Literal

import pytest

from mellea_contribs.systemone.core.judge import (
    ChoiceQ,
    Image,
    Judge,
    RateQ,
    TruthQ,
    Verdict,
)

pytestmark = pytest.mark.unit

typesafe_sdk = pytest.importorskip("typesafe_sdk", reason="needs the [jev] extra")


# ---------------------------------------------------------------------------
# Stubs (same shape as test_phase4_jev.py)
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
            model="ollama-tev1",
            usage=typesafe_sdk.Usage(input_tokens=10, output_tokens=0),
            answers=self._answers,
        )


def _judge(answers: dict[str, Any] | None = None, raises: Exception | None = None):
    from mellea_contribs.systemone.backends.ollama import OllamaJudge

    return OllamaJudge(client=StubClient(answers, raises))


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


def test_name_is_ollama() -> None:
    assert _judge().name == "ollama"


def test_accepts_injected_client() -> None:
    from mellea_contribs.systemone.backends.ollama import OllamaJudge

    stub = StubClient()
    assert OllamaJudge(client=stub).client is stub


def test_default_model_is_tev1() -> None:
    from mellea_contribs.systemone.backends.ollama import OllamaJudge

    judge = OllamaJudge(client=StubClient())
    assert judge.model is None or judge.model == "tev1"


def test_custom_model() -> None:
    from mellea_contribs.systemone.backends.ollama import OllamaJudge

    judge = OllamaJudge(client=StubClient(), model="clef")
    assert judge.model == "clef"


# ---------------------------------------------------------------------------
# 2. Capability boundary
# ---------------------------------------------------------------------------


def test_is_a_judge() -> None:
    assert isinstance(_judge(), Judge)


# ---------------------------------------------------------------------------
# 3. Text-only (no images) — inherits JevJudge question mapping
# ---------------------------------------------------------------------------


def test_batch_without_images() -> None:
    judge = _judge({"team": _choice("billing", 0.9), "urgent": _noul(0.7)})
    results = judge.batch(
        state="ticket text",
        questions={
            "team": ChoiceQ(options={"billing": None, "bug": None}, instructions="which team"),
            "urgent": TruthQ(instructions="urgent?"),
        },
    )
    assert len(judge.client.calls) == 1
    assert results["team"].value == "billing"
    assert results["urgent"].value == pytest.approx(0.7)
    assert "extra_body" not in judge.client.calls[0]


# ---------------------------------------------------------------------------
# 4. Image support
# ---------------------------------------------------------------------------


SAMPLE_IMAGE = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rfake-png-data"


def test_batch_with_images_passes_base64_in_extra_body() -> None:
    judge = _judge({"q": _noul(0.9)})
    judge.batch(
        state="photo context",
        questions={"q": TruthQ(instructions="Is there a cat?")},
        images=[SAMPLE_IMAGE],
    )
    call = judge.client.calls[0]
    assert "extra_body" in call
    expected_b64 = base64.b64encode(SAMPLE_IMAGE).decode("ascii")
    assert call["extra_body"]["images"] == [expected_b64]


def test_batch_with_multiple_images() -> None:
    img_a = b"image-a"
    img_b = b"image-b"
    judge = _judge({"q": _noul(0.5)})
    judge.batch(
        state="two photos",
        questions={"q": TruthQ(instructions="compare")},
        images=[img_a, img_b],
    )
    call = judge.client.calls[0]
    encoded = call["extra_body"]["images"]
    assert len(encoded) == 2
    assert base64.b64decode(encoded[0]) == img_a
    assert base64.b64decode(encoded[1]) == img_b


def test_batch_with_empty_images_list_skips_extra_body() -> None:
    judge = _judge({"q": _noul(0.5)})
    judge.batch(
        state="no images",
        questions={"q": TruthQ(instructions="true?")},
        images=[],
    )
    assert "extra_body" not in judge.client.calls[0]


def test_batch_with_none_images_skips_extra_body() -> None:
    judge = _judge({"q": _noul(0.5)})
    judge.batch(
        state="no images",
        questions={"q": TruthQ(instructions="true?")},
        images=None,
    )
    assert "extra_body" not in judge.client.calls[0]



# ---------------------------------------------------------------------------
# 5. Provider tagging
# ---------------------------------------------------------------------------


def test_verdict_provider_is_ollama() -> None:
    judge = _judge({"_q": _noul(0.5)})
    v = judge.truth(state="x", instructions="?")
    assert v.provider == "ollama"


def test_batch_verdicts_tagged_ollama() -> None:
    judge = _judge({"q1": _choice("a", 0.9), "q2": _noul(0.5)})
    results = judge.batch(
        state="x",
        questions={
            "q1": ChoiceQ(options={"a": None, "b": None}, instructions="pick"),
            "q2": TruthQ(instructions="?"),
        },
    )
    assert results["q1"].provider == "ollama"
    assert results["q2"].provider == "ollama"


# ---------------------------------------------------------------------------
# 6. Image annotation with @decisive
# ---------------------------------------------------------------------------


def _v(value: Any, confidence: float) -> Verdict[Any]:
    return Verdict(
        value=value, confidence=confidence, probabilities=None,
        provider="fake", latency_ms=1.0,
    )


def _fake(**kw: Any):
    from mellea_contribs.systemone.backends.fake import FakeJudge

    return FakeJudge(**kw)


def test_image_param_excluded_from_state() -> None:
    from mellea_contribs.systemone import decisive

    @decisive
    def classify_photo(
        caption: str,
        photo: Annotated[bytes, Image()],
    ) -> Literal["cat", "dog"]:
        """Classify the animal."""

    judge = _fake(choose_verdicts=[_v("cat", 0.9)])
    classify_photo(judge=judge, caption="a cute pet", photo=b"image-bytes")

    call = judge.calls[0]
    assert call.method == "choose"
    assert "photo" not in call.kwargs["state"]
    assert call.kwargs["state"] == {"caption": "a cute pet"}
    assert call.kwargs["images"] == [b"image-bytes"]


def test_no_image_annotation_passes_none() -> None:
    from mellea_contribs.systemone import decisive

    @decisive
    def triage(ticket: str) -> Literal["billing", "bug"]:
        """Route."""

    judge = _fake(choose_verdicts=[_v("billing", 0.9)])
    triage(judge=judge, ticket="text")

    call = judge.calls[0]
    assert call.kwargs["images"] is None


def test_multiple_image_params() -> None:
    from mellea_contribs.systemone import decisive

    @decisive
    def compare(
        query: str,
        img_a: Annotated[bytes, Image()],
        img_b: Annotated[bytes, Image()],
    ) -> Literal["same", "different"]:
        """Compare two images."""

    judge = _fake(choose_verdicts=[_v("same", 0.8)])
    compare(judge=judge, query="are these the same?", img_a=b"aaa", img_b=b"bbb")

    call = judge.calls[0]
    assert call.kwargs["state"] == {"query": "are these the same?"}
    assert call.kwargs["images"] == [b"aaa", b"bbb"]


def test_image_param_with_truth() -> None:
    from mellea_contribs.systemone import decisive

    @decisive
    def has_cat(photo: Annotated[bytes, Image()]) -> bool:
        """Is there a cat?"""

    judge = _fake(truth_verdicts=[_v(0.95, 0.95)])
    result = has_cat(judge=judge, photo=b"cat-image")
    assert result is True

    call = judge.calls[0]
    assert call.kwargs["images"] == [b"cat-image"]
    assert call.kwargs["state"] == {}


def test_list_of_bytes_image_param() -> None:
    from mellea_contribs.systemone import decisive

    @decisive
    def classify_album(
        description: str,
        photos: Annotated[list[bytes], Image()],
    ) -> Literal["vacation", "work"]:
        """Classify the photo album."""

    judge = _fake(choose_verdicts=[_v("vacation", 0.9)])
    classify_album(judge=judge, description="beach trip", photos=[b"img1", b"img2"])

    call = judge.calls[0]
    assert call.kwargs["images"] == [b"img1", b"img2"]
    assert call.kwargs["state"] == {"description": "beach trip"}
