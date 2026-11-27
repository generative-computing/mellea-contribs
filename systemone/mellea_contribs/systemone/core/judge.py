"""Core protocol definitions for mellea-contribs-systemone.

This module defines the structural :class:`Judge` protocol plus the value
types it traffics in.  All other modules in this package depend inward on this
module; it has no dependencies on providers.

Protocol
--------
:class:`Judge`
    Closed-decision interface: choose among options, score a boolean, rate
    against a rubric, or ask many questions in one batch.

Value types
-----------
:class:`Verdict` — a typed answer plus the metadata needed to decide whether
to trust it (confidence, probabilities, provider tag, latency).

:class:`Image` — annotation marker routing a ``@decisive`` parameter to the
judge's ``images`` keyword.

Question tagged union — :class:`ChoiceQ`, :class:`TruthQ`, :class:`RateQ` —
typed question descriptors for :meth:`Judge.batch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ChoiceQ",
    "Image",
    "Judge",
    "Question",
    "RateQ",
    "TruthQ",
    "Verdict",
]


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict[T]:
    """A typed answer plus the metadata needed to decide whether to trust it.

    Attributes:
        value: The typed answer produced by the judge.
        confidence: Calibrated 0–1 probability that ``value`` is correct, or
            ``None`` if the provider does not expose per-prediction confidence.

            .. warning::
                GLiNER2 confidence is **not calibrated** — it is softmax over
                label logits.  Thresholds must be tuned per task and are not
                comparable to Jev confidence values.

        probabilities: Full label-probability distribution, or ``None`` if the
            provider does not expose it.  For ``choose`` / ``rate`` calls this
            maps each option to its probability; for ``truth`` it maps
            ``{"yes": p, "no": 1-p}``.
        provider: Tag identifying which provider produced this verdict —
            ``"gliner2"``, ``"jev"``, ``"ollama"``, or ``"fake"``.
        latency_ms: Wall-clock time for the provider call, in milliseconds.
    """

    value: T
    confidence: float | None
    probabilities: dict[str, float] | None
    provider: str
    latency_ms: float


@dataclass(frozen=True)
class Image:
    """Annotation marker: this parameter carries image bytes for the judge.

    Used with ``Annotated`` to route a ``@decisive`` parameter to the judge's
    ``images`` keyword instead of including it in the state dict::

        @decisive
        def describe_scene(
            caption: str,
            photo: Annotated[bytes, Image()],
        ) -> Literal["indoor", "outdoor"]:
            '''Is this scene indoors or outdoors?'''

    Attributes:
        mime_type: Optional MIME hint (``"image/png"``, ``"image/jpeg"``).
            Currently informational; the judge infers format from the bytes.
    """

    mime_type: str | None = None


# ---------------------------------------------------------------------------
# Question tagged union  (for Judge.batch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChoiceQ:
    """A closed-choice question.

    Attributes:
        options: Mapping from option key to natural-language description (or
            ``None`` if no description is needed).
        instructions: Instruction string for the judge.
    """

    options: dict[str, str | None]
    instructions: str


@dataclass(frozen=True)
class TruthQ:
    """A boolean (Noul / confidence-as-truth) question.

    Attributes:
        instructions: Instruction string for the judge.
    """

    instructions: str


@dataclass(frozen=True)
class RateQ:
    """An ordinal rating question.

    Attributes:
        rubric: Ordered list of rating labels (e.g. ``["1", "2", "3", "4", "5"]``).
        instructions: Instruction string for the judge.
    """

    rubric: list[str]
    instructions: str


#: Tagged union of the three question types accepted by :meth:`Judge.batch`.
Question = ChoiceQ | TruthQ | RateQ


# ---------------------------------------------------------------------------
# Judge protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Judge(Protocol):
    """Structural protocol for closed-decision providers.

    A ``Judge`` answers questions about a *state* (a document, transcript, or
    structured dict) without generating free text.  It exposes four primitives:

    - :meth:`choose` — pick one option from a closed set.
    - :meth:`truth` — return a calibrated 0–1 score for a boolean claim.
    - :meth:`rate` — assign an ordinal rating against a rubric.
    - :meth:`batch` — ask multiple questions in a single provider round-trip.

    All methods return :class:`Verdict` carrying the answer, confidence, and
    provider metadata.

    ``batch`` is not sugar over a loop.  It is one network call for Jev and one
    forward pass for GLiNER2.  Consumers asking N questions about the same
    state **must** use it.
    """

    #: Identifies the provider, e.g. ``"gliner2"``, ``"jev"``, ``"fake"``.
    name: str

    def choose(
        self,
        state: str | dict[str, Any],
        options: dict[str, str | None],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[str]:
        """Pick one option from a closed set.

        Args:
            state: The document or structured context the question is about.
            options: Mapping from option key to description.  The returned
                ``Verdict.value`` is one of these keys.
            instructions: Task instruction for the judge.
            images: Optional list of raw image bytes (PNG/JPEG/WebP) shared
                across the decision.  Only multimodal providers (e.g. Ollama
                Clef) use this; text-only providers accept and ignore it.

        Returns:
            :class:`Verdict` whose ``value`` is one key from ``options``.
        """
        ...

    def truth(
        self,
        state: str | dict[str, Any],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[float]:
        """Return a calibrated 0–1 score for a boolean claim.

        A score of ``1.0`` means the judge is certain the claim is true;
        ``0.0`` means it is certain the claim is false.

        Args:
            state: The document or structured context the question is about.
            instructions: The claim to evaluate, stated as a yes/no question.
            images: Optional list of raw image bytes for multimodal providers.

        Returns:
            :class:`Verdict` whose ``value`` is a float in [0, 1].
        """
        ...

    def rate(
        self,
        state: str | dict[str, Any],
        rubric: list[str],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[str]:
        """Assign an ordinal rating against a rubric.

        Args:
            state: The document or structured context the question is about.
            rubric: Ordered list of rating labels.  The returned
                ``Verdict.value`` is one of these labels.
            instructions: Task instruction for the judge.
            images: Optional list of raw image bytes for multimodal providers.

        Returns:
            :class:`Verdict` whose ``value`` is one element of ``rubric``.
        """
        ...

    def batch(
        self,
        state: str | dict[str, Any],
        questions: dict[str, Question],
        *,
        images: list[bytes] | None = None,
    ) -> dict[str, Verdict[Any]]:
        """Ask multiple questions in a single provider round-trip.

        For Jev this is one ``system_one`` call with N parallel questions.
        For GLiNER2 this is one ``create_schema()`` composed forward pass.

        Args:
            state: The document or structured context all questions are about.
            questions: Mapping from question key to a
                :class:`ChoiceQ`, :class:`TruthQ`, or :class:`RateQ`.
            images: Optional list of raw image bytes shared across all
                questions.  Only multimodal providers use this.

        Returns:
            Dict mapping each key to its :class:`Verdict`.
        """
        ...
