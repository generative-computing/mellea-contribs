"""GLiNER2-backed :class:`Judge`.

GLiNER2 is an Apache-2.0 encoder (DeBERTa-v3 family) that runs locally with no
API key.  It is the default provider for this package: a single forward pass
answers classification questions about a state.

:class:`Gliner2Judge` implements
:class:`~mellea_contribs.systemone.core.judge.Judge` — ``choose``, ``truth``,
``rate``, ``batch``.

Method mapping
--------------
========================  ====================================================
Protocol method           GLiNER2 call
========================  ====================================================
``choose``                ``classify_text`` with one single-label task
``truth``                 ``classify_text`` over ``["yes", "no"]``, inverted
                          for a ``"no"`` label so the float stays "probability
                          the claim is true"
``rate``                  ``classify_text`` with the rubric as labels
``batch``                 ``create_schema()`` + one ``extract`` forward pass
========================  ====================================================

``batch`` composes every question into one schema and issues a single forward
pass.  This is the whole point of the protocol's ``batch`` method: N questions
about the same state cost one pass, not N.

Confidence
----------
.. warning::
    GLiNER2 confidence is **softmax over label logits and is not calibrated**.
    It is comparable *across examples for the same task* but not across tasks,
    and not comparable to Jev's RLCD-calibrated values.  Every threshold in
    this package that consumes a GLiNER2 confidence is marked "tune this".

Absent confidence is surfaced as ``None``, never coerced to ``0.0`` — a missing
measurement and a confident "no" are different facts.

Installation
------------
::

    pip install "mellea-contribs-systemone[gliner2]"   # torch-free client
    pip install "mellea-contribs-systemone[local]"     # + torch, local weights

Constructing without the SDK installed raises
:class:`~mellea_contribs.systemone.core.errors.JudgeUnavailable` with a
``pip install`` hint.

Testing
-------
The constructor accepts an injected ``model``, which bypasses
``from_pretrained`` entirely.  Unit tests pass a stub and therefore download no
checkpoint; see ``tests/unit/test_phase2_gliner2.py``.
"""

from __future__ import annotations

import time
from typing import Any

from mellea_contribs.systemone.core.errors import JudgeUnavailable
from mellea_contribs.systemone.core.judge import (
    ChoiceQ,
    Question,
    RateQ,
    TruthQ,
    Verdict,
)

__all__ = ["DEFAULT_CHECKPOINT", "Gliner2Judge"]

#: Default checkpoint: the 194M base model, a reasonable accuracy/latency
#: trade-off for judging.  Override via the ``checkpoint`` argument.
DEFAULT_CHECKPOINT = "fastino/gliner2.5-base-v1"

#: Task key used for the single-question convenience methods.  GLiNER2 requires
#: every classification task to be named; these names never reach the caller.
_CHOICE_TASK = "_choice"
_TRUTH_TASK = "_truth"
_RATE_TASK = "_rate"

_YES = "yes"
_NO = "no"


def _stringify(state: str | dict[str, Any]) -> str:
    """Flatten a state into the single text field GLiNER2 accepts.

    GLiNER2 takes one ``text`` argument, so a dict state is rendered as
    ``key: value`` lines.  Ordering follows insertion order so the rendering is
    deterministic for a given dict.
    """
    if isinstance(state, str):
        return state
    return "\n".join(f"{k}: {v}" for k, v in state.items())


def _labels_arg(options: dict[str, str | None]) -> list[str] | dict[str, str]:
    """Render protocol options as GLiNER2 labels.

    GLiNER2 accepts either a plain label list or a ``{label: description}``
    mapping, and descriptions measurably improve accuracy.  Descriptions are
    therefore passed through whenever *any* option supplies one; options
    without a description fall back to their own key as the description, since
    dropping the mapping entirely would discard the descriptions that exist.
    """
    if any(desc is not None for desc in options.values()):
        return {
            key: (desc if desc is not None else key) for key, desc in options.items()
        }
    return list(options)


def _read_task(raw: dict[str, Any], task: str) -> dict[str, Any]:
    """Pull one task's result out of a GLiNER2 classification payload.

    GLiNER2 nests results under the task name.  A missing task means the model
    returned nothing for it, which is represented as an empty result rather
    than a ``KeyError`` so callers see ``value=None`` / ``confidence=None``.
    """
    value = raw.get(task)
    if isinstance(value, dict):
        return value
    # Some configurations return a list of scored labels; take the top one.
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


class Gliner2Judge:
    """A :class:`Judge` backed by a local GLiNER2 model.

    Args:
        model: A pre-loaded GLiNER2 extractor (anything exposing
            ``classify_text``, ``create_schema`` and ``extract``).  When
            given, no checkpoint is loaded — this is the injection point
            that keeps unit tests hermetic.
        checkpoint: HuggingFace checkpoint to load when ``model`` is ``None``.
            Defaults to :data:`DEFAULT_CHECKPOINT`.
        threshold: Default confidence threshold passed to GLiNER2 for
            classification calls.

    Raises:
        JudgeUnavailable: If ``model`` is ``None`` and the ``gliner2`` SDK is
            not installed.

    Example::

        judge = Gliner2Judge()
        verdict = judge.choose(
            state=document,
            options={"supports": "backs the claim", "contradicts": "refutes it"},
            instructions="How does the document relate to the claim?",
        )
        if verdict.confidence is not None and verdict.confidence > 0.8:
            accept(verdict.value)
    """

    name = "gliner2"

    def __init__(
        self,
        model: Any | None = None,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.model = model if model is not None else self._load(checkpoint)

    @staticmethod
    def _load(checkpoint: str) -> Any:
        """Load a checkpoint, converting an import failure into a clear error."""
        try:
            # Imported lazily and by module so the SDK is only required when a
            # real model is actually loaded.
            import gliner2
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise JudgeUnavailable(
                "gliner2 is not installed. Install it with:\n"
                '    pip install "mellea-contribs-systemone[local]"\n'
                "(or [gliner2] for the torch-free API client)."
            ) from exc
        return gliner2.AutoExtractor.from_pretrained(checkpoint)

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    def choose(
        self,
        state: str | dict[str, Any],
        options: dict[str, str | None],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[str]:
        """Pick one option from a closed set. See :meth:`Judge.choose`."""
        started = time.perf_counter()
        raw = self.model.classify_text(
            _stringify(state),
            {_CHOICE_TASK: _labels_arg(options)},
            threshold=self.threshold,
            include_confidence=True,
        )
        elapsed = (time.perf_counter() - started) * 1000
        result = _read_task(raw, _CHOICE_TASK)
        return Verdict(
            value=result.get("label"),
            confidence=result.get("confidence"),
            probabilities=result.get("probabilities"),
            provider=self.name,
            latency_ms=elapsed,
        )

    def truth(
        self,
        state: str | dict[str, Any],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[float]:
        """Score a boolean claim in [0, 1]. See :meth:`Judge.truth`.

        GLiNER2 has no native boolean primitive, so this is a two-label
        classification over ``["yes", "no"]``.  A confident ``"no"`` is
        **inverted**: ``value`` always means "probability the claim is true",
        so ``label="no", confidence=0.9`` becomes ``value=0.1``.
        """
        started = time.perf_counter()
        raw = self.model.classify_text(
            _stringify(state),
            {_TRUTH_TASK: {_YES: instructions, _NO: f"not: {instructions}"}},
            threshold=self.threshold,
            include_confidence=True,
        )
        elapsed = (time.perf_counter() - started) * 1000
        result = _read_task(raw, _TRUTH_TASK)
        confidence = result.get("confidence")
        label = result.get("label")

        if confidence is None:
            score: float | None = None
            probabilities = None
        else:
            score = confidence if label == _YES else 1.0 - confidence
            probabilities = {_YES: score, _NO: 1.0 - score}

        return Verdict(
            value=score,
            confidence=confidence,
            probabilities=probabilities,
            provider=self.name,
            latency_ms=elapsed,
        )

    def rate(
        self,
        state: str | dict[str, Any],
        rubric: list[str],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[str]:
        """Assign an ordinal rating. See :meth:`Judge.rate`.

        The rubric is passed as the label set in its given order, so GLiNER2
        sees the ordinal scale rather than an unordered set.
        """
        started = time.perf_counter()
        raw = self.model.classify_text(
            _stringify(state),
            {_RATE_TASK: list(rubric)},
            threshold=self.threshold,
            include_confidence=True,
        )
        elapsed = (time.perf_counter() - started) * 1000
        result = _read_task(raw, _RATE_TASK)
        return Verdict(
            value=result.get("label"),
            confidence=result.get("confidence"),
            probabilities=result.get("probabilities"),
            provider=self.name,
            latency_ms=elapsed,
        )

    def batch(
        self,
        state: str | dict[str, Any],
        questions: dict[str, Question],
        *,
        images: list[bytes] | None = None,
    ) -> dict[str, Verdict[Any]]:
        """Answer every question in **one** forward pass. See :meth:`Judge.batch`.

        Each question becomes one classification task on a composed schema, so
        N questions about the same state cost one pass rather than N.  Boolean
        questions are inverted exactly as in :meth:`truth`.

        Raises:
            TypeError: If a value in ``questions`` is not a ``ChoiceQ``,
                ``TruthQ`` or ``RateQ``.
        """
        if not questions:
            return {}

        schema = self.model.create_schema()
        for key, question in questions.items():
            if isinstance(question, ChoiceQ):
                schema = schema.classification(key, _labels_arg(question.options))
            elif isinstance(question, TruthQ):
                schema = schema.classification(
                    key,
                    {_YES: question.instructions, _NO: f"not: {question.instructions}"},
                )
            elif isinstance(question, RateQ):
                schema = schema.classification(key, list(question.rubric))
            else:
                raise TypeError(
                    f"unknown question type for key {key!r}: {type(question).__name__}"
                )

        started = time.perf_counter()
        raw = self.model.extract(
            _stringify(state),
            schema,
            threshold=self.threshold,
            include_confidence=True,
        )
        elapsed = (time.perf_counter() - started) * 1000

        verdicts: dict[str, Verdict[Any]] = {}
        for key, question in questions.items():
            result = _read_task(raw, key)
            confidence = result.get("confidence")
            label = result.get("label")

            if isinstance(question, TruthQ):
                if confidence is None:
                    value: Any = None
                    probabilities = None
                else:
                    value = confidence if label == _YES else 1.0 - confidence
                    probabilities = {_YES: value, _NO: 1.0 - value}
            else:
                value = label
                probabilities = result.get("probabilities")

            verdicts[key] = Verdict(
                value=value,
                confidence=confidence,
                probabilities=probabilities,
                provider=self.name,
                latency_ms=elapsed,
            )
        return verdicts
