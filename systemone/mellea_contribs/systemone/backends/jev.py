"""Jev-backed :class:`Judge` (TypeSafe "System One").

Jev is a hosted, non-generative decision model.  It answers closed questions
about a state with **RLCD-calibrated** confidence — the property GLiNER2 lacks
and the reason this provider exists.  Roughly 70–500 ms per call, input-priced,
output free, because there is no output to generate.

Capability boundary
-------------------
Jev produces **no string output at all**; it only answers closed questions.
:class:`JevJudge` implements
:class:`~mellea_contribs.systemone.core.judge.Judge`.

Question mapping
----------------
========================  ====================================================
Protocol method           Jev question type
========================  ====================================================
``choose``                ``Choice(criteria={option: description})``
``truth``                 ``Noul(instructions=...)``
``rate``                  ``Score(criteria=[rubric...])``
``batch``                 one ``system_one`` call carrying every question
========================  ====================================================

Response shapes (verified against ``typesafe-sdk`` 0.7.1)
---------------------------------------------------------
- ``SystemOneResponse.answers`` is one flat ``dict[str, Answer]``.
- ``NoulAnswer`` carries ``noul: float`` and **no** confidence field: the noul
  *is* the calibrated probability, so :attr:`Verdict.confidence` is set to it.
- ``ChoiceAnswer`` carries ``choice``, ``confidence``, ``probabilities``.
- ``ScoreAnswer`` carries ``score: float``, ``confidence``, ``legend``
  (``dict[int, str]``) and ``probabilities`` keyed by **int**.  Since the
  protocol promises ``Verdict[str]``, the float is resolved back to the nearest
  rubric label and probabilities are re-keyed to labels.

Status
------
.. warning::
    **Experimental, and unvalidated against the live API.**  API access is
    waitlisted, so this adapter was written against the published SDK's types
    and exercised only against stubs.  The mapping is type-correct and the
    response shapes above were read off the installed SDK, but no response from
    the real service has ever passed through this code.  Treat the first live
    run as the real test.

Installation
------------
::

    pip install "mellea-contribs-systemone[jev]"
    export TYPESAFE_API_KEY=...
"""

from __future__ import annotations

import os
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

__all__ = ["API_KEY_ENV", "JevJudge"]

#: Environment variable holding the TypeSafe API key.
API_KEY_ENV = "TYPESAFE_API_KEY"

#: Key used for the single-question convenience methods; never seen by callers.
_Q = "_q"

_YES = "yes"
_NO = "no"


def _sdk() -> Any:
    """Import the SDK, translating absence into a clear install hint."""
    try:
        import typesafe_sdk
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise JudgeUnavailable(
            "typesafe-sdk is not installed. Install it with:\n"
            '    pip install "mellea-contribs-systemone[jev]"'
        ) from exc
    return typesafe_sdk


class JevJudge:
    """A :class:`Judge` backed by TypeSafe's hosted System One model.

    Args:
        client: A pre-built ``TypeSafeClient``.  When given, no key lookup
            happens — this is the injection point that keeps unit tests
            hermetic.
        api_key: API key.  Defaults to the ``TYPESAFE_API_KEY`` environment
            variable.
        model: Optional model override passed through to the SDK.

    Raises:
        JudgeUnavailable: If the SDK is missing, or if no ``client`` and no API
            key are available.

    Example::

        judge = JevJudge()                      # reads TYPESAFE_API_KEY
        v = judge.truth(state=ticket, instructions="Is this urgent?")
        if v.value > 0.9:                       # calibrated, unlike GLiNER2
            escalate(ticket)
    """

    name = "jev"

    def __init__(
        self,
        client: Any | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return

        key = api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise JudgeUnavailable(
                f"No Jev credentials. Set {API_KEY_ENV}, pass api_key=..., or inject a client.\n"
                "Note that Jev API access is waitlisted; use Gliner2Judge to run locally."
            )
        self.client = _sdk().TypeSafeClient(api_key=key, model=model)

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
        return self._one(state, ChoiceQ(options=options, instructions=instructions), images=images)

    def truth(
        self,
        state: str | dict[str, Any],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[float]:
        """Score a boolean claim in [0, 1]. See :meth:`Judge.truth`.

        This maps to a Noul, Jev's native calibrated-probability primitive, so
        no inversion or thresholding is needed — unlike the GLiNER2 backend,
        which has to synthesise a boolean from a two-label classification.
        """
        return self._one(state, TruthQ(instructions=instructions), images=images)

    def rate(
        self,
        state: str | dict[str, Any],
        rubric: list[str],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[str]:
        """Assign an ordinal rating. See :meth:`Judge.rate`."""
        return self._one(state, RateQ(rubric=rubric, instructions=instructions), images=images)

    def batch(
        self,
        state: str | dict[str, Any],
        questions: dict[str, Question],
        *,
        images: list[bytes] | None = None,
    ) -> dict[str, Verdict[Any]]:
        """Answer every question in one API round trip. See :meth:`Judge.batch`.

        Jev accepts N parallel questions per ``system_one`` call, so this is one
        network round trip regardless of N.  ``state`` is passed through
        unchanged — Jev takes JSON natively, so a dict needs no flattening.

        Raises:
            TypeError: If a value in ``questions`` is not a ``ChoiceQ``,
                ``TruthQ`` or ``RateQ``.
            JudgeUnavailable: On an authentication failure.
        """
        if not questions:
            return {}

        sdk = _sdk()
        payload = {key: _to_sdk_question(sdk, q) for key, q in questions.items()}

        started = time.perf_counter()
        try:
            response = self._call_system_one(sdk, state, payload, images=images)
        except sdk.TypeSafeAuthenticationError as exc:
            raise JudgeUnavailable(
                f"Jev rejected the credentials. Check {API_KEY_ENV}. ({exc})"
            ) from exc
        elapsed = (time.perf_counter() - started) * 1000

        answers = getattr(response, "answers", {}) or {}
        return {
            key: _to_verdict(answers.get(key), question, elapsed, self.name)
            for key, question in questions.items()
        }

    def _call_system_one(
        self, sdk: Any, state: Any, payload: dict[str, Any], *, images: list[bytes] | None = None
    ) -> Any:
        """Execute the SDK ``system_one`` call.  Override in subclasses."""
        return self.client.system_one(state, payload)

    def _one(self, state: str | dict[str, Any], question: Question, *, images: list[bytes] | None = None) -> Verdict[Any]:
        """Run a single question through :meth:`batch`.

        One code path for any N keeps the answer-mapping logic in exactly one
        place; the wrapper key never escapes this method.
        """
        return self.batch(state, {_Q: question}, images=images)[_Q]


def _to_sdk_question(sdk: Any, question: Question) -> Any:
    """Translate a protocol question into its SDK question type."""
    if isinstance(question, ChoiceQ):
        # Jev accepts a None description; an option with no description is sent
        # as its own key so the model still sees a meaningful label.
        return sdk.Choice(
            instructions=question.instructions,
            criteria={
                k: (v if v is not None else k) for k, v in question.options.items()
            },
        )
    if isinstance(question, TruthQ):
        return sdk.Noul(instructions=question.instructions)
    if isinstance(question, RateQ):
        return sdk.Score(
            instructions=question.instructions, criteria=list(question.rubric)
        )
    raise TypeError(f"unknown question type: {type(question).__name__}")


def _to_verdict(answer: Any, question: Question, latency_ms: float, provider: str) -> Verdict[Any]:
    """Translate one SDK answer into a :class:`Verdict`.

    A missing answer (the API did not return the key we asked about) becomes a
    ``None``-valued verdict rather than a ``KeyError``, so one unanswered
    question in a batch cannot take down the rest.
    """
    if answer is None:
        return Verdict(
            value=None,
            confidence=None,
            probabilities=None,
            provider=provider,
            latency_ms=latency_ms,
        )

    if isinstance(question, TruthQ):
        noul = float(getattr(answer, "noul", 0.0))
        return Verdict(
            value=noul,
            confidence=noul,
            probabilities={_YES: noul, _NO: 1.0 - noul},
            provider=provider,
            latency_ms=latency_ms,
        )

    if isinstance(question, RateQ):
        return _score_verdict(answer, question.rubric, latency_ms, provider)

    return Verdict(
        value=getattr(answer, "choice", None),
        confidence=getattr(answer, "confidence", None),
        probabilities=getattr(answer, "probabilities", None),
        provider=provider,
        latency_ms=latency_ms,
    )


def _score_verdict(answer: Any, rubric: list[str], latency_ms: float, provider: str) -> Verdict[str]:
    """Resolve a float ``ScoreAnswer`` back to a rubric label.

    Jev returns ``score`` as a float with an int-keyed ``legend``, but the
    protocol promises ``Verdict[str]``.  The score is rounded to the nearest
    index and clamped into range, so a score outside the legend cannot raise.
    Probabilities are re-keyed from indices to labels for the same reason.
    """
    score = getattr(answer, "score", None)
    if score is None or not rubric:
        label: str | None = None
    else:
        index = max(0, min(len(rubric) - 1, round(float(score))))
        label = rubric[index]

    raw_probabilities = getattr(answer, "probabilities", None) or {}
    probabilities: dict[str, float] | None = None
    if raw_probabilities:
        probabilities = {}
        for key, value in raw_probabilities.items():
            try:
                probabilities[rubric[int(key)]] = float(value)
            except (ValueError, TypeError, IndexError):
                # An index outside the rubric means the legend and the rubric
                # disagree; keep the raw key rather than dropping the mass.
                probabilities[str(key)] = float(value)

    return Verdict(
        value=label,
        confidence=getattr(answer, "confidence", None),
        probabilities=probabilities,
        provider=provider,
        latency_ms=latency_ms,
    )
