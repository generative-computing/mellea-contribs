"""FakeJudge — scripted provider for hermetic unit tests.

:class:`FakeJudge` implements :class:`~mellea_contribs.systemone.core.judge.Judge`.
Every method is backed by a pre-scripted queue of :class:`~mellea_contribs.systemone.core.judge.Verdict`
values, allowing tests to assert on exact ``Verdict`` sequences with no network
access and no torch dependency.

Usage pattern::

    from mellea_contribs.systemone.backends.fake import FakeJudge
    from mellea_contribs.systemone.core.judge import Verdict

    judge = FakeJudge(
        choose_verdicts=[
            Verdict(value="supports",     confidence=0.92, probabilities=None, provider="fake", latency_ms=1.0),
            Verdict(value="contradicts",  confidence=0.88, probabilities=None, provider="fake", latency_ms=1.0),
        ],
        truth_verdicts=[
            Verdict(value=0.95, confidence=0.95, probabilities=None, provider="fake", latency_ms=1.0),
        ],
    )

    v = judge.choose(state="some text", options={"a": None, "b": None}, instructions="pick one")
    assert v.value == "supports"

``FakeJudge`` also records every call in :attr:`calls` so tests can verify the
sequence of method invocations and their arguments.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from mellea_contribs.systemone.core.judge import (
    ChoiceQ,
    Question,
    RateQ,
    TruthQ,
    Verdict,
)

__all__ = ["FakeCall", "FakeJudge"]


@dataclass
class FakeCall:
    """Record of a single method call made to :class:`FakeJudge`.

    Attributes:
        method: Name of the method called (``"choose"``, ``"truth"``,
            ``"rate"``, ``"batch"``).
        kwargs: All arguments passed, as keyword arguments.
    """

    method: str
    kwargs: dict[str, Any]


class FakeJudge:
    """Scripted provider implementing :class:`Judge`.

    Each method drains from its own :class:`collections.deque` of
    pre-scripted :class:`~mellea_contribs.systemone.core.judge.Verdict` values.
    When a deque is empty and the method is called, a ``ValueError`` is raised
    — the test has not scripted enough responses.

    ``batch`` drains one verdict per question from whichever underlying queue
    matches the question type (``ChoiceQ`` → ``choose_verdicts``,
    ``TruthQ`` → ``truth_verdicts``, ``RateQ`` → ``rate_verdicts``).

    Attributes:
        name: Always ``"fake"``.
        calls: Ordered list of :class:`FakeCall` records — inspected in tests
            to verify call sequences.
    """

    name: str = "fake"

    def __init__(
        self,
        *,
        choose_verdicts: list[Verdict[str]] | None = None,
        truth_verdicts: list[Verdict[float]] | None = None,
        rate_verdicts: list[Verdict[str]] | None = None,
    ) -> None:
        """
        Args:
            choose_verdicts: Pre-scripted responses for :meth:`choose` calls,
                in order.
            truth_verdicts: Pre-scripted responses for :meth:`truth` calls.
            rate_verdicts: Pre-scripted responses for :meth:`rate` calls.
        """
        self._choose: deque[Verdict[str]] = deque(choose_verdicts or [])
        self._truth: deque[Verdict[float]] = deque(truth_verdicts or [])
        self._rate: deque[Verdict[str]] = deque(rate_verdicts or [])
        self.calls: list[FakeCall] = []

    # ------------------------------------------------------------------
    # Judge protocol
    # ------------------------------------------------------------------

    def choose(
        self,
        state: str | dict[str, Any],
        options: dict[str, str | None],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[str]:
        """Return the next scripted choose verdict.

        Args:
            state: Document or context (recorded in :attr:`calls`).
            options: Closed-choice options (recorded in :attr:`calls`).
            instructions: Task instruction (recorded in :attr:`calls`).
            images: Optional image bytes (recorded in :attr:`calls`).

        Returns:
            The next :class:`Verdict` from the ``choose_verdicts`` queue.

        Raises:
            ValueError: When the queue is empty.
        """
        self.calls.append(
            FakeCall(
                "choose",
                {"state": state, "options": options, "instructions": instructions, "images": images},
            )
        )
        if not self._choose:
            raise ValueError("FakeJudge.choose: no more scripted verdicts")
        return self._choose.popleft()

    def truth(
        self,
        state: str | dict[str, Any],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[float]:
        """Return the next scripted truth verdict.

        Args:
            state: Document or context.
            instructions: The boolean claim to evaluate.
            images: Optional image bytes (recorded in :attr:`calls`).

        Returns:
            The next :class:`Verdict` from the ``truth_verdicts`` queue.

        Raises:
            ValueError: When the queue is empty.
        """
        self.calls.append(
            FakeCall("truth", {"state": state, "instructions": instructions, "images": images})
        )
        if not self._truth:
            raise ValueError("FakeJudge.truth: no more scripted verdicts")
        return self._truth.popleft()

    def rate(
        self,
        state: str | dict[str, Any],
        rubric: list[str],
        instructions: str,
        *,
        images: list[bytes] | None = None,
    ) -> Verdict[str]:
        """Return the next scripted rate verdict.

        Args:
            state: Document or context.
            rubric: Ordered rating labels.
            instructions: Task instruction.
            images: Optional image bytes (recorded in :attr:`calls`).

        Returns:
            The next :class:`Verdict` from the ``rate_verdicts`` queue.

        Raises:
            ValueError: When the queue is empty.
        """
        self.calls.append(
            FakeCall(
                "rate", {"state": state, "rubric": rubric, "instructions": instructions, "images": images}
            )
        )
        if not self._rate:
            raise ValueError("FakeJudge.rate: no more scripted verdicts")
        return self._rate.popleft()

    def batch(
        self,
        state: str | dict[str, Any],
        questions: dict[str, Question],
        *,
        images: list[bytes] | None = None,
    ) -> dict[str, Verdict[Any]]:
        """Drain one verdict per question from the matching type queue.

        Each question dispatches to the same queue as its standalone method:
        ``ChoiceQ`` → ``choose_verdicts``, ``TruthQ`` → ``truth_verdicts``,
        ``RateQ`` → ``rate_verdicts``.

        Args:
            state: Document or context shared across all questions.
            questions: Mapping from question key to question descriptor.
            images: Optional image bytes (recorded in :attr:`calls`).

        Returns:
            Dict mapping each key to its :class:`Verdict`.

        Raises:
            ValueError: When any per-type queue is exhausted before all
                questions of that type are answered.
            TypeError: If an unknown question type is encountered.
        """
        self.calls.append(FakeCall("batch", {"state": state, "questions": questions, "images": images}))
        results: dict[str, Verdict[Any]] = {}
        for key, q in questions.items():
            if isinstance(q, ChoiceQ):
                if not self._choose:
                    raise ValueError(
                        f"FakeJudge.batch: no more choose verdicts for question '{key}'"
                    )
                results[key] = self._choose.popleft()
            elif isinstance(q, TruthQ):
                if not self._truth:
                    raise ValueError(
                        f"FakeJudge.batch: no more truth verdicts for question '{key}'"
                    )
                results[key] = self._truth.popleft()
            elif isinstance(q, RateQ):
                if not self._rate:
                    raise ValueError(
                        f"FakeJudge.batch: no more rate verdicts for question '{key}'"
                    )
                results[key] = self._rate.popleft()
            else:
                raise TypeError(f"FakeJudge.batch: unknown question type: {type(q)}")
        return results
