"""``@decisive`` — a generative function whose answer is a closed decision.

``@generative`` turns a type-annotated stub into an LLM call.  ``@decisive``
turns the same stub into a *judge* call: the return annotation selects the
provider primitive, and the answer comes back with a confidence attached.

::

    @decisive
    def triage(ticket: str) -> Literal["billing", "bug", "feature"]:
        '''Route a support ticket to the owning team.'''

    triage(judge=g, ticket="I was charged twice")          # -> "billing"
    triage.verdict(judge=g, ticket="I was charged twice")  # -> Verdict(...)

Why this is a sibling of ``@generative`` rather than a flag on it: the two have
different failure modes.  A generation can be wrong in unbounded ways, so it
needs ``requirements=`` and a repair loop.  A decision is schema-guaranteed to
be one of the declared options — it cannot be malformed, only *unconfident*.
So ``@decisive`` takes no ``requirements=`` or ``strategy=``; low confidence is
surfaced on the :class:`Verdict` for the caller to act on.

Return-type mapping
-------------------
===========================================  =====================
Return annotation                            Judge primitive
===========================================  =====================
``Literal["a", "b"]`` / ``StrEnum``          :meth:`Judge.choose`
``bool``                                     :meth:`Judge.truth`
``float``                                    :meth:`Judge.truth`
``Literal[...]`` with ``rubric=``            :meth:`Judge.rate`
===========================================  =====================

Anything else raises
:class:`~mellea_contribs.systemone.core.errors.UnsupportedReturnType` **at
decoration time**.
"""

from __future__ import annotations

import enum
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any, get_type_hints, overload

from mellea_contribs.systemone.core.errors import UnsupportedReturnType
from mellea_contribs.systemone.core.judge import Image, Judge, Verdict
from mellea_contribs.systemone.core.signatures import (
    GENERATIVE_HINT,
    literal_values,
    render_annotation,
    return_annotation,
    strip_annotated,
)

__all__ = ["DecisiveStub", "Image", "decisive"]

#: Default cut-off turning a ``truth`` score into a ``bool``.
DEFAULT_THRESHOLD = 0.5


def _image_param_names(func: Callable[..., Any]) -> frozenset[str]:
    """Return parameter names annotated with :class:`Image`."""
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        return frozenset()
    names: list[str] = []
    for name, ann in hints.items():
        if name == "return":
            continue
        _inner, meta = strip_annotated(ann)
        if any(isinstance(m, Image) for m in meta):
            names.append(name)
    return frozenset(names)


def _build_state(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    image_params: frozenset[str],
) -> tuple[dict[str, Any], list[bytes] | None]:
    """Bind call arguments and separate image parameters from state.

    Args:
        func: The decorated stub, whose signature defines the parameters.
        args: Positional arguments from the call site.
        kwargs: Keyword arguments from the call site.
        image_params: Parameter names annotated with :class:`Image`.

    Returns:
        ``(state_dict, images_or_none)``.  Image-annotated parameters are
        extracted into a flat list of ``bytes``; everything else goes into
        the state dict.

    Raises:
        TypeError: If the arguments do not satisfy ``func``'s signature.
    """
    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()

    if not image_params:
        return dict(bound.arguments), None

    state: dict[str, Any] = {}
    images: list[bytes] = []
    for name, value in bound.arguments.items():
        if name in image_params:
            if isinstance(value, bytes):
                images.append(value)
            elif isinstance(value, (list, tuple)):
                images.extend(value)
        else:
            state[name] = value
    return state, images or None


def _build_instructions(func: Callable[..., Any]) -> str:
    """Compose the instruction string from the stub's name and docstring.

    The docstring is the task description; the name is included because it
    often carries information the docstring omits (``is_urgent`` tells the model
    the polarity of the question).

    Args:
        func: The decorated stub.

    Returns:
        The instruction string sent to the judge.
    """
    doc = inspect.getdoc(func) or ""
    name = func.__name__.replace("_", " ")
    if not doc:
        return name
    return f"{name}: {doc}"


class DecisiveStub[**P, R]:
    """Callable wrapper returned by :func:`decisive`.

    Calling the instance returns the bare decision; :meth:`verdict` returns the
    full :class:`~mellea_contribs.systemone.core.judge.Verdict` including
    confidence, the probability distribution, and the provider tag.

    Attributes:
        func: The original undecorated stub.
        mode: Which judge primitive this stub dispatches to — ``"choose"``,
            ``"truth"``, or ``"rate"``.
        options: Declared labels for ``choose`` / ``rate`` modes, else ``None``.
        threshold: Cut-off applied to a ``truth`` score for ``bool`` returns.
    """

    def __init__(
        self,
        func: Callable[P, R],
        *,
        mode: str,
        options: list[str] | None,
        enum_type: type[enum.Enum] | None,
        returns_bool: bool,
        threshold: float,
    ) -> None:
        """
        Args:
            func: The stub being decorated.
            mode: ``"choose"``, ``"truth"`` or ``"rate"``.
            options: Label list for ``choose`` / ``rate``.
            enum_type: The ``StrEnum`` to coerce back into, if the annotation
                was an enum rather than a ``Literal``.
            returns_bool: Whether to threshold the ``truth`` score into a bool.
            threshold: The cut-off used when ``returns_bool`` is set.
        """
        self.func = func
        self.mode = mode
        self.options = options
        self.enum_type = enum_type
        self.returns_bool = returns_bool
        self.threshold = threshold
        self._image_params = _image_param_names(func)
        functools.update_wrapper(self, func)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, judge: Judge | None = None, /, *args: Any, **kwargs: Any) -> R:
        """Run the decision and return the bare value.

        Args:
            judge: The provider to ask.  Accepted positionally (mirroring
                ``@generative``, which takes its session positionally) or as the
                ``judge=`` keyword.
            *args: Positional arguments for the decorated stub.
            **kwargs: Keyword arguments for the decorated stub.

        Returns:
            The decision, coerced to the declared return type.
        """
        return self.verdict(judge, *args, **kwargs).value

    def verdict(
        self, judge: Judge | None = None, /, *args: Any, **kwargs: Any
    ) -> Verdict[R]:
        """Run the decision and return the full verdict.

        Args:
            judge: The provider to ask, positionally or as ``judge=``.
            *args: Positional arguments for the decorated stub.
            **kwargs: Keyword arguments for the decorated stub.

        Returns:
            The provider's :class:`Verdict`, with ``value`` coerced to the
            declared return type and all provider metadata preserved.

        Raises:
            TypeError: If no judge is supplied, or the stub's arguments are
                not satisfied.
        """
        if judge is None:
            judge = kwargs.pop("judge", None)
        if judge is None:
            raise TypeError(
                f"{self.func.__name__}() requires a judge: pass it positionally "
                f"or as judge=..."
            )

        state, images = _build_state(self.func, args, kwargs, self._image_params)
        instructions = _build_instructions(self.func)

        if self.mode == "choose":
            assert self.options is not None
            raw = judge.choose(
                state=state,
                options={label: None for label in self.options},
                instructions=instructions,
                images=images,
            )
        elif self.mode == "rate":
            assert self.options is not None
            raw = judge.rate(
                state=state, rubric=list(self.options), instructions=instructions,
                images=images,
            )
        else:
            raw = judge.truth(state=state, instructions=instructions, images=images)

        return self._coerce(raw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _coerce(self, raw: Verdict[Any]) -> Verdict[R]:
        """Map the provider's raw value onto the declared return type.

        ``truth`` returns a float; a ``bool``-annotated stub must threshold it.
        ``choose`` returns a label string; a ``StrEnum``-annotated stub must map
        it back to the member.  Confidence is never rewritten — for a
        thresholded bool the float stays visible in ``confidence`` so callers
        can see how close the decision was.

        Args:
            raw: The verdict as returned by the provider.

        Returns:
            A verdict whose ``value`` has the declared type.
        """
        value: Any = raw.value

        if self.returns_bool:
            score = float(value) if value is not None else 0.0
            value = score >= self.threshold
        elif self.enum_type is not None:
            value = self.enum_type(value)

        if value is raw.value:
            return raw
        return Verdict(
            value=value,
            confidence=raw.confidence,
            probabilities=raw.probabilities,
            provider=raw.provider,
            latency_ms=raw.latency_ms,
        )


def _classify(func: Callable[..., Any], rubric: Sequence[str] | None) -> dict[str, Any]:
    """Resolve a stub's return annotation into :class:`DecisiveStub` parameters.

    Args:
        func: The stub being decorated.
        rubric: Explicit ordinal rubric, which switches ``Literal`` returns from
            ``choose`` to ``rate``.

    Returns:
        Keyword arguments for the :class:`DecisiveStub` constructor.

    Raises:
        UnsupportedReturnType: For any annotation outside the supported table,
            or a ``rubric`` that disagrees with the annotation's values.
    """
    annotation = return_annotation(func)
    inner, _meta = strip_annotated(annotation)

    enum_type = (
        inner if isinstance(inner, type) and issubclass(inner, enum.Enum) else None
    )
    values = literal_values(inner)

    if values is not None:
        if rubric is not None:
            if list(rubric) != values:
                raise UnsupportedReturnType(
                    f"{func.__name__}: rubric {list(rubric)!r} does not match the "
                    f"return annotation's values {values!r}. A rate verdict is one "
                    f"of the rubric labels, so a mismatch could never produce a "
                    f"value of the declared type."
                )
            return {
                "mode": "rate",
                "options": values,
                "enum_type": enum_type,
                "returns_bool": False,
                "threshold": DEFAULT_THRESHOLD,
            }
        return {
            "mode": "choose",
            "options": values,
            "enum_type": enum_type,
            "returns_bool": False,
            "threshold": DEFAULT_THRESHOLD,
        }

    if inner is bool:
        return {
            "mode": "truth",
            "options": None,
            "enum_type": None,
            "returns_bool": True,
            "threshold": DEFAULT_THRESHOLD,
        }

    if inner is float:
        return {
            "mode": "truth",
            "options": None,
            "enum_type": None,
            "returns_bool": False,
            "threshold": DEFAULT_THRESHOLD,
        }

    if rubric is not None:
        raise UnsupportedReturnType(
            f"{func.__name__} returns {render_annotation(annotation)}; rubric= "
            f"requires a Literal of strings matching the rubric labels."
        )

    raise UnsupportedReturnType(
        f"{func.__name__} returns {render_annotation(annotation)}, which "
        f"@decisive cannot express. Supported: Literal of strings, StrEnum, "
        f"bool, float. {GENERATIVE_HINT}"
    )


@overload
def decisive[**P, R](func: Callable[P, R], /) -> DecisiveStub[P, R]: ...


@overload
def decisive[**P, R](
    *, threshold: float = ..., rubric: Sequence[str] | None = ...
) -> Callable[[Callable[P, R]], DecisiveStub[P, R]]: ...


def decisive(
    func: Callable[..., Any] | None = None,
    /,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    rubric: Sequence[str] | None = None,
) -> Any:
    """Turn a type-annotated stub into a judge-backed decision function.

    Usable bare (``@decisive``) or with arguments
    (``@decisive(threshold=0.9)``).

    Args:
        func: The stub, when applied bare.
        threshold: For ``bool`` returns, the ``truth`` score at or above which
            the answer is ``True``.  Ignored for other return types.
        rubric: Ordered rating labels.  Supplying this switches a ``Literal``
            return from :meth:`Judge.choose` to :meth:`Judge.rate`, which tells
            the provider the labels are ordinal rather than unordered.

    Returns:
        A :class:`DecisiveStub`, or a decorator producing one.

    Raises:
        UnsupportedReturnType: At decoration time, if the return annotation is
            outside the supported set or conflicts with ``rubric``.

    Example::

        @decisive(rubric=["low", "medium", "high"])
        def severity(report: str) -> Literal["low", "medium", "high"]:
            '''How severe the described problem is.'''

        v = severity.verdict(judge=g, report=text)
        if (v.confidence or 0) < 0.8:
            escalate()
    """

    def wrap(target: Callable[..., Any]) -> DecisiveStub[Any, Any]:
        params = _classify(target, rubric)
        params["threshold"] = threshold
        return DecisiveStub(target, **params)

    if func is not None:
        return wrap(func)
    return wrap
