"""Return-annotation introspection for ``@decisive``.

The decorator must answer one question at decoration time: *which provider
primitive does this return annotation map to, and is it supported at all?*

The mapping is deliberately narrow.  A judge answers closed questions; anything
open-ended belongs to ``@generative``.  The full table lives in the spec, and
:func:`render_annotation` exists so every rejection message can name the
annotation the user actually wrote.
"""

from __future__ import annotations

import enum
import inspect
from collections.abc import Callable
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from mellea_contribs.systemone.core.errors import UnsupportedReturnType

__all__ = [
    "GENERATIVE_HINT",
    "literal_values",
    "render_annotation",
    "return_annotation",
    "strip_annotated",
]

#: Appended to rejection messages so the user has somewhere to go next.
GENERATIVE_HINT = (
    "Use @generative from mellea.stdlib.components.genstub for functions that "
    "return free text or open-ended structures."
)


def render_annotation(annotation: Any) -> str:
    """Render an annotation the way a user would have written it.

    ``typing`` reprs are noisy (``typing.Literal['a', 'b']``), and a bare class
    reprs as ``<class 'str'>``.  Error messages are much easier to act on when
    they echo the source text, so this normalises both cases.

    Args:
        annotation: Any type annotation object.

    Returns:
        A short source-like rendering, e.g. ``"list[str]"``, ``"Literal['a']"``.
    """
    if annotation is inspect.Signature.empty:
        return "<no annotation>"
    if annotation is None or annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    text = str(annotation)
    return text.removeprefix("typing.")


def return_annotation(func: Callable[..., Any]) -> Any:
    """Resolve ``func``'s return annotation, following string forward refs.

    ``from __future__ import annotations`` turns every annotation into a string,
    so a decorator that reads ``__annotations__`` directly sees ``"bool"`` and
    not :class:`bool`.  :func:`typing.get_type_hints` resolves those against the
    function's own module globals.

    Args:
        func: The function being decorated.

    Returns:
        The resolved return annotation object.

    Raises:
        UnsupportedReturnType: If ``func`` has no return annotation, or one that
            cannot be resolved (e.g. a forward reference to a name that is not
            importable at decoration time).
    """
    raw = getattr(func, "__annotations__", {})
    if "return" not in raw:
        raise UnsupportedReturnType(
            f"{func.__name__} has no return annotation. "
            f"@decisive derives the provider call from the return type, so "
            f"it is required."
        )
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:  # pragma: no cover - depends on user's module
        raise UnsupportedReturnType(
            f"could not resolve the return annotation of {func.__name__} "
            f"({raw['return']!r}): {exc}"
        ) from exc
    return hints["return"]


def strip_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Split ``Annotated[X, *meta]`` into ``(X, meta)``.

    Args:
        annotation: Possibly-``Annotated`` annotation.

    Returns:
        ``(underlying_type, metadata_tuple)``.  For a non-``Annotated``
        annotation the metadata tuple is empty.
    """
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        return args[0], tuple(args[1:])
    return annotation, ()


def literal_values(annotation: Any) -> list[str] | None:
    """Return the string values of a ``Literal`` or :class:`enum.StrEnum`.

    Args:
        annotation: The annotation to inspect.

    Returns:
        Ordered list of string values, or ``None`` if ``annotation`` is neither
        a ``Literal`` nor a ``StrEnum`` subclass.

    Raises:
        UnsupportedReturnType: If the annotation *is* a ``Literal`` but contains
            a non-string member.  Both providers label with strings, so an
            ``int`` label has no round-trip.
    """
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = [m.value for m in annotation]
        if not all(isinstance(v, str) for v in values):
            raise UnsupportedReturnType(
                f"{annotation.__name__} has non-string member values. "
                f"Judges label with strings; use a StrEnum."
            )
        return values

    if get_origin(annotation) is Literal:
        args = get_args(annotation)
        if not args:
            raise UnsupportedReturnType("Literal[] with no members is not a decision.")
        non_str = [a for a in args if not isinstance(a, str)]
        if non_str:
            raise UnsupportedReturnType(
                f"Literal members must be strings; got {non_str!r}. "
                f"Judges return label strings, so non-string members cannot "
                f"round-trip. Use a Literal of strings and convert afterwards."
            )
        return list(args)

    return None
