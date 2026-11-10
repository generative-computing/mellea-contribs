"""Exceptions for mellea-contribs-systemone.

Two exception types cover the only two failure modes this package owns:

:class:`JudgeUnavailable`
    Raised when a provider's optional dependency is not installed or its
    credentials are missing.

:class:`UnsupportedReturnType`
    Raised **at decoration time** when ``@decisive`` is applied to a function whose return annotation is outside the supported
    type set.  Failing at import rather than at call time is deliberate:
    the restriction is a documented contract, and import-time failure makes
    it impossible to ship code that only breaks in production.
"""

from __future__ import annotations

__all__ = ["JudgeUnavailable", "UnsupportedReturnType"]


class JudgeUnavailable(RuntimeError):
    """Raised when a provider's SDK is not installed or cannot be configured.

    For example, calling ``Gliner2Judge(...)`` without
    ``pip install "mellea-contribs-systemone[gliner2]"``, or ``JevJudge()``
    with no ``TYPESAFE_API_KEY``.

    The exception message always includes a ``pip install`` hint when the cause
    is a missing SDK.

    Example::

        try:
            judge = Gliner2Judge()
        except JudgeUnavailable as exc:
            print(exc)
            # "gliner2 is not installed. Install it with: ..."
    """


class UnsupportedReturnType(TypeError):
    """Raised at decoration time for unsupported return type annotations.

    ``@decisive`` enforces its accepted return types at import time so that mis-decorated functions are caught before they ever
    reach production.

    Supported return types are documented in the spec; anything else raises
    this exception with a message naming the offending annotation and pointing
    at ``@generative`` for prose-returning functions.

    Example::

        # Raises UnsupportedReturnType at import time:
        @decisive
        def bad(text: str) -> list[str]:   # list[str] is not supported
            ...

        # Message: "Unsupported return type for @decisive: list[str].
        #  Use @generative for prose-returning functions."
    """
