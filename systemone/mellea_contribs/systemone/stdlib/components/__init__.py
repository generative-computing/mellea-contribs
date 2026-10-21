"""Decision functions: ``@decisive``.

A sibling of Mellea's ``@generative``.  It derives the provider call from the
decorated stub's return annotation and rejects unsupported annotations at
decoration time.
"""

from mellea_contribs.systemone.stdlib.components.decisive import (
    DEFAULT_THRESHOLD,
    DecisiveStub,
    Image,
    decisive,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "DecisiveStub",
    "Image",
    "decisive",
]
