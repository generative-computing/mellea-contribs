"""mellea-contribs-systemone

Calibrated decision models (GLiNER2, Jev) as typed Mellea decision functions.

Public surface:

    from mellea_contribs.systemone import (
        # protocol
        Judge,
        Verdict,
        Question,
        # errors
        UnsupportedReturnType,
        JudgeUnavailable,
        # providers (import-guarded; raises JudgeUnavailable if extra missing)
        FakeJudge,
        Gliner2Judge,   # requires [gliner2] or [local]
        JevJudge,       # requires [jev]
        OllamaJudge,    # requires [ollama]
        # decision functions
        decisive,
        Image,
    )
"""

from mellea_contribs.systemone.backends.fake import FakeJudge
from mellea_contribs.systemone.backends.gliner2 import Gliner2Judge
from mellea_contribs.systemone.backends.jev import JevJudge
from mellea_contribs.systemone.backends.ollama import OllamaJudge
from mellea_contribs.systemone.core.errors import (
    JudgeUnavailable,
    UnsupportedReturnType,
)
from mellea_contribs.systemone.core.judge import (
    ChoiceQ,
    Image,
    Judge,
    Question,
    RateQ,
    TruthQ,
    Verdict,
)
from mellea_contribs.systemone.stdlib.components.decisive import decisive

# Grouped by role rather than sorted alphabetically: the grouping is the
# package's public map, and reordering it would lose that.
__all__ = [  # noqa: RUF022
    # protocols / types
    "Judge",
    "Verdict",
    "Image",
    "Question",
    "ChoiceQ",
    "TruthQ",
    "RateQ",
    # errors
    "UnsupportedReturnType",
    "JudgeUnavailable",
    # providers always available
    "FakeJudge",
    # providers whose SDK is imported lazily: importing the class is always
    # safe; constructing one without the extra raises JudgeUnavailable.
    "Gliner2Judge",
    "JevJudge",
    "OllamaJudge",
    # decision functions
    "decisive",
]
