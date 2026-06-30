"""Mellea DSPy Integration.

This package provides DSPy-compatible language model classes that wrap Mellea,
enabling DSPy applications to use Mellea's generative programming capabilities.
"""

from mellea_contribs.dspy.core.lm import MelleaLM
from mellea_contribs.dspy.core.verification import (
    MelleaBestOfN,
    MelleaRefine,
    create_reward_fn,
)

__all__ = ["MelleaBestOfN", "MelleaLM", "MelleaRefine", "create_reward_fn"]
__version__ = "0.1.0"
