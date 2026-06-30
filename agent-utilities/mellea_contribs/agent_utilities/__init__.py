"""Agent-side utilities for Mellea: selectors, evaluators, and robustness tools.

The selectors (`top_k`, `double_round_robin`) are eagerly available. The
benchdrift integration (`benchdrift_runner`) is intentionally NOT
re-exported here — it depends on the optional `[robustness]` extra,
which not every consumer wants installed. Import it explicitly:

    from mellea_contribs.agent_utilities.core.benchdrift_runner import ...
"""

from mellea_contribs.agent_utilities.core import double_round_robin, top_k

__all__ = ["double_round_robin", "top_k"]
