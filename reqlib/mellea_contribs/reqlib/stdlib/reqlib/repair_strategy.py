"""Context-guided repair strategy for Mellea agents.

Purpose: Provides :class:`SimpleContextGuidedRepairStrategy`, a multi-attempt
repair strategy that injects structured failure feedback into the
``Instruction`` repair slot on every failed attempt.

What it does:
- On each failed attempt, formats the model's previous response and the
  specific validation failure reasons into a repair prompt.
- Resets the context to the original ``old_ctx`` on every retry, keeping each
  attempt stateless — safe for ``SimpleContext`` where there is no conversation
  history.
- Ships two built-in repair templates: a generic one
  (``_DEFAULT_REPAIR_TEMPLATE``) and a faithfulness-aware one
  (``_REPAIR_TEMPLATE_V2``) that understands the structured unsupported-claim
  fields produced by
  :mod:`~mellea_contribs.reqlib.stdlib.reqlib.faithfulness_requirement`.

When should an agent use it?
- When the agent uses ``instruct()`` with one or more ``Requirement`` checks
  and needs guided repair rather than blind rejection-sampling.
- When the generation context is stateless (``SimpleContext``) — the strategy
  is safe to use with stateful contexts too, but ``old_ctx`` reuse is what
  makes it especially suitable for the stateless case.
- Pair with :class:`~mellea_contribs.reqlib.stdlib.reqlib.faithfulness_requirement.FaithfulnessRequirement`
  and :data:`_REPAIR_TEMPLATE_V2` for hallucination-repair loops.
"""

from __future__ import annotations

from mellea.core import (
    Component,
    ComputedModelOutputThunk,
    Context,
    Requirement,
    ValidationResult,
)
from mellea.stdlib.components import Instruction
from mellea.stdlib.sampling import BaseSamplingStrategy

# ---------------------------------------------------------------------------
# Built-in repair templates
# ---------------------------------------------------------------------------

_DEFAULT_REPAIR_TEMPLATE = """\
Your previous response was:
---
{last_response}
---

It did not satisfy the following requirements:
{failure_reasons}

Please rewrite your response so that it satisfies every requirement listed above."""

# V2: faithfulness-aware repair template.
# The failure reason will include unsupported_claim, actual_claim, and
# should_be_replaced_by fields from the faithfulness evaluator.
# Use should_be_replaced_by to patch each hallucinated claim directly —
# "remove" means the claim should be deleted from the response entirely.
REPAIR_TEMPLATE_V2 = """\
You are given a previous response and the failures seen in its verification. The faithfulness \
failure will show unsupported claims, actual_claims, and should_be_replaced_by. The field \
should_be_replaced_by has values which can either be content to replace the unsupported claim \
with, or the word "remove" which means the unsupported_claim should be deleted from the response.

Your previous response was:
---
{last_response}
---

It did not satisfy the following requirements:
{failure_reasons}

Please rewrite your response so that it satisfies every requirement listed above."""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class SimpleContextGuidedRepairStrategy(BaseSamplingStrategy):
    """Guided multi-attempt repair strategy compatible with SimpleContext (and any context type).

    On each failed attempt the ``Instruction``'s repair slot is populated with
    the model's previous response and the specific validation failure reasons.
    The context is reset to the original ``old_ctx`` on every retry, keeping
    each attempt stateless — safe for ``SimpleContext`` where there is no
    conversation history.

    Example::

        from mellea_contribs.reqlib.stdlib.reqlib.repair_strategy import (
            SimpleContextGuidedRepairStrategy,
            _REPAIR_TEMPLATE_V2,
        )

        result = session.instruct(
            "Summarize ...",
            requirements=[faithfulness_req],
            strategy=SimpleContextGuidedRepairStrategy(
                loop_budget=3,
                repair_template=_REPAIR_TEMPLATE_V2,
            ),
            return_sampling_results=True,
        )
    """

    def __init__(
        self, *, loop_budget: int = 3, repair_template: str | None = None, **kwargs
    ):
        """
        Args:
            loop_budget: Maximum number of generation attempts.
            repair_template: A Python format-string with ``{last_response}``
                and ``{failure_reasons}`` placeholders. Defaults to
                :data:`_DEFAULT_REPAIR_TEMPLATE`.
            **kwargs: Forwarded to ``BaseSamplingStrategy`` (e.g.
                ``requirements``).
        """
        super().__init__(loop_budget=loop_budget, **kwargs)
        self._repair_template = repair_template or _DEFAULT_REPAIR_TEMPLATE

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ComputedModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        """Return the index of the attempt that passed the most requirements.

        Args:
            sampled_actions: The ``Component`` used in each attempt.
            sampled_results: The model outputs for each attempt.
            sampled_val: Validation results for each attempt.

        Returns:
            Index of the best attempt (0-based).
        """
        if not sampled_val:
            return 0
        return max(
            range(len(sampled_val)),
            key=lambda i: sum(1 for _, r in sampled_val[i] if bool(r)),
        )

    def repair(
        self,
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ComputedModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Build repair feedback from the last response and failure reasons.

        Injects the repair string into the ``Instruction``'s repair slot and
        returns ``old_ctx`` (not ``new_ctx``) so that each attempt starts from
        a clean, stateless context — safe for ``SimpleContext``.

        Args:
            old_ctx: The original context before any attempts were made.
            new_ctx: The context after the last attempt (not used).
            past_actions: The ``Component`` used in each prior attempt.
            past_results: The model outputs for each prior attempt.
            past_val: Validation results paired with requirements for each
                prior attempt.

        Returns:
            A ``(repaired_action, old_ctx)`` tuple ready for the next attempt.
        """
        last_action = past_actions[-1]

        if not isinstance(last_action, Instruction):
            return last_action, old_ctx

        last_response = str(past_results[-1].value)

        failure_lines = []
        for req, val_result in past_val[-1]:
            if not bool(val_result):
                reason = val_result.reason or "(no reason provided)"
                failure_lines.append(f"- [{req.description}]: {reason}")

        failure_reasons = "\n".join(failure_lines) or "- (unknown failure)"

        repair_string = self._repair_template.format(
            last_response=last_response,
            failure_reasons=failure_reasons,
        )
        repaired_action = last_action.copy_and_repair(repair_string=repair_string)

        # Return old_ctx (not new_ctx) — keeps SimpleContext stateless across retries
        return repaired_action, old_ctx
