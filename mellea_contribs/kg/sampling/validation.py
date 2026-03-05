"""Query validation sampling strategy for knowledge graph queries.

Implements the Instruct/Validate/Repair loop for LLM-guided query generation.
"""

from mellea.stdlib.components import CBlock, Component, ModelOutputThunk
from mellea.stdlib.context import Context
from mellea.stdlib.requirements import Requirement, ValidationResult
from mellea.stdlib.sampling import BaseSamplingStrategy

from mellea_contribs.kg.graph_dbs.base import GraphBackend


class QueryValidationStrategy(BaseSamplingStrategy):
    """Sampling strategy for generating and validating graph queries.

    Uses the Instruct/Validate/Repair pattern:
    1. Generate a query from natural language.
    2. Validate syntax and executability.
    3. If invalid, repair using error feedback.

    Args:
        backend: Graph backend used to validate queries.
        loop_budget: Maximum number of repair attempts.
        requirements: Query validation requirements to check.
    """

    def __init__(
        self,
        backend: GraphBackend,
        loop_budget: int = 3,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize QueryValidationStrategy.

        Args:
            backend: Graph backend for validation.
            loop_budget: Max repair attempts.
            requirements: Query validation requirements.
        """
        super().__init__(loop_budget=loop_budget, requirements=requirements)
        self._backend = backend

    @staticmethod
    def repair(
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Repair a failed query using validation error feedback.

        Constructs a new repair instruction from the last validation failures
        and appends it to the context for the next generation attempt.

        Args:
            old_ctx: Context before the last action and output.
            new_ctx: Context including the last action and output.
            past_actions: Actions executed so far without success.
            past_results: Generation results for those actions.
            past_val: Validation results for each attempt.

        Returns:
            A tuple of (repair instruction component, updated context).
        """
        last_validation = past_val[-1]

        error_messages = [
            result.reason
            for _, result in last_validation
            if not bool(result) and result.reason
        ]

        failed_query = str(past_results[-1].value) if past_results else ""

        repair_text = (
            f"The previous query failed validation:\n"
            f"Query: {failed_query}\n"
            f"Errors: {', '.join(error_messages)}\n"
            f"Please generate a corrected query."
        )

        return CBlock(repair_text), new_ctx

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        """Select the best query when all attempts have failed.

        Picks the attempt with the fewest validation errors.

        Args:
            sampled_actions: All actions attempted without success.
            sampled_results: Generation results for those actions.
            sampled_val: Validation results for each attempt.

        Returns:
            Index of the attempt with the fewest validation errors.
        """
        error_counts = [
            sum(1 for _, result in validation if not bool(result))
            for validation in sampled_val
        ]
        return error_counts.index(min(error_counts))
