"""Verification wrappers for DSPy with Mellea requirements.

This module provides wrappers for dspy.BestOfN and dspy.Refine that automatically
convert Mellea requirements into DSPy reward functions for runtime verification
and optimization.
"""

import json
import re
from collections.abc import Callable
from typing import Any

import dspy


def requirement_to_reward(requirement: str | Callable) -> Callable:
    """Convert Mellea requirement to DSPy reward function.

    Supports multiple requirement patterns and returns a reward function
    that scores predictions from 0.0 (fails) to 1.0 (perfect).

    Args:
        requirement: String requirement pattern or callable validator

    Returns:
        Reward function with signature (args, pred) -> float

    Examples:
        >>> reward_fn = requirement_to_reward("Must be under 50 words")
        >>> reward_fn(args, pred)  # Returns 1.0 if <= 50 words, else gradual penalty

        >>> reward_fn = requirement_to_reward("Must mention AI")
        >>> reward_fn(args, pred)  # Returns 1.0 if "AI" in text, else 0.0
    """
    # If already callable, wrap it to ensure correct signature
    if callable(requirement):
        return _wrap_callable_requirement(requirement)

    # Parse string requirement
    req_lower = requirement.lower().strip()

    # Length constraints
    if "under" in req_lower and "word" in req_lower:
        return _create_max_words_reward(requirement)
    elif "at least" in req_lower and "word" in req_lower:
        return _create_min_words_reward(requirement)
    elif "between" in req_lower and "word" in req_lower:
        return _create_word_range_reward(requirement)
    elif "under" in req_lower and "character" in req_lower:
        return _create_max_chars_reward(requirement)

    # Content requirements
    elif "must mention" in req_lower or "must include" in req_lower:
        return _create_content_check_reward(requirement)
    elif "must not mention" in req_lower:
        return _create_negative_content_reward(requirement)

    # Format requirements
    elif "bullet point" in req_lower:
        return _create_bullet_format_reward()
    elif "numbered list" in req_lower:
        return _create_numbered_format_reward()
    elif "valid json" in req_lower:
        return _create_json_validation_reward()

    # Quality requirements
    elif "must be concise" in req_lower:
        return _create_conciseness_reward()
    elif "must be detailed" in req_lower:
        return _create_detail_reward()
    elif "must be professional" in req_lower:
        return _create_professional_reward()

    # Default: exact string match (case-insensitive)
    else:
        return _create_default_reward(requirement)


def combine_rewards(reward_fns: list[Callable], strategy: str = "average") -> Callable:
    """Combine multiple reward functions.

    Args:
        reward_fns: List of reward functions
        strategy: "average", "min", "weighted", or "product"

    Returns:
        Combined reward function with signature (args, pred) -> float
    """

    def combined_reward(args, pred):
        if not reward_fns:
            return 1.0

        scores = [fn(args, pred) for fn in reward_fns]

        if strategy == "average":
            return sum(scores) / len(scores)
        elif strategy == "min":
            return min(scores)
        elif strategy == "product":
            result = 1.0
            for score in scores:
                result *= score
            return result
        else:
            return sum(scores) / len(scores)

    return combined_reward


# Helper functions for creating specific reward functions


def _wrap_callable_requirement(fn: Callable) -> Callable:
    """Wrap callable to ensure DSPy reward signature."""

    def reward(args, pred):
        try:
            # Try DSPy signature first (args, pred)
            result = fn(args, pred)
        except TypeError:
            # Fallback: try with just pred
            try:
                result = fn(pred)
            except Exception:
                return 0.0

        # Ensure return value is float 0.0-1.0
        if isinstance(result, bool):
            return 1.0 if result else 0.0
        elif isinstance(result, (int, float)):
            return max(0.0, min(1.0, float(result)))
        else:
            return 0.0

    return reward


def _create_max_words_reward(requirement: str) -> Callable:
    """Create reward for 'Must be under X words'."""
    # Extract number from requirement
    match = re.search(r"(\d+)\s+word", requirement.lower())
    max_words = int(match.group(1)) if match else 100

    def reward(args, pred):
        text = _extract_text_from_pred(pred)
        word_count = len(text.split())

        if word_count <= max_words:
            return 1.0
        else:
            # Gradual penalty for exceeding limit
            excess = word_count - max_words
            penalty = min(1.0, excess / max_words)
            return max(0.0, 1.0 - penalty)

    return reward


def _create_min_words_reward(requirement: str) -> Callable:
    """Create reward for 'Must be at least X words'."""
    match = re.search(r"(\d+)\s+word", requirement.lower())
    min_words = int(match.group(1)) if match else 10

    def reward(args, pred):
        text = _extract_text_from_pred(pred)
        word_count = len(text.split())

        if word_count >= min_words:
            return 1.0
        else:
            # Gradual score based on progress
            return word_count / min_words if min_words > 0 else 0.0

    return reward


def _create_word_range_reward(requirement: str) -> Callable:
    """Create reward for 'Must be between X and Y words'."""
    matches = re.findall(r"(\d+)", requirement)
    if len(matches) >= 2:
        min_words, max_words = int(matches[0]), int(matches[1])
    else:
        min_words, max_words = 10, 100

    def reward(args, pred):
        text = _extract_text_from_pred(pred)
        word_count = len(text.split())

        if min_words <= word_count <= max_words:
            return 1.0
        elif word_count < min_words:
            return word_count / min_words if min_words > 0 else 0.0
        else:
            excess = word_count - max_words
            penalty = min(1.0, excess / max_words) if max_words > 0 else 1.0
            return max(0.0, 1.0 - penalty)

    return reward


def _create_max_chars_reward(requirement: str) -> Callable:
    """Create reward for 'Must be under X characters'."""
    match = re.search(r"(\d+)\s+character", requirement.lower())
    max_chars = int(match.group(1)) if match else 500

    def reward(args, pred):
        text = _extract_text_from_pred(pred)
        char_count = len(text)

        if char_count <= max_chars:
            return 1.0
        else:
            excess = char_count - max_chars
            penalty = min(1.0, excess / max_chars) if max_chars > 0 else 1.0
            return max(0.0, 1.0 - penalty)

    return reward


def _create_content_check_reward(requirement: str) -> Callable:
    """Create reward for 'Must mention X' or 'Must include X'."""
    # Extract the term to check for
    if "must mention" in requirement.lower():
        term = requirement.lower().split("must mention")[-1].strip()
    else:
        term = requirement.lower().split("must include")[-1].strip()

    # Remove quotes if present
    term = term.strip("\"'")

    def reward(args, pred):
        text = _extract_text_from_pred(pred).lower()
        return 1.0 if term in text else 0.0

    return reward


def _create_negative_content_reward(requirement: str) -> Callable:
    """Create reward for 'Must not mention X'."""
    term = requirement.lower().split("must not mention")[-1].strip()
    term = term.strip("\"'")

    def reward(args, pred):
        text = _extract_text_from_pred(pred).lower()
        return 0.0 if term in text else 1.0

    return reward


def _create_bullet_format_reward() -> Callable:
    """Create reward for bullet point format."""

    def reward(args, pred):
        text = _extract_text_from_pred(pred)
        lines = text.split("\n")

        # Check for bullet markers
        bullet_markers = ["-", "*", "•", "◦", "▪"]
        bullet_lines = sum(
            1
            for line in lines
            if any(line.strip().startswith(m) for m in bullet_markers)
        )

        if bullet_lines >= 2:
            return 1.0
        elif bullet_lines == 1:
            return 0.5
        else:
            return 0.0

    return reward


def _create_numbered_format_reward() -> Callable:
    """Create reward for numbered list format."""

    def reward(args, pred):
        text = _extract_text_from_pred(pred)
        lines = text.split("\n")

        # Check for numbered items (1., 2., etc.)
        numbered_pattern = re.compile(r"^\s*\d+[\.)]\s+")
        numbered_lines = sum(1 for line in lines if numbered_pattern.match(line))

        if numbered_lines >= 2:
            return 1.0
        elif numbered_lines == 1:
            return 0.5
        else:
            return 0.0

    return reward


def _create_json_validation_reward() -> Callable:
    """Create reward for valid JSON format."""

    def reward(args, pred):
        text = _extract_text_from_pred(pred)
        try:
            json.loads(text)
            return 1.0
        except Exception:
            return 0.0

    return reward


def _create_conciseness_reward() -> Callable:
    """Create reward for concise output."""

    def reward(args, pred):
        text = _extract_text_from_pred(pred)
        word_count = len(text.split())

        # Score based on brevity (prefer 10-50 words)
        if word_count <= 50:
            return 1.0
        elif word_count <= 100:
            return 0.7
        elif word_count <= 200:
            return 0.4
        else:
            return 0.2

    return reward


def _create_detail_reward() -> Callable:
    """Create reward for detailed output."""

    def reward(args, pred):
        text = _extract_text_from_pred(pred)
        word_count = len(text.split())

        # Score based on length (prefer 100+ words)
        if word_count >= 100:
            return 1.0
        elif word_count >= 50:
            return 0.7
        elif word_count >= 25:
            return 0.4
        else:
            return 0.2

    return reward


def _create_professional_reward() -> Callable:
    """Create reward for professional tone."""
    casual_words = ["hey", "gonna", "wanna", "yeah", "nah", "lol", "omg"]

    def reward(args, pred):
        text = _extract_text_from_pred(pred).lower()

        # Check for casual language
        casual_count = sum(1 for word in casual_words if word in text)

        if casual_count == 0:
            return 1.0
        elif casual_count <= 2:
            return 0.5
        else:
            return 0.0

    return reward


def _create_default_reward(requirement: str) -> Callable:
    """Create default reward for unrecognized patterns."""

    def reward(args, pred):
        text = _extract_text_from_pred(pred).lower()
        req_lower = requirement.lower()

        # Simple substring match
        return 1.0 if req_lower in text else 0.0

    return reward


def _extract_text_from_pred(pred: Any) -> str:
    """Extract text from prediction object.

    Handles different DSPy prediction formats:
    - pred.answer
    - pred.output
    - pred.summary
    - pred.result
    - str(pred)
    """
    # Try common field names
    for field in ["answer", "output", "summary", "result", "text", "content"]:
        if hasattr(pred, field):
            value = getattr(pred, field)
            if isinstance(value, str):
                return value

    # Fallback to string representation
    return str(pred)


# Public API for creating reward functions


def create_reward_fn(
    requirements: list[str | Callable], strategy: str = "average"
) -> Callable:
    """Create a DSPy reward function from Mellea requirements.

    This is a lower-level function for advanced users who want to use
    dspy.BestOfN or dspy.Refine directly. Most users should use
    MelleaBestOfN or MelleaRefine instead.

    Args:
        requirements: List of requirement strings or callables
        strategy: Combination strategy ("average", "min", "product")

    Returns:
        Reward function compatible with dspy.BestOfN and dspy.Refine
        with signature (args, pred) -> float (0.0-1.0)

    Example:
        ```python
        import dspy
        from mellea_dspy import create_reward_fn

        # Create reward function
        reward_fn = create_reward_fn(
            requirements=["Must be under 50 words", "Must mention AI"],
            strategy="average"
        )

        # Use with native DSPy
        qa = dspy.ChainOfThought("question -> answer")
        best_of_5 = dspy.BestOfN(
            module=qa,
            N=5,
            reward_fn=reward_fn,
            threshold=0.8
        )

        result = best_of_5(question="What is Python?")
        ```
    """
    if not requirements:
        # No requirements, always return 1.0
        return lambda args, pred: 1.0

    # Convert each requirement to a reward function
    reward_fns = [requirement_to_reward(req) for req in requirements]

    # Combine them using the specified strategy
    return combine_rewards(reward_fns, strategy=strategy)


# Wrapper classes


class MelleaBestOfN:
    """Generate N candidates and select best using Mellea requirements.

    Wraps dspy.BestOfN with automatic requirement-to-reward conversion.

    Args:
        module: DSPy module to wrap
        N: Number of candidates to generate
        requirements: List of requirement strings or callables
        threshold: Minimum reward score to accept (0.0-1.0)
        combination: How to combine multiple requirements ("average", "min", "product")
        **kwargs: Additional arguments for dspy.BestOfN

    Example:
        ```python
        import dspy
        from mellea_dspy import MelleaBestOfN

        qa = dspy.ChainOfThought("question -> answer")

        best_of_5 = MelleaBestOfN(
            module=qa,
            N=5,
            requirements=["Must be one word", "Must be a proper noun"],
            threshold=0.8
        )

        result = best_of_5(question="What is the capital of Belgium?")
        print(result.answer)  # "Brussels"
        ```
    """

    def __init__(
        self,
        module: dspy.Module,
        N: int = 3,
        requirements: list[str | Callable] | None = None,
        threshold: float = 0.8,
        combination: str = "average",
        **kwargs: Any,
    ):
        """Initialize MelleaBestOfN wrapper.

        Args:
            module: DSPy module to wrap
            N: Number of candidates to generate
            requirements: List of requirement strings or callables
            threshold: Minimum reward score to accept (0.0-1.0)
            combination: How to combine multiple requirements
            **kwargs: Additional arguments for dspy.BestOfN
        """
        self.module = module
        self.N = N
        self.requirements = requirements or []
        self.threshold = threshold
        self.combination = combination

        # Convert requirements to reward function using public API
        self.reward_fn = create_reward_fn(
            requirements=self.requirements, strategy=self.combination
        )

        # Create underlying dspy.BestOfN
        self.best_of_n = dspy.BestOfN(
            module=module, N=N, reward_fn=self.reward_fn, threshold=threshold, **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward call to wrapped BestOfN."""
        return self.best_of_n(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward method for DSPy compatibility."""
        return self.best_of_n.forward(*args, **kwargs)


class MelleaRefine:
    """Iteratively refine outputs using Mellea requirements.

    Wraps dspy.Refine with automatic requirement-to-reward conversion
    and feedback generation.

    Args:
        module: DSPy module to wrap
        N: Maximum refinement iterations
        requirements: List of requirement strings or callables
        threshold: Minimum reward score to accept (0.0-1.0)
        combination: How to combine multiple requirements
        **kwargs: Additional arguments for dspy.Refine

    Example:
        ```python
        import dspy
        from mellea_dspy import MelleaRefine

        summarizer = dspy.Predict("text -> summary")

        refiner = MelleaRefine(
            module=summarizer,
            N=3,
            requirements=["Must be under 50 words", "Must be professional"],
            threshold=0.9
        )

        result = refiner(text="Long article text...")
        print(result.summary)
        ```
    """

    def __init__(
        self,
        module: dspy.Module,
        N: int = 3,
        requirements: list[str | Callable] | None = None,
        threshold: float = 0.9,
        combination: str = "average",
        **kwargs: Any,
    ):
        """Initialize MelleaRefine wrapper.

        Args:
            module: DSPy module to wrap
            N: Maximum refinement iterations
            requirements: List of requirement strings or callables
            threshold: Minimum reward score to accept (0.0-1.0)
            combination: How to combine multiple requirements
            **kwargs: Additional arguments for dspy.Refine
        """
        self.module = module
        self.N = N
        self.requirements = requirements or []
        self.threshold = threshold
        self.combination = combination

        # Convert requirements to reward function using public API
        self.reward_fn = create_reward_fn(
            requirements=self.requirements, strategy=self.combination
        )

        # Create underlying dspy.Refine
        self.refine = dspy.Refine(
            module=module, N=N, reward_fn=self.reward_fn, threshold=threshold, **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward call to wrapped Refine."""
        return self.refine(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward method for DSPy compatibility."""
        return self.refine.forward(*args, **kwargs)


__all__ = ["MelleaBestOfN", "MelleaRefine", "combine_rewards", "requirement_to_reward"]
