"""Unit tests for verification module."""

from unittest.mock import MagicMock

from mellea_dspy.verification import (
    MelleaBestOfN,
    MelleaRefine,
    combine_rewards,
    requirement_to_reward,
)


class MockPrediction:
    """Mock prediction object for testing."""

    def __init__(self, answer="test answer", output=None, summary=None):
        """Initialize mock prediction."""
        self.answer = answer
        if output:
            self.output = output
        if summary:
            self.summary = summary


class TestRequirementToReward:
    """Test requirement_to_reward function."""

    def test_max_words_reward_under_limit(self):
        """Test max words reward when under limit."""
        reward_fn = requirement_to_reward("Must be under 50 words")
        pred = MockPrediction(answer="This is a short answer")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_max_words_reward_over_limit(self):
        """Test max words reward when over limit."""
        reward_fn = requirement_to_reward("Must be under 5 words")
        pred = MockPrediction(answer="This is a much longer answer than allowed")

        score = reward_fn(None, pred)
        assert 0.0 <= score < 1.0

    def test_min_words_reward_meets_requirement(self):
        """Test min words reward when requirement is met."""
        reward_fn = requirement_to_reward("Must be at least 5 words")
        pred = MockPrediction(answer="This is a sufficient length answer")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_min_words_reward_below_requirement(self):
        """Test min words reward when below requirement."""
        reward_fn = requirement_to_reward("Must be at least 10 words")
        pred = MockPrediction(answer="Too short")

        score = reward_fn(None, pred)
        assert 0.0 < score < 1.0

    def test_word_range_reward_in_range(self):
        """Test word range reward when in range."""
        reward_fn = requirement_to_reward("Must be between 5 and 10 words")
        pred = MockPrediction(answer="This answer has exactly seven words total")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_word_range_reward_below_range(self):
        """Test word range reward when below range."""
        reward_fn = requirement_to_reward("Must be between 10 and 20 words")
        pred = MockPrediction(answer="Too short")

        score = reward_fn(None, pred)
        assert 0.0 < score < 1.0

    def test_word_range_reward_above_range(self):
        """Test word range reward when above range."""
        reward_fn = requirement_to_reward("Must be between 5 and 10 words")
        pred = MockPrediction(
            answer="This is a much longer answer that exceeds the maximum word count"
        )

        score = reward_fn(None, pred)
        assert 0.0 <= score < 1.0

    def test_max_chars_reward_under_limit(self):
        """Test max characters reward when under limit."""
        reward_fn = requirement_to_reward("Must be under 100 characters")
        pred = MockPrediction(answer="Short")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_max_chars_reward_over_limit(self):
        """Test max characters reward when over limit."""
        reward_fn = requirement_to_reward("Must be under 10 characters")
        pred = MockPrediction(answer="This is way too long")

        score = reward_fn(None, pred)
        assert 0.0 <= score < 1.0

    def test_content_check_reward_present(self):
        """Test content check when term is present."""
        reward_fn = requirement_to_reward("Must mention AI")
        pred = MockPrediction(answer="AI is transforming technology")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_content_check_reward_absent(self):
        """Test content check when term is absent."""
        reward_fn = requirement_to_reward("Must mention AI")
        pred = MockPrediction(answer="Technology is advancing")

        score = reward_fn(None, pred)
        assert score == 0.0

    def test_content_check_case_insensitive(self):
        """Test content check is case insensitive."""
        reward_fn = requirement_to_reward("Must mention python")
        pred = MockPrediction(answer="Python is a programming language")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_negative_content_reward_absent(self):
        """Test negative content when term is absent."""
        reward_fn = requirement_to_reward("Must not mention politics")
        pred = MockPrediction(answer="This is about technology")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_negative_content_reward_present(self):
        """Test negative content when term is present."""
        reward_fn = requirement_to_reward("Must not mention politics")
        pred = MockPrediction(answer="Politics is controversial")

        score = reward_fn(None, pred)
        assert score == 0.0

    def test_bullet_format_reward_has_bullets(self):
        """Test bullet format when bullets are present."""
        reward_fn = requirement_to_reward("Must be in bullet points")
        pred = MockPrediction(answer="- Item 1\n- Item 2\n- Item 3")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_bullet_format_reward_no_bullets(self):
        """Test bullet format when no bullets."""
        reward_fn = requirement_to_reward("Must be in bullet points")
        pred = MockPrediction(answer="Plain text without bullets")

        score = reward_fn(None, pred)
        assert score == 0.0

    def test_numbered_format_reward_has_numbers(self):
        """Test numbered format when numbers are present."""
        reward_fn = requirement_to_reward("Must be numbered list")
        pred = MockPrediction(answer="1. First item\n2. Second item\n3. Third item")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_numbered_format_reward_no_numbers(self):
        """Test numbered format when no numbers."""
        reward_fn = requirement_to_reward("Must be numbered list")
        pred = MockPrediction(answer="Plain text without numbers")

        score = reward_fn(None, pred)
        assert score == 0.0

    def test_json_validation_reward_valid(self):
        """Test JSON validation with valid JSON."""
        reward_fn = requirement_to_reward("Must be valid JSON")
        pred = MockPrediction(answer='{"key": "value", "number": 42}')

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_json_validation_reward_invalid(self):
        """Test JSON validation with invalid JSON."""
        reward_fn = requirement_to_reward("Must be valid JSON")
        pred = MockPrediction(answer="Not valid JSON")

        score = reward_fn(None, pred)
        assert score == 0.0

    def test_conciseness_reward_concise(self):
        """Test conciseness reward with concise text."""
        reward_fn = requirement_to_reward("Must be concise")
        pred = MockPrediction(answer="Short and concise answer")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_conciseness_reward_verbose(self):
        """Test conciseness reward with verbose text."""
        reward_fn = requirement_to_reward("Must be concise")
        long_text = " ".join(["word"] * 300)
        pred = MockPrediction(answer=long_text)

        score = reward_fn(None, pred)
        assert score < 1.0

    def test_detail_reward_detailed(self):
        """Test detail reward with detailed text."""
        reward_fn = requirement_to_reward("Must be detailed")
        long_text = " ".join(["word"] * 150)
        pred = MockPrediction(answer=long_text)

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_detail_reward_brief(self):
        """Test detail reward with brief text."""
        reward_fn = requirement_to_reward("Must be detailed")
        pred = MockPrediction(answer="Too brief")

        score = reward_fn(None, pred)
        assert score < 1.0

    def test_professional_reward_professional(self):
        """Test professional reward with professional text."""
        reward_fn = requirement_to_reward("Must be professional")
        pred = MockPrediction(answer="This is a professional response")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_professional_reward_casual(self):
        """Test professional reward with casual text."""
        reward_fn = requirement_to_reward("Must be professional")
        pred = MockPrediction(answer="Hey, gonna send you the info, yeah?")

        score = reward_fn(None, pred)
        assert score < 1.0

    def test_callable_requirement_bool_return(self):
        """Test callable requirement with bool return."""

        def custom_check(args, pred):
            return len(pred.answer.split()) < 10

        reward_fn = requirement_to_reward(custom_check)
        pred = MockPrediction(answer="Short answer")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_callable_requirement_float_return(self):
        """Test callable requirement with float return."""

        def custom_check(args, pred):
            return 0.75

        reward_fn = requirement_to_reward(custom_check)
        pred = MockPrediction(answer="Test")

        score = reward_fn(None, pred)
        assert score == 0.75

    def test_callable_requirement_pred_only(self):
        """Test callable requirement with pred-only signature."""

        def custom_check(pred):
            return len(pred.answer) < 50

        reward_fn = requirement_to_reward(custom_check)
        pred = MockPrediction(answer="Short")

        score = reward_fn(None, pred)
        assert score == 1.0

    def test_extract_text_from_different_fields(self):
        """Test text extraction from different prediction fields."""
        # Test with output field
        reward_fn = requirement_to_reward("Must mention test")
        pred = MockPrediction(answer=None, output="This is a test")

        score = reward_fn(None, pred)
        assert score == 1.0

        # Test with summary field
        pred2 = MockPrediction(answer=None, output=None, summary="Test summary")
        score2 = reward_fn(None, pred2)
        assert score2 == 1.0


class TestCombineRewards:
    """Test combine_rewards function."""

    def test_combine_average_strategy(self):
        """Test combining rewards with average strategy."""

        def reward1(args, pred):
            return 1.0

        def reward2(args, pred):
            return 0.5

        combined = combine_rewards([reward1, reward2], strategy="average")
        score = combined(None, None)

        assert score == 0.75

    def test_combine_min_strategy(self):
        """Test combining rewards with min strategy."""

        def reward1(args, pred):
            return 1.0

        def reward2(args, pred):
            return 0.3

        combined = combine_rewards([reward1, reward2], strategy="min")
        score = combined(None, None)

        assert score == 0.3

    def test_combine_product_strategy(self):
        """Test combining rewards with product strategy."""

        def reward1(args, pred):
            return 0.8

        def reward2(args, pred):
            return 0.5

        combined = combine_rewards([reward1, reward2], strategy="product")
        score = combined(None, None)

        assert score == 0.4

    def test_combine_empty_list(self):
        """Test combining empty reward list."""
        combined = combine_rewards([], strategy="average")
        score = combined(None, None)

        assert score == 1.0


class TestMelleaBestOfN:
    """Test MelleaBestOfN wrapper."""

    def test_initialization_with_requirements(self):
        """Test initialization with requirements."""
        mock_module = MagicMock()

        best_of_n = MelleaBestOfN(
            module=mock_module,
            N=5,
            requirements=["Must be under 50 words", "Must mention AI"],
            threshold=0.8,
        )

        assert best_of_n.N == 5
        assert len(best_of_n.requirements) == 2
        assert best_of_n.threshold == 0.8
        assert best_of_n.reward_fn is not None

    def test_initialization_without_requirements(self):
        """Test initialization without requirements."""
        mock_module = MagicMock()

        best_of_n = MelleaBestOfN(module=mock_module, N=3)

        assert best_of_n.N == 3
        assert len(best_of_n.requirements) == 0
        assert best_of_n.reward_fn is not None

    def test_initialization_with_callable_requirements(self):
        """Test initialization with callable requirements."""
        mock_module = MagicMock()

        def custom_check(args, pred):
            return True

        best_of_n = MelleaBestOfN(
            module=mock_module, N=3, requirements=[custom_check, "Must be concise"]
        )

        assert len(best_of_n.requirements) == 2

    def test_reward_function_creation(self):
        """Test reward function is created correctly."""
        mock_module = MagicMock()

        best_of_n = MelleaBestOfN(
            module=mock_module, N=3, requirements=["Must be under 50 words"]
        )

        # Test the reward function works
        pred = MockPrediction(answer="Short answer")
        score = best_of_n.reward_fn(None, pred)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_combination_strategies(self):
        """Test different combination strategies."""
        mock_module = MagicMock()

        # Test average
        best_of_n_avg = MelleaBestOfN(
            module=mock_module,
            N=3,
            requirements=["Must be under 50 words", "Must mention test"],
            combination="average",
        )
        assert best_of_n_avg.combination == "average"

        # Test min
        best_of_n_min = MelleaBestOfN(
            module=mock_module,
            N=3,
            requirements=["Must be under 50 words"],
            combination="min",
        )
        assert best_of_n_min.combination == "min"


class SimpleTestModule:
    """Simple test module for Refine tests.

    This is a real module (not a mock) so dspy.Refine can inspect its source.
    """

    def __init__(self):
        """Initialize the test module."""
        pass

    def __call__(self, **kwargs):
        """Simple forward method."""
        return MockPrediction(answer="test")

    def forward(self, **kwargs):
        """Forward method."""
        return MockPrediction(answer="test")


class TestMelleaRefine:
    """Test MelleaRefine wrapper."""

    def test_initialization_with_requirements(self):
        """Test initialization with requirements."""
        # Use a real module instead of mock for dspy.Refine
        test_module = SimpleTestModule()

        refine = MelleaRefine(
            module=test_module,
            N=3,
            requirements=["Must be under 50 words", "Must be professional"],
            threshold=0.9,
        )

        assert refine.N == 3
        assert len(refine.requirements) == 2
        assert refine.threshold == 0.9
        assert refine.reward_fn is not None

    def test_initialization_without_requirements(self):
        """Test initialization without requirements."""
        # Use a real module instead of mock for dspy.Refine
        test_module = SimpleTestModule()

        refine = MelleaRefine(module=test_module, N=3)

        assert refine.N == 3
        assert len(refine.requirements) == 0
        assert refine.reward_fn is not None

    def test_reward_function_creation(self):
        """Test reward function is created correctly."""
        # Use a real module instead of mock for dspy.Refine
        test_module = SimpleTestModule()

        refine = MelleaRefine(
            module=test_module, N=3, requirements=["Must be professional"]
        )

        # Test the reward function works
        pred = MockPrediction(answer="Professional response")
        score = refine.reward_fn(None, pred)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_combination_strategies(self):
        """Test different combination strategies."""
        # Use a real module instead of mock for dspy.Refine
        test_module = SimpleTestModule()

        # Test average
        refine_avg = MelleaRefine(
            module=test_module,
            N=3,
            requirements=["Must be under 50 words", "Must be detailed"],
            combination="average",
        )
        assert refine_avg.combination == "average"

        # Test product
        test_module2 = SimpleTestModule()
        refine_prod = MelleaRefine(
            module=test_module2,
            N=3,
            requirements=["Must be professional"],
            combination="product",
        )
        assert refine_prod.combination == "product"
