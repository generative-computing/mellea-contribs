"""Integration tests for sampling strategies with live Mellea session."""

import dspy
import pytest
from mellea.stdlib.sampling import MultiTurnStrategy, RejectionSamplingStrategy
from mellea_dspy import MelleaLM

pytestmark = pytest.mark.integration


class TestStrategyBasic:
    """Basic strategy tests with live session."""

    def test_forward_with_rejection_strategy(self, lm):
        """Test forward with RejectionSamplingStrategy."""
        strategy = RejectionSamplingStrategy(loop_budget=3)
        response = lm.forward(prompt="Say hello", strategy=strategy)

        assert response is not None
        assert hasattr(response, "choices")
        assert response.choices[0].message.content

    def test_forward_with_multi_turn_strategy(self, lm):
        """Test forward with MultiTurnStrategy."""
        strategy = MultiTurnStrategy(loop_budget=2)
        response = lm.forward(prompt="Generate a greeting", strategy=strategy)

        assert response is not None
        assert response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_aforward_with_strategy(self, lm):
        """Test async forward with strategy."""
        strategy = RejectionSamplingStrategy(loop_budget=3)
        response = await lm.aforward(prompt="Say hello", strategy=strategy)

        assert response is not None
        assert response.choices[0].message.content


class TestStrategyWithRequirements:
    """Test strategies combined with requirements."""

    def test_strategy_and_requirements_together(self, lm):
        """Test using both strategy and requirements."""
        strategy = RejectionSamplingStrategy(loop_budget=5)
        requirements = ["be concise", "use examples"]

        response = lm.forward(
            prompt="Explain Python", strategy=strategy, requirements=requirements
        )

        assert response is not None
        assert response.choices[0].message.content

    def test_rejection_sampling_with_requirements(self, lm):
        """Test rejection sampling strategy with requirements."""
        strategy = RejectionSamplingStrategy(loop_budget=10)
        requirements = [
            "must include code",
            "must be valid Python",
            "must include comments",
        ]

        response = lm.forward(
            prompt="Write a factorial function",
            strategy=strategy,
            requirements=requirements,
        )

        assert response is not None
        assert response.choices[0].message.content
        # Should contain some code-like content
        assert len(response.choices[0].message.content) > 10


class TestStrategyTypes:
    """Test different strategy types."""

    def test_rejection_strategy_with_retries(self, lm):
        """Test rejection sampling with different loop budgets."""
        for loop_budget in [1, 3, 5]:
            strategy = RejectionSamplingStrategy(loop_budget=loop_budget)
            response = lm.forward(
                prompt="Generate a creative slogan for AI", strategy=strategy
            )

            assert response is not None
            assert response.choices[0].message.content

    def test_multi_turn_strategy(self, lm):
        """Test multi-turn strategy."""
        strategy = MultiTurnStrategy(loop_budget=3)
        response = lm.forward(
            prompt="Answer: What are the key benefits of generative programming?",
            strategy=strategy,
        )

        assert response is not None
        assert response.choices[0].message.content

    def test_strategy_with_temperature(self, mellea_session):
        """Test strategy combined with temperature."""
        strategy = RejectionSamplingStrategy(loop_budget=3)

        for temp in [0.0, 0.5, 1.0]:
            lm = MelleaLM(
                mellea_session=mellea_session, model="mellea-test", temperature=temp
            )

            response = lm.forward(
                prompt="Write a creative story opening", strategy=strategy
            )

            assert response is not None
            assert response.choices[0].message.content


class TestStrategyWithDSPy:
    """Test strategies with DSPy modules."""

    def test_predict_with_strategy(self, mellea_session):
        """Test DSPy Predict with strategy."""
        strategy = RejectionSamplingStrategy(loop_budget=3)
        lm = MelleaLM(
            mellea_session=mellea_session, model="mellea-test", strategy=strategy
        )
        dspy.configure(lm=lm)

        predictor = dspy.Predict("question -> answer")
        response = predictor(question="What is Python?")

        assert hasattr(response, "answer")
        assert response.answer

    def test_chain_of_thought_with_strategy(self, mellea_session):
        """Test ChainOfThought with strategy."""
        strategy = MultiTurnStrategy(loop_budget=2)
        lm = MelleaLM(
            mellea_session=mellea_session, model="mellea-test", strategy=strategy
        )
        dspy.configure(lm=lm)

        cot = dspy.ChainOfThought("question -> answer")
        response = cot(question="Why is the sky blue?")

        assert hasattr(response, "answer")
        assert response.answer


class TestStrategyScenarios:
    """Test realistic strategy usage scenarios."""

    def test_code_generation_with_validation(self, lm):
        """Test code generation with validation strategy."""
        strategy = RejectionSamplingStrategy(loop_budget=5)
        requirements = [
            "must be valid Python",
            "must include docstring",
            "must handle edge cases",
        ]

        response = lm.forward(
            prompt="Write a function to reverse a string",
            strategy=strategy,
            requirements=requirements,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert len(content) > 0

    def test_creative_generation_with_retries(self, lm):
        """Test creative generation with retry strategy."""
        strategy = RejectionSamplingStrategy(loop_budget=3)

        response = lm.forward(
            prompt="Write a unique metaphor for programming", strategy=strategy
        )

        assert response is not None
        assert response.choices[0].message.content

    def test_factual_qa_with_strategy(self, lm):
        """Test factual QA with strategy."""
        strategy = RejectionSamplingStrategy(loop_budget=2)

        response = lm.forward(
            prompt="What is the capital of France?", strategy=strategy
        )

        assert response is not None
        assert response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_concurrent_with_strategies(self, lm):
        """Test concurrent requests with strategies."""
        import asyncio

        strategy1 = RejectionSamplingStrategy(loop_budget=2)
        strategy2 = MultiTurnStrategy(loop_budget=2)

        results = await asyncio.gather(
            lm.aforward(prompt="Say hello", strategy=strategy1),
            lm.aforward(prompt="Say goodbye", strategy=strategy2),
            lm.aforward(prompt="Say thanks", strategy=None),
        )

        assert len(results) == 3
        for response in results:
            assert response is not None
            assert response.choices[0].message.content


class TestStrategyEdgeCases:
    """Test edge cases for strategies."""

    def test_strategy_none_explicitly(self, lm):
        """Test with strategy explicitly set to None."""
        response = lm.forward(prompt="Say hello", strategy=None)

        assert response is not None
        assert response.choices[0].message.content

    def test_strategy_with_zero_retries(self, lm):
        """Test rejection strategy with loop_budget of 1 (minimum)."""
        strategy = RejectionSamplingStrategy(loop_budget=1)

        response = lm.forward(prompt="Say hello", strategy=strategy)

        assert response is not None
        assert response.choices[0].message.content

    def test_strategy_override_instance_strategy(self, mellea_session):
        """Test that forward strategy overrides instance strategy."""
        instance_strategy = RejectionSamplingStrategy(loop_budget=1)
        lm = MelleaLM(
            mellea_session=mellea_session,
            model="mellea-test",
            strategy=instance_strategy,
        )

        forward_strategy = MultiTurnStrategy(loop_budget=2)
        response = lm.forward(prompt="Say hello", strategy=forward_strategy)

        assert response is not None
        assert response.choices[0].message.content

    def test_strategy_with_requirements_and_temperature(self, mellea_session):
        """Test strategy combined with requirements and temperature."""
        strategy = RejectionSamplingStrategy(loop_budget=3)
        requirements = ["be creative", "use metaphors"]

        lm = MelleaLM(
            mellea_session=mellea_session, model="mellea-test", temperature=0.8
        )

        response = lm.forward(
            prompt="Describe programming", strategy=strategy, requirements=requirements
        )

        assert response is not None
        assert response.choices[0].message.content


class TestStrategyConfiguration:
    """Test strategy configuration options."""

    def test_rejection_strategy_loop_budget(self, lm):
        """Test rejection strategy with different loop_budget values."""
        for budget in [1, 5, 10]:
            strategy = RejectionSamplingStrategy(loop_budget=budget)
            response = lm.forward(prompt="Generate text", strategy=strategy)
            assert response is not None

    def test_multi_turn_strategy_loop_budget(self, lm):
        """Test multi-turn strategy with different loop_budget values."""
        for budget in [1, 2, 3]:
            strategy = MultiTurnStrategy(loop_budget=budget)
            response = lm.forward(prompt="Answer a question", strategy=strategy)
            assert response is not None

    def test_strategy_in_constructor_vs_forward(self, mellea_session):
        """Test strategy can be set in constructor or forward."""
        # Strategy in constructor
        strategy1 = RejectionSamplingStrategy(loop_budget=2)
        lm1 = MelleaLM(
            mellea_session=mellea_session, model="mellea-test", strategy=strategy1
        )
        response1 = lm1.forward(prompt="Test 1")
        assert response1 is not None

        # Strategy in forward
        lm2 = MelleaLM(mellea_session=mellea_session, model="mellea-test")
        strategy2 = MultiTurnStrategy(loop_budget=2)
        response2 = lm2.forward(prompt="Test 2", strategy=strategy2)
        assert response2 is not None
