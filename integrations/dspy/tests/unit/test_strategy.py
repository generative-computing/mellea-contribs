"""Tests for strategy parameter handling in MelleaLM."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from mellea_dspy import MelleaLM


@pytest.fixture
def mock_session():
    """Create a mock Mellea session."""
    session = Mock()
    session.chat = Mock(return_value=Mock(content="Test response"))
    session.achat = AsyncMock(return_value=Mock(content="Async response"))
    session.instruct = Mock(return_value=Mock(content="Instruct response"))
    session.ainstruct = AsyncMock(return_value=Mock(content="Async instruct response"))
    return session


@pytest.fixture
def lm(mock_session):
    """Create a MelleaLM instance with mock session."""
    return MelleaLM(mellea_session=mock_session, model="mellea-test")


@pytest.fixture
def mock_strategy():
    """Create a mock sampling strategy."""
    strategy = Mock()
    strategy.name = "test_strategy"
    return strategy


class TestStrategyConstruction:
    """Tests for strategy in LM construction."""

    def test_strategy_stored_on_construction(self, mock_session, mock_strategy):
        """Test strategy is stored when passed to constructor."""
        lm = MelleaLM(mellea_session=mock_session, strategy=mock_strategy)

        assert lm._strategy == mock_strategy

    def test_strategy_none_by_default(self, mock_session):
        """Test strategy defaults to None."""
        lm = MelleaLM(mellea_session=mock_session)

        assert lm._strategy is None

    def test_strategy_with_requirements(self, mock_session, mock_strategy):
        """Test strategy can be combined with requirements."""
        requirements = ["be concise"]
        lm = MelleaLM(
            mellea_session=mock_session,
            requirements=requirements,
            strategy=mock_strategy,
        )

        assert lm._strategy == mock_strategy
        assert lm._requirements == requirements


class TestStrategyInForward:
    """Tests for strategy parameter in forward() method."""

    def test_forward_with_strategy_kwarg(self, lm, mock_strategy):
        """Test forward accepts strategy as kwarg."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            lm.forward(prompt="Test", strategy=mock_strategy)

            # Verify strategy was passed to generation
            call_kwargs = mock_gen.call_args[1]
            assert "strategy" in call_kwargs
            assert call_kwargs["strategy"] == mock_strategy

    def test_forward_strategy_override_instance_strategy(self, mock_session):
        """Test forward strategy overrides instance strategy."""
        instance_strategy = Mock(name="instance_strategy")
        lm = MelleaLM(mellea_session=mock_session, strategy=instance_strategy)

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            forward_strategy = Mock(name="forward_strategy")
            
            lm.forward(prompt="Test", strategy=forward_strategy)

            # Forward strategy should take precedence
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == forward_strategy

    def test_forward_with_strategy_and_requirements(self, lm, mock_strategy):
        """Test forward with both strategy and requirements."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = ["be concise", "use examples"]
            
            lm.forward(prompt="Test", strategy=mock_strategy, requirements=requirements)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == mock_strategy
            assert call_kwargs["requirements"] == requirements

    def test_forward_strategy_none_explicitly(self, lm):
        """Test forward with strategy explicitly set to None."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            lm.forward(prompt="Test", strategy=None)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] is None


class TestStrategyInAforward:
    """Tests for strategy parameter in aforward() async method."""

    @pytest.mark.asyncio
    async def test_aforward_with_strategy_kwarg(self, lm, mock_strategy):
        """Test aforward accepts strategy as kwarg."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            await lm.aforward(prompt="Test", strategy=mock_strategy)

            call_kwargs = mock_gen.call_args[1]
            assert "strategy" in call_kwargs
            assert call_kwargs["strategy"] == mock_strategy

    @pytest.mark.asyncio
    async def test_aforward_strategy_override_instance_strategy(self, mock_session):
        """Test aforward strategy overrides instance strategy."""
        instance_strategy = Mock(name="instance_strategy")
        lm = MelleaLM(mellea_session=mock_session, strategy=instance_strategy)

        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            forward_strategy = Mock(name="forward_strategy")
            
            await lm.aforward(prompt="Test", strategy=forward_strategy)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == forward_strategy

    @pytest.mark.asyncio
    async def test_aforward_with_strategy_and_requirements(self, lm, mock_strategy):
        """Test aforward with both strategy and requirements."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = ["be concise"]
            
            await lm.aforward(
                prompt="Test", strategy=mock_strategy, requirements=requirements
            )

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == mock_strategy
            assert call_kwargs["requirements"] == requirements


class TestStrategyWithMessages:
    """Tests for strategy with message-based input."""

    def test_forward_messages_with_strategy(self, lm, mock_strategy):
        """Test forward with messages and strategy."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            messages = [{"role": "user", "content": "Hello"}]
            
            lm.forward(messages=messages, strategy=mock_strategy)

            call_kwargs = mock_gen.call_args[1]
            assert "strategy" in call_kwargs
            assert call_kwargs["strategy"] == mock_strategy

    @pytest.mark.asyncio
    async def test_aforward_messages_with_strategy(self, lm, mock_strategy):
        """Test aforward with messages and strategy."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            messages = [{"role": "user", "content": "Hello"}]
            
            await lm.aforward(messages=messages, strategy=mock_strategy)

            call_kwargs = mock_gen.call_args[1]
            assert "strategy" in call_kwargs
            assert call_kwargs["strategy"] == mock_strategy


class TestStrategyTypes:
    """Tests for different types of strategies."""

    def test_strategy_as_object(self, lm):
        """Test strategy as an object with attributes."""
        strategy = Mock()
        strategy.name = "rejection_sampling"
        strategy.max_retries = 5
        strategy.temperature = 0.7

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            lm.forward(prompt="Test", strategy=strategy)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == strategy
            assert call_kwargs["strategy"].name == "rejection_sampling"

    def test_strategy_as_callable(self, lm):
        """Test strategy as a callable."""
        def custom_strategy():
            return "custom_strategy_result"

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            lm.forward(prompt="Test", strategy=custom_strategy)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == custom_strategy
            assert callable(call_kwargs["strategy"])

    def test_strategy_as_dict(self, lm):
        """Test strategy as a dictionary configuration."""
        strategy = {
            "type": "best_of_n",
            "n": 3,
            "temperature": 0.8,
        }

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            lm.forward(prompt="Test", strategy=strategy)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == strategy
            assert call_kwargs["strategy"]["type"] == "best_of_n"


class TestStrategyScenarios:
    """Tests for realistic strategy usage scenarios."""

    def test_rejection_sampling_scenario(self, lm):
        """Test rejection sampling strategy scenario."""
        rejection_strategy = Mock()
        rejection_strategy.type = "rejection"
        rejection_strategy.max_retries = 10
        rejection_strategy.validator = Mock()

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = ["must include code", "must be valid Python"]
            
            lm.forward(
                prompt="Generate a factorial function",
                strategy=rejection_strategy,
                requirements=requirements,
            )

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == rejection_strategy
            assert call_kwargs["requirements"] == requirements

    def test_best_of_n_scenario(self, lm):
        """Test best-of-n sampling strategy scenario."""
        best_of_n_strategy = Mock()
        best_of_n_strategy.type = "best_of_n"
        best_of_n_strategy.n = 5
        best_of_n_strategy.scorer = Mock()

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            lm.forward(
                prompt="Generate a creative slogan",
                strategy=best_of_n_strategy,
            )

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == best_of_n_strategy
            assert call_kwargs["strategy"].n == 5

    def test_ensemble_scenario(self, lm):
        """Test ensemble strategy scenario."""
        ensemble_strategy = Mock()
        ensemble_strategy.type = "ensemble"
        ensemble_strategy.models = ["model1", "model2", "model3"]
        ensemble_strategy.aggregation = "voting"

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            lm.forward(
                prompt="Classify this text",
                strategy=ensemble_strategy,
            )

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == ensemble_strategy
            assert len(call_kwargs["strategy"].models) == 3

    @pytest.mark.asyncio
    async def test_adaptive_sampling_scenario(self, lm):
        """Test adaptive sampling strategy scenario."""
        adaptive_strategy = Mock()
        adaptive_strategy.type = "adaptive"
        adaptive_strategy.confidence_threshold = 0.8
        adaptive_strategy.fallback_strategy = Mock()

        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            await lm.aforward(
                prompt="Answer this question",
                strategy=adaptive_strategy,
            )

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["strategy"] == adaptive_strategy
            assert call_kwargs["strategy"].confidence_threshold == 0.8

# Made with Bob
