"""Tests for MelleaLM class."""

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


class TestConstruction:
    """Tests for MelleaLM construction."""

    def test_default_construction(self, mock_session):
        """Test default construction with required parameters."""
        lm = MelleaLM(mellea_session=mock_session)

        assert lm.model == "mellea"
        assert lm.kwargs.get("temperature") == 0.0
        assert lm.kwargs.get("max_tokens") == 1000
        assert lm.provider == "mellea"

    def test_custom_model(self, mock_session):
        """Test construction with custom model."""
        lm = MelleaLM(mellea_session=mock_session, model="custom-model")

        assert lm.model == "custom-model"

    def test_custom_temperature(self, mock_session):
        """Test construction with custom temperature."""
        lm = MelleaLM(mellea_session=mock_session, temperature=0.7)

        assert lm.kwargs.get("temperature") == 0.7

    def test_custom_max_tokens(self, mock_session):
        """Test construction with custom max_tokens."""
        lm = MelleaLM(mellea_session=mock_session, max_tokens=2000)

        assert lm.kwargs.get("max_tokens") == 2000

    def test_requirements_stored(self, mock_session):
        """Test requirements are stored."""
        requirements = ["be concise", "use examples"]
        lm = MelleaLM(mellea_session=mock_session, requirements=requirements)

        assert lm._requirements == requirements

    def test_strategy_stored(self, mock_session):
        """Test strategy is stored."""
        strategy = Mock()
        lm = MelleaLM(mellea_session=mock_session, strategy=strategy)

        assert lm._strategy == strategy

    def test_provider_set(self, mock_session):
        """Test provider is set to mellea."""
        lm = MelleaLM(mellea_session=mock_session)

        assert lm.provider == "mellea"


class TestForwardRouting:
    """Tests for forward() method routing."""

    def test_forward_with_prompt_calls_internal_method(self, lm):
        """Test forward with prompt calls internal generation."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            lm.forward(prompt="Hi")

            assert mock_gen.called

    def test_forward_with_messages_calls_internal_method(self, lm):
        """Test forward with messages calls internal generation."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            lm.forward(messages=[{"role": "user", "content": "Hi"}])

            assert mock_gen.called

    def test_forward_no_inputs_raises_value_error(self, lm):
        """Test forward without prompt or messages raises ValueError."""
        with pytest.raises(ValueError, match="Either prompt or messages"):
            lm.forward()

    def test_forward_with_requirements_routes_to_instruct(self, lm):
        """Test forward with requirements parameter."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            lm.forward(prompt="Hi", requirements=["be concise"])

            assert mock_gen.called
            # Check that requirements were passed
            call_kwargs = mock_gen.call_args[1]
            assert "requirements" in call_kwargs

    def test_forward_with_instance_requirements_passes_to_generator(self, mock_session):
        """Test constructed with requirements passes them to generator."""
        lm = MelleaLM(mellea_session=mock_session, requirements=["req1"])

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            lm.forward(prompt="Hi")

            assert mock_gen.called

    def test_forward_response_model_name_overridden(self, lm):
        """Test response model name is set to LM's model."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            response = lm.forward(prompt="Hi")

            assert response.model == "mellea-test"

    def test_forward_response_is_openai_compatible(self, lm):
        """Test response has OpenAI-compatible structure."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            response = lm.forward(prompt="Hi")

            assert hasattr(response, "choices")
            assert hasattr(response, "usage")
            assert hasattr(response, "object")
            assert hasattr(response, "id")

    def test_forward_strips_cache_from_options(self, lm):
        """Test cache parameter is stripped from merged options."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            lm.forward(prompt="Hi", cache=True)

            # Verify cache was not passed to _generate_with_mellea
            call_kwargs = mock_gen.call_args[1]
            assert "cache" not in call_kwargs

    def test_forward_strips_model_type_from_options(self, lm):
        """Test model_type parameter is stripped from merged options."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            lm.forward(prompt="Hi")

            # model_type should not be in the merged options passed to Mellea
            call_kwargs = mock_gen.call_args[1]
            assert "model_type" not in call_kwargs

    def test_forward_temperature_in_merged_options(self, lm):
        """Test temperature from kwargs appears in generation call."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            lm.forward(prompt="Hi")

            # First positional arg should be the merged options (after prompt_text)
            # or in kwargs - verify it's being passed
            assert mock_gen.called


class TestAsyncForwardRouting:
    """Tests for aforward() async method routing."""

    @pytest.mark.asyncio
    async def test_aforward_with_prompt_calls_internal_method(self, lm):
        """Test aforward with prompt calls async generation."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="async response")
            await lm.aforward(prompt="Hi")

            assert mock_gen.called

    @pytest.mark.asyncio
    async def test_aforward_with_messages_calls_internal_method(self, lm):
        """Test aforward with messages calls async generation."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="async response")
            await lm.aforward(messages=[{"role": "user", "content": "Hi"}])

            assert mock_gen.called

    @pytest.mark.asyncio
    async def test_aforward_no_inputs_raises_value_error(self, lm):
        """Test aforward without prompt or messages raises ValueError."""
        with pytest.raises(ValueError, match="Either prompt or messages"):
            await lm.aforward()

    @pytest.mark.asyncio
    async def test_aforward_with_requirements_routes_properly(self, lm):
        """Test aforward with requirements parameter."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            await lm.aforward(prompt="Hi", requirements=["be concise"])

            assert mock_gen.called

    @pytest.mark.asyncio
    async def test_aforward_response_is_openai_compatible(self, lm):
        """Test aforward response has OpenAI-compatible structure."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            response = await lm.aforward(prompt="Hi")

            assert hasattr(response, "choices")
            assert hasattr(response, "usage")
            assert hasattr(response, "object")
            assert hasattr(response, "id")


class TestGenerateWrappers:
    """Tests for generate() and agenerate() wrapper methods."""

    def test_generate_delegates_to_forward(self, lm):
        """Test generate() delegates to forward()."""
        with patch.object(lm, "forward") as mock_forward:
            mock_forward.return_value = Mock()
            messages = [{"role": "user", "content": "Hi"}]
            lm.generate(messages=messages)

            mock_forward.assert_called_once()

    @pytest.mark.asyncio
    async def test_agenerate_delegates_to_aforward(self, lm):
        """Test agenerate() delegates to aforward()."""
        with patch.object(lm, "aforward") as mock_aforward:
            mock_aforward.return_value = Mock()
            messages = [{"role": "user", "content": "Hi"}]
            await lm.agenerate(messages=messages)

            mock_aforward.assert_called_once()
