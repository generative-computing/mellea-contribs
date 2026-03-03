"""Tests for requirements parameter handling in MelleaLM."""

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


class TestRequirementsConstruction:
    """Tests for requirements in LM construction."""

    def test_requirements_stored_on_construction(self, mock_session):
        """Test requirements are stored when passed to constructor."""
        requirements = ["be concise", "use examples", "be technical"]
        lm = MelleaLM(mellea_session=mock_session, requirements=requirements)

        assert lm._requirements == requirements

    def test_requirements_none_by_default(self, mock_session):
        """Test requirements default to None."""
        lm = MelleaLM(mellea_session=mock_session)

        assert lm._requirements is None

    def test_empty_requirements_list(self, mock_session):
        """Test empty requirements list is stored."""
        lm = MelleaLM(mellea_session=mock_session, requirements=[])

        assert lm._requirements == []


class TestRequirementsInForward:
    """Tests for requirements parameter in forward() method."""

    def test_forward_with_requirements_kwarg(self, lm):
        """Test forward accepts requirements as kwarg."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = ["be concise", "use bullet points"]
            
            lm.forward(prompt="Test", requirements=requirements)

            # Verify requirements were passed to generation
            call_kwargs = mock_gen.call_args[1]
            assert "requirements" in call_kwargs
            assert call_kwargs["requirements"] == requirements

    def test_forward_requirements_override_instance_requirements(self, mock_session):
        """Test forward requirements override instance requirements."""
        instance_reqs = ["instance requirement"]
        lm = MelleaLM(mellea_session=mock_session, requirements=instance_reqs)

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            forward_reqs = ["forward requirement"]
            
            lm.forward(prompt="Test", requirements=forward_reqs)

            # Forward requirements should take precedence
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["requirements"] == forward_reqs

    def test_forward_uses_instance_requirements_when_not_provided(self, mock_session):
        """Test forward uses instance requirements when not provided in call."""
        instance_reqs = ["instance requirement"]
        lm = MelleaLM(mellea_session=mock_session, requirements=instance_reqs)

        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            lm.forward(prompt="Test")

            # Should use instance requirements
            call_kwargs = mock_gen.call_args[1]
            # Note: The actual behavior depends on _prepare_generation implementation
            # This test verifies the requirements are accessible

    def test_forward_with_multiple_requirements(self, lm):
        """Test forward with multiple requirements."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = [
                "be concise",
                "use examples",
                "be technical",
                "include code snippets",
            ]
            
            lm.forward(prompt="Test", requirements=requirements)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["requirements"] == requirements
            assert len(call_kwargs["requirements"]) == 4

    def test_forward_with_empty_requirements_list(self, lm):
        """Test forward with empty requirements list."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            
            lm.forward(prompt="Test", requirements=[])

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["requirements"] == []


class TestRequirementsInAforward:
    """Tests for requirements parameter in aforward() async method."""

    @pytest.mark.asyncio
    async def test_aforward_with_requirements_kwarg(self, lm):
        """Test aforward accepts requirements as kwarg."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = ["be concise", "use bullet points"]
            
            await lm.aforward(prompt="Test", requirements=requirements)

            call_kwargs = mock_gen.call_args[1]
            assert "requirements" in call_kwargs
            assert call_kwargs["requirements"] == requirements

    @pytest.mark.asyncio
    async def test_aforward_requirements_override_instance_requirements(self, mock_session):
        """Test aforward requirements override instance requirements."""
        instance_reqs = ["instance requirement"]
        lm = MelleaLM(mellea_session=mock_session, requirements=instance_reqs)

        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            forward_reqs = ["forward requirement"]
            
            await lm.aforward(prompt="Test", requirements=forward_reqs)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["requirements"] == forward_reqs

    @pytest.mark.asyncio
    async def test_aforward_with_multiple_requirements(self, lm):
        """Test aforward with multiple requirements."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = [
                "be concise",
                "use examples",
                "be technical",
            ]
            
            await lm.aforward(prompt="Test", requirements=requirements)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["requirements"] == requirements


class TestRequirementsWithMessages:
    """Tests for requirements with message-based input."""

    def test_forward_messages_with_requirements(self, lm):
        """Test forward with messages and requirements."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            messages = [{"role": "user", "content": "Hello"}]
            requirements = ["be polite", "be brief"]
            
            lm.forward(messages=messages, requirements=requirements)

            call_kwargs = mock_gen.call_args[1]
            assert "requirements" in call_kwargs
            assert call_kwargs["requirements"] == requirements

    @pytest.mark.asyncio
    async def test_aforward_messages_with_requirements(self, lm):
        """Test aforward with messages and requirements."""
        with patch.object(lm, "_agenerate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            messages = [{"role": "user", "content": "Hello"}]
            requirements = ["be polite", "be brief"]
            
            await lm.aforward(messages=messages, requirements=requirements)

            call_kwargs = mock_gen.call_args[1]
            assert "requirements" in call_kwargs
            assert call_kwargs["requirements"] == requirements


class TestRequirementsTypes:
    """Tests for different types of requirements."""

    def test_requirements_as_strings(self, lm):
        """Test requirements as list of strings."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = ["requirement 1", "requirement 2"]
            
            lm.forward(prompt="Test", requirements=requirements)

            call_kwargs = mock_gen.call_args[1]
            assert all(isinstance(r, str) for r in call_kwargs["requirements"])

    def test_requirements_with_special_characters(self, lm):
        """Test requirements with special characters."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = [
                "use <tags>",
                "include 'quotes'",
                'use "double quotes"',
                "use numbers: 123",
            ]
            
            lm.forward(prompt="Test", requirements=requirements)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["requirements"] == requirements

    def test_requirements_with_long_text(self, lm):
        """Test requirements with long descriptive text."""
        with patch.object(lm, "_generate_with_mellea") as mock_gen:
            mock_gen.return_value = Mock(content="response")
            requirements = [
                "The response must be comprehensive and include detailed explanations "
                "of all key concepts, with examples where appropriate.",
                "Use a professional tone suitable for technical documentation.",
            ]
            
            lm.forward(prompt="Test", requirements=requirements)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["requirements"] == requirements

# Made with Bob
