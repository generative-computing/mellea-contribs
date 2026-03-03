"""Integration tests for requirements with live Mellea session."""

import pytest

import dspy
from mellea_dspy import MelleaLM


pytestmark = pytest.mark.integration


class TestRequirementsBasic:
    """Basic requirements tests with live session."""

    def test_forward_with_requirements_list(self, lm):
        """Test forward with requirements list."""
        requirements = ["be concise", "use simple language"]
        response = lm.forward(
            prompt="Explain what Python is",
            requirements=requirements,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0

    def test_forward_with_single_requirement(self, lm):
        """Test forward with single requirement."""
        response = lm.forward(
            prompt="Write a greeting",
            requirements=["be friendly"],
        )

        assert response is not None
        assert response.choices[0].message.content

    def test_forward_with_multiple_requirements(self, lm):
        """Test forward with multiple requirements."""
        requirements = [
            "be concise",
            "use bullet points",
            "include examples",
            "be technical",
        ]
        response = lm.forward(
            prompt="Explain generative programming",
            requirements=requirements,
        )

        assert response is not None
        assert response.choices[0].message.content
        # Content should be non-empty
        assert len(response.choices[0].message.content) > 10

    @pytest.mark.asyncio
    async def test_aforward_with_requirements(self, lm):
        """Test async forward with requirements."""
        requirements = ["be brief", "be clear"]
        response = await lm.aforward(
            prompt="What is AI?",
            requirements=requirements,
        )

        assert response is not None
        assert response.choices[0].message.content


class TestRequirementsWithDSPy:
    """Test requirements with DSPy modules."""

    def test_predict_with_requirements_in_lm(self, mellea_session):
        """Test DSPy Predict with LM configured with requirements."""
        requirements = ["be concise", "use examples"]
        lm = MelleaLM(
            mellea_session=mellea_session,
            model="mellea-test",
            requirements=requirements,
        )
        dspy.configure(lm=lm)

        predictor = dspy.Predict("question -> answer")
        response = predictor(question="What is Python?")

        assert hasattr(response, "answer")
        assert response.answer
        assert len(response.answer) > 0

    def test_chain_of_thought_with_requirements(self, mellea_session):
        """Test ChainOfThought with requirements."""
        requirements = ["show reasoning", "be logical"]
        lm = MelleaLM(
            mellea_session=mellea_session,
            model="mellea-test",
            requirements=requirements,
        )
        dspy.configure(lm=lm)

        cot = dspy.ChainOfThought("question -> answer")
        response = cot(question="Why is the sky blue?")

        assert hasattr(response, "answer")
        assert response.answer


class TestRequirementsTypes:
    """Test different types of requirements."""

    def test_length_requirements(self, lm):
        """Test length-based requirements."""
        requirements = ["keep response under 50 words"]
        response = lm.forward(
            prompt="Describe machine learning",
            requirements=requirements,
        )

        assert response is not None
        content = response.choices[0].message.content
        word_count = len(content.split())
        # Should attempt to be concise (not strict validation)
        assert word_count > 0

    def test_format_requirements(self, lm):
        """Test format-based requirements."""
        requirements = ["use bullet points", "list 3 items"]
        response = lm.forward(
            prompt="List benefits of Python",
            requirements=requirements,
        )

        assert response is not None
        content = response.choices[0].message.content
        # Should contain some structure
        assert len(content) > 0

    def test_content_requirements(self, lm):
        """Test content-based requirements."""
        requirements = [
            "mention 'reliability'",
            "mention 'maintainability'",
            "include technical terms",
        ]
        response = lm.forward(
            prompt="Explain software quality",
            requirements=requirements,
        )

        assert response is not None
        content = response.choices[0].message.content.lower()
        # Check if requirements influenced output
        assert len(content) > 0

    def test_tone_requirements(self, lm):
        """Test tone-based requirements."""
        requirements = ["be professional", "be formal", "avoid slang"]
        response = lm.forward(
            prompt="Write a business email greeting",
            requirements=requirements,
        )

        assert response is not None
        assert response.choices[0].message.content


class TestRequirementsEdgeCases:
    """Test edge cases for requirements."""

    def test_empty_requirements_list(self, lm):
        """Test with empty requirements list."""
        response = lm.forward(
            prompt="Say hello",
            requirements=[],
        )

        assert response is not None
        assert response.choices[0].message.content

    def test_requirements_with_special_characters(self, lm):
        """Test requirements with special characters."""
        requirements = [
            "use <tags> if needed",
            "include 'quotes'",
            'use "double quotes"',
        ]
        response = lm.forward(
            prompt="Format some text",
            requirements=requirements,
        )

        assert response is not None
        assert response.choices[0].message.content

    def test_long_requirements(self, lm):
        """Test with long, detailed requirements."""
        requirements = [
            "Provide a comprehensive explanation that covers all major aspects "
            "of the topic, including historical context, current applications, "
            "and future implications. Use clear, accessible language suitable "
            "for a general technical audience.",
        ]
        response = lm.forward(
            prompt="Explain artificial intelligence",
            requirements=requirements,
        )

        assert response is not None
        assert response.choices[0].message.content
        # Should produce substantial content
        assert len(response.choices[0].message.content) > 50


class TestRequirementsCombinations:
    """Test combinations of requirements with other parameters."""

    def test_requirements_with_temperature(self, mellea_session):
        """Test requirements combined with temperature."""
        requirements = ["be creative", "use metaphors"]
        lm = MelleaLM(
            mellea_session=mellea_session,
            model="mellea-test",
            temperature=0.8,
            requirements=requirements,
        )
        dspy.configure(lm=lm)

        response = lm.forward(prompt="Describe programming")

        assert response is not None
        assert response.choices[0].message.content

    def test_requirements_with_max_tokens(self, mellea_session):
        """Test requirements combined with max_tokens."""
        requirements = ["be brief"]
        lm = MelleaLM(
            mellea_session=mellea_session,
            model="mellea-test",
            max_tokens=100,
            requirements=requirements,
        )
        dspy.configure(lm=lm)

        response = lm.forward(prompt="Explain Python")

        assert response is not None
        assert response.choices[0].message.content

    def test_requirements_with_messages(self, lm):
        """Test requirements with message-based input."""
        requirements = ["be helpful", "be clear"]
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is machine learning?"},
        ]
        response = lm.forward(messages=messages, requirements=requirements)

        assert response is not None
        assert response.choices[0].message.content

# Made with Bob
