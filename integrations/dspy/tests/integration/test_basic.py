"""Basic integration tests for MelleaLM with live Mellea session."""

import asyncio

import pytest

import dspy


pytestmark = pytest.mark.integration


class TestForwardLive:
    """Tests for forward() method with live Mellea session."""

    def test_forward_with_prompt(self, lm):
        """Test forward with a simple prompt."""
        response = lm.forward(prompt="Say hello")

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0

    def test_forward_with_messages(self, lm):
        """Test forward with message list."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        response = lm.forward(messages=messages)

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0

    def test_forward_response_openai_shape(self, lm):
        """Test forward response has correct OpenAI structure."""
        response = lm.forward(prompt="Say hello")

        # Verify OpenAI-compatible structure
        assert hasattr(response, "id")
        assert response.id.startswith("mellea-")
        assert hasattr(response, "object")
        assert response.object == "chat.completion"
        assert hasattr(response, "choices")
        assert hasattr(response, "usage")

    def test_forward_model_name_in_response(self, lm):
        """Test response contains the correct model name."""
        response = lm.forward(prompt="Say hello")

        assert response.model == "mellea-test"

    def test_forward_multiple_calls_independent(self, lm):
        """Test multiple forward calls produce independent responses."""
        response1 = lm.forward(prompt="Say hello")
        response2 = lm.forward(prompt="Say goodbye")

        assert response1.choices[0].message.content
        assert response2.choices[0].message.content
        # IDs should be different due to different content
        assert response1.id != response2.id


class TestAforwardLive:
    """Tests for aforward() async method with live Mellea session."""

    @pytest.mark.asyncio
    async def test_aforward_with_prompt(self, lm):
        """Test async forward with a simple prompt."""
        response = await lm.aforward(prompt="Say hello")

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.asyncio
    async def test_aforward_with_messages(self, lm):
        """Test async forward with message list."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        response = await lm.aforward(messages=messages)

        assert response is not None
        assert hasattr(response, "choices")
        assert response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_aforward_concurrent(self, lm):
        """Test concurrent async forward calls."""
        results = await asyncio.gather(
            lm.aforward(prompt="Say hello"),
            lm.aforward(prompt="Say goodbye"),
            lm.aforward(prompt="Say goodbye"),
        )

        assert len(results) == 3
        for response in results:
            assert hasattr(response, "choices")
            assert response.choices[0].message.content
            assert len(response.choices[0].message.content) > 0


class TestPredictLive:
    """Tests for DSPy Predict with live Mellea session."""

    def test_predict_simple_signature(self, lm):
        """Test DSPy Predict with simple signature."""
        predictor = dspy.Predict("question -> answer")
        response = predictor(question="What is 2+2?")

        assert hasattr(response, "answer")
        assert response.answer
        assert len(response.answer) > 0

    def test_predict_no_error_empty_prompt_fields(self, lm):
        """Test Predict handles minimal input gracefully."""
        predictor = dspy.Predict("input -> output")
        response = predictor(input="test")

        assert hasattr(response, "output")


class TestChainOfThoughtLive:
    """Tests for DSPy ChainOfThought with live Mellea session."""

    def test_cot_basic(self, lm):
        """Test ChainOfThought produces reasoning and answer."""
        cot = dspy.ChainOfThought("question -> answer")
        response = cot(question="Why is the sky blue?")

        assert hasattr(response, "answer")
        assert response.answer
        assert len(response.answer) > 0

    def test_cot_has_reasoning_or_answer(self, lm):
        """Test ChainOfThought produces either reasoning or answer."""
        cot = dspy.ChainOfThought("question -> answer")
        response = cot(question="What is Python?")

        # DSPy CoT should have at least answer (and ideally rationale/reasoning)
        assert hasattr(response, "answer") or hasattr(response, "rationale")
        assert response.answer if hasattr(response, "answer") else response.rationale


class TestCustomSignatureClassLive:
    """Tests for custom signature classes with live session."""

    def test_class_signature_single_output(self, lm):
        """Test custom signature class with single output."""

        class SummarizeSignature(dspy.Signature):
            """Summarize text concisely."""

            text: str = dspy.InputField()
            summary: str = dspy.OutputField()

        predictor = dspy.Predict(SummarizeSignature)
        response = predictor(
            text="The quick brown fox jumps over the lazy dog. "
            "This sentence contains all letters of the alphabet."
        )

        assert hasattr(response, "summary")
        assert response.summary
        assert len(response.summary) > 0

    def test_class_signature_multiple_outputs(self, lm):
        """Test custom signature class with multiple outputs."""

        class AnalyzeText(dspy.Signature):
            """Analyze text and provide multiple outputs."""

            text: str = dspy.InputField()
            sentiment: str = dspy.OutputField()
            key_points: str = dspy.OutputField()
            category: str = dspy.OutputField()

        predictor = dspy.Predict(AnalyzeText)
        response = predictor(text="The service was excellent and fast!")

        assert hasattr(response, "sentiment")
        assert hasattr(response, "key_points")
        assert hasattr(response, "category")
        assert response.sentiment
        assert response.key_points
        assert response.category


class TestRequirementsLive:
    """Tests for requirements parameter with live session."""

    def test_requirements_kwarg_does_not_error(self, lm):
        """Test forward accepts requirements without error."""
        response = lm.forward(
            prompt="Be concise: What is Python?", requirements=["be concise"]
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert response.choices[0].message.content
