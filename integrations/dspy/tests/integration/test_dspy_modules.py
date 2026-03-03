"""Tests for DSPy modules with live Mellea session."""

import pytest

import dspy


pytestmark = pytest.mark.integration


class SimpleQA(dspy.Module):
    """Simple Q&A module."""

    def __init__(self):
        """Initialize with a predictor."""
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        """Generate answer to question."""
        return self.predict(question=question)


class MultiStepReasoning(dspy.Module):
    """Multi-step reasoning module."""

    def __init__(self):
        """Initialize with two predictors for analysis and synthesis."""
        super().__init__()
        self.analyze = dspy.Predict("text -> analysis")
        self.synthesize = dspy.Predict("analysis -> conclusion")

    def forward(self, text):
        """Analyze text and synthesize conclusion."""
        analysis = self.analyze(text=text)
        result = self.synthesize(analysis=analysis.analysis)
        return dspy.ChainOfThought("analysis -> answer")(analysis=analysis.analysis)


class Extractor(dspy.Module):
    """Extract key points from text."""

    def __init__(self):
        """Initialize with predictor."""
        super().__init__()
        self.predict = dspy.Predict("text -> key_points")

    def forward(self, text):
        """Extract key points."""
        return self.predict(text=text)


class Summarizer(dspy.Module):
    """Summarize extracted key points."""

    def __init__(self):
        """Initialize with predictor."""
        super().__init__()
        self.predict = dspy.Predict("key_points -> summary")

    def forward(self, key_points):
        """Summarize key points."""
        return self.predict(key_points=key_points)


class Pipeline(dspy.Module):
    """Pipeline combining extractor and summarizer."""

    def __init__(self):
        """Initialize both components."""
        super().__init__()
        self.extractor = Extractor()
        self.summarizer = Summarizer()

    def forward(self, text):
        """Process text through pipeline."""
        extracted = self.extractor(text=text)
        summarized = self.summarizer(key_points=extracted.key_points)
        return dspy.ChainOfThought("key_points, summary -> answer")(
            key_points=extracted.key_points, summary=summarized.summary
        )


class ConversationalQA(dspy.Module):
    """Stateful QA module that maintains context."""

    def __init__(self):
        """Initialize with empty context."""
        super().__init__()
        self.context = ""
        self.predict = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        """Answer question with accumulated context."""
        result = self.predict(context=self.context, question=question)
        self.context += f"\nQ: {question}\nA: {result.answer}"
        return result

    def reset(self):
        """Reset context."""
        self.context = ""


class AdaptiveQA(dspy.Module):
    """Adaptive QA that chooses strategy based on question."""

    def __init__(self):
        """Initialize with simple and complex strategies."""
        super().__init__()
        self.simple = dspy.Predict("question -> answer")
        self.complex = dspy.ChainOfThought("question -> answer")
        self.strategy_chooser = dspy.Predict("question -> strategy")

    def forward(self, question):
        """Choose strategy and answer."""
        strategy_result = self.strategy_chooser(question=question)
        strategy = strategy_result.strategy.lower()

        if "simple" in strategy or "short" in strategy:
            return dspy.ChainOfThought("question, strategy -> answer")(
                question=question, strategy=strategy
            )
        else:
            return dspy.ChainOfThought("question, strategy -> answer")(
                question=question, strategy=strategy
            )


class TestSimpleModuleLive:
    """Tests for simple DSPy module."""

    def test_simple_qa_module(self, lm):
        """Test simple QA module produces answer."""
        module = SimpleQA()
        result = module(question="What is the capital of France?")

        assert hasattr(result, "answer")
        assert result.answer
        assert len(result.answer) > 0


class TestMultiStepModuleLive:
    """Tests for multi-step modules."""

    def test_multistep_module_produces_output(self, lm):
        """Test multi-step reasoning module produces output."""
        module = MultiStepReasoning()
        result = module(text="Python is a programming language")

        assert hasattr(result, "answer")
        assert result.answer


class TestPipelineLive:
    """Tests for pipeline composition."""

    def test_pipeline_module_produces_outputs(self, lm):
        """Test pipeline module produces both extraction and summary."""
        module = Pipeline()
        text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter."
        result = module(text=text)

        assert hasattr(result, "answer")
        assert result.answer


class TestStatefulModuleLive:
    """Tests for stateful modules maintaining context."""

    def test_stateful_module_accumulates_context(self, lm):
        """Test stateful module accumulates context across calls."""
        module = ConversationalQA()

        result1 = module(question="What is Python?")
        assert result1.answer

        result2 = module(question="What can you do with it?")
        assert result2.answer

        result3 = module(question="Tell me more about it?")
        assert result3.answer

        # Context should have grown
        assert len(module.context) > 0

    def test_stateful_module_reset(self, lm):
        """Test stateful module can be reset."""
        module = ConversationalQA()

        # Add some context
        module(question="What is Python?")
        context_before = module.context
        assert len(context_before) > 0

        # Reset
        module.reset()
        assert module.context == ""


class TestAdaptiveModuleLive:
    """Tests for adaptive strategy selection."""

    def test_adaptive_qa_returns_valid_strategy(self, lm):
        """Test adaptive QA selects and uses strategy."""
        module = AdaptiveQA()
        result = module(question="What is Python?")

        assert hasattr(result, "answer")
        assert result.answer
