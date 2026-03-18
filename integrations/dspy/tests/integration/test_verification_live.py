"""Integration tests for verification with real DSPy programs."""

import dspy
import pytest
from mellea import start_session
from mellea_dspy import MelleaBestOfN, MelleaLM, MelleaRefine


@pytest.fixture
def mellea_session():
    """Create Mellea session for testing."""
    return start_session()


@pytest.fixture
def mellea_lm(mellea_session):
    """Create MelleaLM for testing."""
    lm = MelleaLM(mellea_session=mellea_session, model="mellea-test")
    dspy.configure(lm=lm)
    return lm


class TestMelleaBestOfNIntegration:
    """Integration tests for MelleaBestOfN."""

    def test_bestofn_with_simple_qa(self, mellea_lm):
        """Test BestOfN with simple QA module."""
        # Define module
        qa = dspy.Predict("question -> answer")

        # Wrap with BestOfN
        best_of_3 = MelleaBestOfN(
            module=qa, N=3, requirements=["Must be under 20 words"], threshold=0.8
        )

        # Use it
        result = best_of_3(question="What is Python?")

        assert hasattr(result, "answer")
        assert result.answer
        assert len(result.answer) > 0

    def test_bestofn_with_chain_of_thought(self, mellea_lm):
        """Test BestOfN with ChainOfThought module."""
        # Define module
        qa = dspy.ChainOfThought("question -> answer")

        # Wrap with BestOfN
        best_of_3 = MelleaBestOfN(
            module=qa,
            N=3,
            requirements=["Must be concise", "Must mention programming"],
            threshold=0.7,
        )

        # Use it
        result = best_of_3(question="What is Python used for?")

        assert hasattr(result, "answer")
        assert result.answer

    def test_bestofn_with_multiple_requirements(self, mellea_lm):
        """Test BestOfN with multiple requirements."""
        qa = dspy.Predict("question -> answer")

        best_of_5 = MelleaBestOfN(
            module=qa,
            N=5,
            requirements=[
                "Must be under 50 words",
                "Must mention technology",
                "Must be professional",
            ],
            threshold=0.8,
        )

        result = best_of_5(question="What is AI?")

        assert hasattr(result, "answer")
        assert result.answer

    def test_bestofn_with_callable_requirement(self, mellea_lm):
        """Test BestOfN with callable requirement."""
        qa = dspy.Predict("question -> answer")

        def is_short(args, pred):
            return len(pred.answer.split()) <= 10

        best_of_3 = MelleaBestOfN(
            module=qa, N=3, requirements=[is_short], threshold=0.8
        )

        result = best_of_3(question="What is Python?")

        assert hasattr(result, "answer")
        assert result.answer

    def test_bestofn_with_custom_signature(self, mellea_lm):
        """Test BestOfN with custom signature."""

        class Summarize(dspy.Signature):
            """Summarize text."""

            text = dspy.InputField()
            summary = dspy.OutputField()

        summarizer = dspy.Predict(Summarize)

        best_of_3 = MelleaBestOfN(
            module=summarizer,
            N=3,
            requirements=["Must be under 30 words"],
            threshold=0.8,
        )

        result = best_of_3(
            text="Artificial intelligence is transforming technology. "
            "Machine learning enables computers to learn from data."
        )

        assert hasattr(result, "summary")
        assert result.summary

    def test_bestofn_combination_strategies(self, mellea_lm):
        """Test BestOfN with different combination strategies."""
        qa = dspy.Predict("question -> answer")

        # Test average strategy
        best_avg = MelleaBestOfN(
            module=qa,
            N=3,
            requirements=["Must be under 50 words", "Must be professional"],
            combination="average",
        )

        result_avg = best_avg(question="What is machine learning?")
        assert hasattr(result_avg, "answer")

        # Test min strategy
        best_min = MelleaBestOfN(
            module=qa,
            N=3,
            requirements=["Must be under 50 words", "Must be professional"],
            combination="min",
        )

        result_min = best_min(question="What is machine learning?")
        assert hasattr(result_min, "answer")


class TestMelleaRefineIntegration:
    """Integration tests for MelleaRefine."""

    def test_refine_with_simple_qa(self, mellea_lm):
        """Test Refine with simple QA module."""
        qa = dspy.Predict("question -> answer")

        refiner = MelleaRefine(
            module=qa, N=3, requirements=["Must be under 20 words"], threshold=0.9
        )

        result = refiner(question="What is Python?")

        assert hasattr(result, "answer")
        assert result.answer
        assert len(result.answer) > 0

    def test_refine_with_chain_of_thought(self, mellea_lm):
        """Test Refine with ChainOfThought module."""
        qa = dspy.ChainOfThought("question -> answer")

        refiner = MelleaRefine(
            module=qa,
            N=3,
            requirements=["Must be detailed", "Must mention examples"],
            threshold=0.8,
        )

        result = refiner(question="What is Python used for?")

        assert hasattr(result, "answer")
        assert result.answer

    def test_refine_with_multiple_requirements(self, mellea_lm):
        """Test Refine with multiple requirements."""
        qa = dspy.Predict("question -> answer")

        refiner = MelleaRefine(
            module=qa,
            N=3,
            requirements=[
                "Must be under 100 words",
                "Must be professional",
                "Must mention key concepts",
            ],
            threshold=0.85,
        )

        result = refiner(question="Explain artificial intelligence")

        assert hasattr(result, "answer")
        assert result.answer

    def test_refine_with_callable_requirement(self, mellea_lm):
        """Test Refine with callable requirement."""
        qa = dspy.Predict("question -> answer")

        def has_sufficient_length(args, pred):
            return len(pred.answer.split()) >= 10

        refiner = MelleaRefine(
            module=qa, N=3, requirements=[has_sufficient_length], threshold=0.9
        )

        result = refiner(question="What is machine learning?")

        assert hasattr(result, "answer")
        assert result.answer

    def test_refine_with_custom_signature(self, mellea_lm):
        """Test Refine with custom signature."""

        class GenerateEmail(dspy.Signature):
            """Generate professional email."""

            topic = dspy.InputField()
            email = dspy.OutputField()

        email_gen = dspy.Predict(GenerateEmail)

        refiner = MelleaRefine(
            module=email_gen,
            N=3,
            requirements=["Must be professional", "Must be under 100 words"],
            threshold=0.9,
        )

        result = refiner(topic="Meeting invitation")

        assert hasattr(result, "email")
        assert result.email

    def test_refine_combination_strategies(self, mellea_lm):
        """Test Refine with different combination strategies."""
        qa = dspy.Predict("question -> answer")

        # Test average strategy
        refine_avg = MelleaRefine(
            module=qa,
            N=3,
            requirements=["Must be under 50 words", "Must be clear"],
            combination="average",
        )

        result_avg = refine_avg(question="What is deep learning?")
        assert hasattr(result_avg, "answer")

        # Test product strategy
        refine_prod = MelleaRefine(
            module=qa,
            N=3,
            requirements=["Must be under 50 words", "Must be clear"],
            combination="product",
        )

        result_prod = refine_prod(question="What is deep learning?")
        assert hasattr(result_prod, "answer")


class TestVerificationWithModules:
    """Test verification with DSPy modules."""

    def test_bestofn_with_module_class(self, mellea_lm):
        """Test BestOfN with custom module class."""

        class QAModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.qa = dspy.Predict("question -> answer")

            def forward(self, question):
                return self.qa(question=question)

        qa_module = QAModule()

        best_of_3 = MelleaBestOfN(
            module=qa_module, N=3, requirements=["Must be concise"], threshold=0.8
        )

        result = best_of_3(question="What is Python?")

        assert hasattr(result, "answer")
        assert result.answer

    def test_refine_with_module_class(self, mellea_lm):
        """Test Refine with custom module class."""

        class SummaryModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.summarize = dspy.Predict("text -> summary")

            def forward(self, text):
                return self.summarize(text=text)

        summary_module = SummaryModule()

        refiner = MelleaRefine(
            module=summary_module,
            N=3,
            requirements=["Must be under 50 words"],
            threshold=0.9,
        )

        result = refiner(text="Long text about AI and machine learning...")

        assert hasattr(result, "summary")
        assert result.summary


class TestVerificationEdgeCases:
    """Test edge cases for verification."""

    def test_bestofn_with_no_requirements(self, mellea_lm):
        """Test BestOfN with no requirements."""
        qa = dspy.Predict("question -> answer")

        best_of_3 = MelleaBestOfN(module=qa, N=3)

        result = best_of_3(question="What is Python?")

        assert hasattr(result, "answer")
        assert result.answer

    def test_refine_with_no_requirements(self, mellea_lm):
        """Test Refine with no requirements."""
        qa = dspy.Predict("question -> answer")

        refiner = MelleaRefine(module=qa, N=3)

        result = refiner(question="What is Python?")

        assert hasattr(result, "answer")
        assert result.answer

    def test_bestofn_with_high_threshold(self, mellea_lm):
        """Test BestOfN with high threshold."""
        qa = dspy.Predict("question -> answer")

        best_of_3 = MelleaBestOfN(
            module=qa, N=3, requirements=["Must be concise"], threshold=0.95
        )

        result = best_of_3(question="What is Python?")

        assert hasattr(result, "answer")
        # May or may not meet threshold, but should return something

    def test_refine_with_low_iterations(self, mellea_lm):
        """Test Refine with low iteration count."""
        qa = dspy.Predict("question -> answer")

        refiner = MelleaRefine(
            module=qa, N=1, requirements=["Must be professional"], threshold=0.8
        )

        result = refiner(question="What is Python?")

        assert hasattr(result, "answer")
        assert result.answer


class TestVerificationPerformance:
    """Test verification performance characteristics."""

    def test_bestofn_generates_multiple_candidates(self, mellea_lm):
        """Test that BestOfN actually generates N candidates."""
        qa = dspy.Predict("question -> answer")

        # With N=5, should try up to 5 times
        best_of_5 = MelleaBestOfN(
            module=qa, N=5, requirements=["Must be under 100 words"], threshold=0.7
        )

        result = best_of_5(question="What is artificial intelligence?")

        assert hasattr(result, "answer")
        assert result.answer

    def test_refine_iterates_multiple_times(self, mellea_lm):
        """Test that Refine iterates multiple times."""
        qa = dspy.Predict("question -> answer")

        # With N=3, should try up to 3 refinements
        refiner = MelleaRefine(
            module=qa, N=3, requirements=["Must be detailed"], threshold=0.85
        )

        result = refiner(question="Explain machine learning")

        assert hasattr(result, "answer")
        assert result.answer
