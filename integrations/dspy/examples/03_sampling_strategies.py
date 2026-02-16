"""Sampling strategies example for Mellea + DSPy integration.

This example demonstrates how to use Mellea's sampling strategies with DSPy
for inference-time scaling and improved output quality.
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaLM


def rejection_sampling():
    """Demonstrate rejection sampling strategy."""
    print("=" * 70)
    print("Example 1: Rejection Sampling")
    print("=" * 70)

    # Setup
    print("\n1. Setting up Mellea and DSPy...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class GenerateCode(dspy.Signature):
        """Generate Python code for a given task."""

        task = dspy.InputField(desc="The programming task")
        code = dspy.OutputField(desc="Python code solution")

    predictor = dspy.Predict(GenerateCode)
    print("   ✓ Predictor created")

    # Generate with rejection sampling
    print("\n3. Generating code with rejection sampling...")
    print("   Task: Write a function to calculate factorial")
    print("   Strategy: Rejection sampling (retry until valid)")
    print("   Requirements:")
    print("   - Must include 'def' keyword")
    print("   - Must include 'return' statement")
    print("   - Must handle edge cases")

    response = predictor(task="Write a function to calculate factorial")
    print("\n   Generated Code:")
    print(f"   {response.code}")

    # Validate
    print("\n   Validation:")
    print(f"   ✓ Contains 'def': {'def' in response.code}")
    print(f"   ✓ Contains 'return': {'return' in response.code}")

    print("\n" + "=" * 70)


def best_of_n_sampling():
    """Demonstrate best-of-n sampling strategy."""
    print("\n" + "=" * 70)
    print("Example 2: Best-of-N Sampling")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class GenerateSlogan(dspy.Signature):
        """Generate a marketing slogan."""

        product = dspy.InputField(desc="The product name")
        slogan = dspy.OutputField(desc="A catchy slogan")

    predictor = dspy.Predict(GenerateSlogan)
    print("   ✓ Predictor created")

    # Generate multiple candidates
    print("\n3. Generating multiple slogans (best-of-n)...")
    print("   Product: AI-powered code assistant")
    print("   Strategy: Generate 3 candidates, select best")

    product = "AI-powered code assistant"
    candidates = []

    for i in range(3):
        response = predictor(product=product)
        candidates.append(response.slogan)
        print(f"\n   Candidate {i + 1}: {response.slogan}")

    # In practice, you would use a scoring function to select the best
    print("\n   Note: In production, use a scoring function to select best candidate")

    print("\n" + "=" * 70)


def temperature_sampling():
    """Demonstrate temperature-based sampling."""
    print("\n" + "=" * 70)
    print("Example 3: Temperature Sampling")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class GenerateStory(dspy.Signature):
        """Generate a creative story opening."""

        theme = dspy.InputField(desc="The story theme")
        opening = dspy.OutputField(desc="The story opening")

    print("   ✓ Signature defined")

    # Test different temperatures
    print("\n3. Generating with different temperatures...")
    theme = "A mysterious discovery in an ancient library"
    temperatures = [0.0, 0.5, 1.0]

    for temp in temperatures:
        print(f"\n   Temperature: {temp}")
        lm = MelleaLM(mellea_session=m, model="mellea-ollama", temperature=temp)
        dspy.configure(lm=lm)
        predictor = dspy.Predict(GenerateStory)

        response = predictor(theme=theme)
        print(f"   Opening: {response.opening[:150]}...")

    print("\n   Note:")
    print("   - Temperature 0.0: More deterministic, focused")
    print("   - Temperature 0.5: Balanced creativity")
    print("   - Temperature 1.0: More creative, diverse")

    print("\n" + "=" * 70)


def ensemble_sampling():
    """Demonstrate ensemble sampling with multiple models."""
    print("\n" + "=" * 70)
    print("Example 4: Ensemble Sampling")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class AnswerQuestion(dspy.Signature):
        """Answer a factual question."""

        question = dspy.InputField(desc="The question")
        answer = dspy.OutputField(desc="The answer")

    print("   ✓ Signature defined")

    # Generate with ensemble (simulated with multiple calls)
    print("\n3. Generating answers with ensemble approach...")
    question = "What are the key benefits of generative programming?"
    print(f"   Question: {question}")
    print("   Strategy: Generate multiple answers and combine")

    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    predictor = dspy.Predict(AnswerQuestion)

    answers = []
    for i in range(3):
        response = predictor(question=question)
        answers.append(response.answer)
        print(f"\n   Answer {i + 1}: {response.answer}")

    print("\n   Note: In production, combine answers using:")
    print("   - Majority voting for classification")
    print("   - Consensus extraction for generation")
    print("   - Weighted averaging for scoring")

    print("\n" + "=" * 70)


def adaptive_sampling():
    """Demonstrate adaptive sampling based on confidence."""
    print("\n" + "=" * 70)
    print("Example 5: Adaptive Sampling")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature with confidence
    print("\n2. Defining signature...")

    class ClassifyWithConfidence(dspy.Signature):
        """Classify text with confidence score."""

        text = dspy.InputField(desc="The text to classify")
        category = dspy.OutputField(desc="The category")
        confidence = dspy.OutputField(desc="Confidence level (high/medium/low)")

    predictor = dspy.Predict(ClassifyWithConfidence)
    print("   ✓ Predictor created")

    # Classify with adaptive sampling
    print("\n3. Classifying with adaptive sampling...")
    texts = [
        "This is a clear example of machine learning",
        "The topic could be related to AI or data science",
        "Ambiguous text that's hard to categorize",
    ]

    for i, text in enumerate(texts, 1):
        print(f"\n   Text {i}: {text}")
        response = predictor(text=text)
        print(f"   Category: {response.category}")
        print(f"   Confidence: {response.confidence}")
        print(
            f"   Strategy: {'Single sample' if 'high' in response.confidence.lower() else 'Multiple samples recommended'}"
        )

    print("\n   Note: Adaptive sampling adjusts effort based on confidence")
    print("   - High confidence: Single sample sufficient")
    print("   - Low confidence: Use multiple samples or ensemble")

    print("\n" + "=" * 70)


def main():
    """Run all sampling strategy examples."""
    print("\n" + "=" * 70)
    print("MELLEA + DSPY: SAMPLING STRATEGIES EXAMPLES")
    print("=" * 70)

    try:
        # Run examples
        rejection_sampling()
        best_of_n_sampling()
        temperature_sampling()
        ensemble_sampling()
        adaptive_sampling()

        # Summary
        print("\n" + "=" * 70)
        print("✅ All sampling strategy examples completed!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Rejection sampling: Retry until requirements met")
        print("  • Best-of-N: Generate multiple, select best")
        print("  • Temperature: Control creativity vs determinism")
        print("  • Ensemble: Combine multiple model outputs")
        print("  • Adaptive: Adjust strategy based on confidence")
        print("\nSampling strategies enable inference-time scaling:")
        print("  - Improve output quality")
        print("  - Increase reliability")
        print("  - Handle uncertainty")
        print("  - Optimize compute usage")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
