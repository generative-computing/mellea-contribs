"""Basic usage example for Mellea + DSPy integration.

This example demonstrates the fundamental usage patterns of using Mellea
as a backend for DSPy, including simple predictions and chain of thought reasoning.
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaLM


def simple_prediction():
    """Demonstrate simple prediction with DSPy and Mellea."""
    print("=" * 70)
    print("Example 1: Simple Prediction")
    print("=" * 70)

    # Step 1: Create Mellea session
    print("\n1. Creating Mellea session...")
    m = start_session()
    print("   ✓ Session created")

    # Step 2: Create MelleaLM and configure DSPy
    print("\n2. Setting up DSPy with Mellea backend...")
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ DSPy configured")

    # Step 3: Create a simple predictor
    print("\n3. Creating predictor with signature 'question -> answer'...")
    qa = dspy.Predict("question -> answer")
    print("   ✓ Predictor created")

    # Step 4: Make predictions
    print("\n4. Making predictions...")
    questions = [
        "What is generative programming?",
        "How does DSPy help with prompt engineering?",
        "What are the benefits of using Mellea?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n   Question {i}: {question}")
        response = qa(question=question)
        print(f"   Answer: {response.answer}")

    print("\n" + "=" * 70)


def chain_of_thought():
    """Demonstrate chain of thought reasoning."""
    print("\n" + "=" * 70)
    print("Example 2: Chain of Thought Reasoning")
    print("=" * 70)

    # Setup
    print("\n1. Setting up Mellea and DSPy...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Create Chain of Thought predictor
    print("\n2. Creating Chain of Thought predictor...")
    cot = dspy.ChainOfThought("question -> answer")
    print("   ✓ CoT predictor created")

    # Complex questions that benefit from reasoning
    print("\n3. Asking questions that require reasoning...")
    questions = [
        "Why is structured prompting better than ad-hoc prompts?",
        "How does Mellea's requirements validation improve reliability?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n   Question {i}: {question}")
        response = cot(question=question)
        # Check if rationale exists (DSPy may use different attribute names)
        if hasattr(response, "rationale"):
            print(f"   Reasoning: {response.rationale}")
        elif hasattr(response, "reasoning"):
            print(f"   Reasoning: {response.reasoning}")
        print(f"   Answer: {response.answer}")

    print("\n" + "=" * 70)


def custom_signature():
    """Demonstrate using custom DSPy signatures."""
    print("\n" + "=" * 70)
    print("Example 3: Custom Signatures")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define custom signature
    print("\n2. Defining custom signature...")

    class Summarize(dspy.Signature):
        """Summarize a text into a concise summary."""

        text = dspy.InputField(desc="The text to summarize")
        summary = dspy.OutputField(desc="A concise summary of the text")

    print("   ✓ Signature defined")

    # Create predictor with custom signature
    print("\n3. Creating predictor with custom signature...")
    summarizer = dspy.Predict(Summarize)
    print("   ✓ Predictor created")

    # Test with example text
    print("\n4. Summarizing text...")
    text = """
    Mellea is a library for writing generative programs. Generative programming
    replaces flaky agents and brittle prompts with structured, maintainable,
    robust, and efficient AI workflows. It provides a standard library of
    opinionated prompting patterns and sampling strategies for inference-time scaling.
    The library supports multiple model backends including Ollama, OpenAI, and Anthropic,
    making it flexible for different deployment scenarios.
    """

    response = summarizer(text=text)
    print(f"   Original length: {len(text)} characters")
    print(f"   Summary: {response.summary}")

    print("\n" + "=" * 70)


def multiple_outputs():
    """Demonstrate signatures with multiple outputs."""
    print("\n" + "=" * 70)
    print("Example 4: Multiple Output Fields")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature with multiple outputs
    print("\n2. Defining signature with multiple outputs...")

    class AnalyzeText(dspy.Signature):
        """Analyze text and provide multiple insights."""

        text = dspy.InputField(desc="The text to analyze")
        sentiment = dspy.OutputField(desc="The sentiment (positive/negative/neutral)")
        key_points = dspy.OutputField(desc="Key points from the text")
        category = dspy.OutputField(desc="The category or topic of the text")

    print("   ✓ Signature defined")

    # Create predictor
    print("\n3. Creating analyzer...")
    analyzer = dspy.Predict(AnalyzeText)
    print("   ✓ Analyzer created")

    # Analyze sample text
    print("\n4. Analyzing text...")
    text = """
    The new AI framework has revolutionized how developers build applications.
    It provides excellent tools for structured prompting and validation, making
    it much easier to create reliable AI systems. The community response has been
    overwhelmingly positive, with many developers praising its ease of use.
    """

    response = analyzer(text=text)
    print(f"   Text: {text.strip()[:100]}...")
    print(f"   Sentiment: {response.sentiment}")
    print(f"   Key Points: {response.key_points}")
    print(f"   Category: {response.category}")

    print("\n" + "=" * 70)


def main():
    """Run all basic examples."""
    print("\n" + "=" * 70)
    print("MELLEA + DSPY INTEGRATION: BASIC USAGE EXAMPLES")
    print("=" * 70)

    try:
        # Run examples
        simple_prediction()
        chain_of_thought()
        custom_signature()
        multiple_outputs()

        # Summary
        print("\n" + "=" * 70)
        print("✅ All basic examples completed successfully!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • MelleaLM integrates seamlessly with DSPy")
        print("  • Use dspy.Predict for simple predictions")
        print("  • Use dspy.ChainOfThought for reasoning tasks")
        print("  • Custom signatures provide structured I/O")
        print("  • Multiple output fields are fully supported")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
