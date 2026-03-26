"""BestOfN verification with Mellea requirements.

This example demonstrates how to use MelleaBestOfN to generate multiple
candidates and select the best one based on Mellea requirements.
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaBestOfN, MelleaLM


def basic_bestofn_example():
    """Example 1: Basic BestOfN usage."""
    print("=" * 70)
    print("Example 1: Basic BestOfN Usage")
    print("=" * 70)

    # Setup
    print("\n1. Setting up Mellea and DSPy...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define module
    print("\n2. Defining QA module...")
    qa = dspy.ChainOfThought("question -> answer")
    print("   ✓ Module defined")

    # Wrap with BestOfN
    print("\n3. Creating BestOfN wrapper with requirements...")
    best_of_5 = MelleaBestOfN(
        module=qa,
        N=5,
        requirements=["Must be one word", "Must be a proper noun"],
        threshold=0.8,
    )
    print("   ✓ BestOfN wrapper created")
    print("   - Will generate 5 candidates")
    print("   - Requirements: one word, proper noun")
    print("   - Threshold: 0.8")

    # Use it
    print("\n4. Asking question...")
    question = "What is the capital of Belgium?"
    print(f"   Question: {question}")

    result = best_of_5(question=question)

    print(f"\n   Answer: {result.answer}")
    print("   ✓ BestOfN selected the best answer from 5 candidates")

    print("\n" + "=" * 70)


def multiple_requirements_example():
    """Example 2: Multiple requirements."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Requirements")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define module
    print("\n2. Defining summarization module...")
    summarizer = dspy.Predict("text -> summary")
    print("   ✓ Module defined")

    # Wrap with multiple requirements
    print("\n3. Creating BestOfN with multiple requirements...")
    best_of_3 = MelleaBestOfN(
        module=summarizer,
        N=3,
        requirements=[
            "Must be under 50 words",
            "Must mention key points",
            "Must be professional",
        ],
        threshold=0.85,
    )
    print("   ✓ BestOfN wrapper created")
    print("   - 3 requirements specified")
    print("   - All must be satisfied for high score")

    # Use it
    print("\n4. Summarizing text...")
    text = """
    Artificial intelligence is transforming how we interact with technology.
    Machine learning algorithms can now process vast amounts of data to identify
    patterns and make predictions. Deep learning, a subset of machine learning,
    uses neural networks to achieve human-like performance in tasks such as
    image recognition and natural language processing.
    """

    result = best_of_3(text=text)

    print(f"\n   Summary: {result.summary}")
    print(f"   Word count: {len(result.summary.split())} words")
    print("   ✓ Best summary selected based on all requirements")

    print("\n" + "=" * 70)


def custom_requirement_example():
    """Example 3: Custom callable requirements."""
    print("\n" + "=" * 70)
    print("Example 3: Custom Callable Requirements")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define custom requirement functions
    print("\n2. Defining custom requirement functions...")

    def is_concise(args, pred):
        """Check if answer is concise (under 15 words)."""
        word_count = len(pred.answer.split())
        if word_count <= 10:
            return 1.0
        elif word_count <= 15:
            return 0.7
        else:
            return 0.3

    def mentions_technology(args, pred):
        """Check if answer mentions technology-related terms."""
        tech_terms = ["technology", "software", "programming", "code", "computer"]
        text_lower = pred.answer.lower()
        return 1.0 if any(term in text_lower for term in tech_terms) else 0.0

    print("   ✓ Custom functions defined")

    # Define module
    print("\n3. Defining QA module...")
    qa = dspy.Predict("question -> answer")
    print("   ✓ Module defined")

    # Wrap with custom requirements
    print("\n4. Creating BestOfN with custom requirements...")
    best_of_5 = MelleaBestOfN(
        module=qa,
        N=5,
        requirements=[is_concise, mentions_technology, "Must be professional"],
        threshold=0.8,
    )
    print("   ✓ BestOfN wrapper created")
    print("   - Mix of callable and string requirements")

    # Use it
    print("\n5. Asking question...")
    question = "What is Python?"
    print(f"   Question: {question}")

    result = best_of_5(question=question)

    print(f"\n   Answer: {result.answer}")
    print(f"   Word count: {len(result.answer.split())} words")
    print("   ✓ Answer meets custom requirements")

    print("\n" + "=" * 70)


def combination_strategies_example():
    """Example 4: Different combination strategies."""
    print("\n" + "=" * 70)
    print("Example 4: Combination Strategies")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define module
    qa = dspy.Predict("question -> answer")

    # Test different strategies
    print("\n2. Testing different combination strategies...")

    strategies = ["average", "min", "product"]
    question = "What is machine learning?"

    for strategy in strategies:
        print(f"\n   Strategy: {strategy}")

        best_of_3 = MelleaBestOfN(
            module=qa,
            N=3,
            requirements=["Must be under 50 words", "Must mention AI"],
            combination=strategy,
            threshold=0.7,
        )

        result = best_of_3(question=question)
        print(f"   Answer: {result.answer[:100]}...")

    print("\n3. Strategy comparison:")
    print("   - average: Average score of all requirements")
    print("   - min: Minimum score (strictest)")
    print("   - product: Product of scores (balanced)")

    print("\n" + "=" * 70)


def format_requirements_example():
    """Example 5: Format requirements."""
    print("\n" + "=" * 70)
    print("Example 5: Format Requirements")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Test bullet points
    print("\n2. Testing bullet point format...")
    list_gen = dspy.Predict("topic -> list")

    best_bullets = MelleaBestOfN(
        module=list_gen,
        N=3,
        requirements=["Must be in bullet points", "Must have at least 3 items"],
        threshold=0.8,
    )

    result = best_bullets(topic="Benefits of Python")
    print(f"   Result:\n{result.list}")

    # Test numbered list
    print("\n3. Testing numbered list format...")
    best_numbered = MelleaBestOfN(
        module=list_gen,
        N=3,
        requirements=["Must be numbered list", "Must be under 100 words"],
        threshold=0.8,
    )

    result = best_numbered(topic="Steps to learn programming")
    print(f"   Result:\n{result.list}")

    print("\n" + "=" * 70)


def threshold_behavior_example():
    """Example 6: Threshold behavior."""
    print("\n" + "=" * 70)
    print("Example 6: Threshold Behavior")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define module
    qa = dspy.Predict("question -> answer")

    # Test different thresholds
    print("\n2. Testing different thresholds...")

    thresholds = [0.5, 0.8, 0.95]
    question = "What is Python?"

    for threshold in thresholds:
        print(f"\n   Threshold: {threshold}")

        best_of_5 = MelleaBestOfN(
            module=qa,
            N=5,
            requirements=["Must be under 20 words", "Must be professional"],
            threshold=threshold,
        )

        result = best_of_5(question=question)
        print(f"   Answer: {result.answer}")

    print("\n3. Threshold explanation:")
    print("   - Lower threshold: More lenient, accepts partial matches")
    print("   - Higher threshold: Stricter, requires better matches")
    print("   - 1.0: Perfect match required")

    print("\n" + "=" * 70)


def main():
    """Run all BestOfN examples."""
    print("\n" + "=" * 70)
    print("MELLEA + DSPY: BESTOFN VERIFICATION EXAMPLES")
    print("=" * 70)

    try:
        # Run examples
        basic_bestofn_example()
        multiple_requirements_example()
        custom_requirement_example()
        combination_strategies_example()
        format_requirements_example()
        threshold_behavior_example()

        # Summary
        print("\n" + "=" * 70)
        print("✅ All BestOfN examples completed!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • BestOfN generates N candidates and selects best")
        print("  • Requirements can be strings or callables")
        print("  • Multiple requirements are combined")
        print("  • Threshold controls acceptance criteria")
        print("  • Different combination strategies available")
        print("  • Format requirements ensure output structure")
        print("\nBest Practices:")
        print("  • Use N=3-5 for good balance")
        print("  • Set threshold based on strictness needed")
        print("  • Combine multiple requirements for quality")
        print("  • Use custom callables for complex checks")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
