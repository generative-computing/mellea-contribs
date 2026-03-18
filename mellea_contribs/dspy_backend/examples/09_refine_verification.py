"""Refine verification with Mellea requirements.

This example demonstrates how to use MelleaRefine to iteratively improve
outputs based on Mellea requirements.
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaLM, MelleaRefine


def basic_refine_example():
    """Example 1: Basic Refine usage."""
    print("=" * 70)
    print("Example 1: Basic Refine Usage")
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

    # Wrap with Refine
    print("\n3. Creating Refine wrapper with requirements...")
    refiner = MelleaRefine(
        module=qa,
        N=3,
        requirements=["Must be under 30 words", "Must be professional"],
        threshold=0.9,
    )
    print("   ✓ Refine wrapper created")
    print("   - Will refine up to 3 times")
    print("   - Requirements: under 30 words, professional")
    print("   - Threshold: 0.9")

    # Use it
    print("\n4. Asking question...")
    question = "What is Python?"
    print(f"   Question: {question}")

    result = refiner(question=question)

    print(f"\n   Answer: {result.answer}")
    print(f"   Word count: {len(result.answer.split())} words")
    print("   ✓ Refine iteratively improved the answer")

    print("\n" + "=" * 70)


def iterative_improvement_example():
    """Example 2: Iterative improvement."""
    print("\n" + "=" * 70)
    print("Example 2: Iterative Improvement")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define module
    print("\n2. Defining email generation module...")

    class GenerateEmail(dspy.Signature):
        """Generate professional email."""

        topic = dspy.InputField()
        recipient = dspy.InputField()
        email = dspy.OutputField()

    email_gen = dspy.Predict(GenerateEmail)
    print("   ✓ Module defined")

    # Wrap with Refine
    print("\n3. Creating Refine with strict requirements...")
    refiner = MelleaRefine(
        module=email_gen,
        N=5,
        requirements=[
            "Must be under 150 words",
            "Must be professional",
            "Must have greeting and closing",
        ],
        threshold=0.95,
    )
    print("   ✓ Refine wrapper created")
    print("   - Up to 5 refinement iterations")
    print("   - High threshold (0.95) for quality")

    # Use it
    print("\n4. Generating email...")
    result = refiner(topic="Project update meeting", recipient="Dr. Smith")

    print(f"\n   Email:\n{result.email}")
    print("\n   ✓ Email refined to meet all requirements")

    print("\n" + "=" * 70)


def quality_improvement_example():
    """Example 3: Quality improvement with detailed requirements."""
    print("\n" + "=" * 70)
    print("Example 3: Quality Improvement")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define module
    print("\n2. Defining explanation module...")
    explainer = dspy.Predict("concept -> explanation")
    print("   ✓ Module defined")

    # Wrap with Refine for quality
    print("\n3. Creating Refine for quality improvement...")
    refiner = MelleaRefine(
        module=explainer,
        N=4,
        requirements=[
            "Must be detailed",
            "Must include examples",
            "Must be under 200 words",
            "Must be professional",
        ],
        threshold=0.9,
    )
    print("   ✓ Refine wrapper created")
    print("   - Multiple quality requirements")
    print("   - Ensures detailed, example-rich output")

    # Use it
    print("\n4. Explaining concept...")
    concept = "Machine Learning"
    result = refiner(concept=concept)

    print(f"\n   Explanation:\n{result.explanation}")
    print(f"\n   Word count: {len(result.explanation.split())} words")
    print("   ✓ Explanation refined for quality")

    print("\n" + "=" * 70)


def custom_refinement_example():
    """Example 4: Custom refinement criteria."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Refinement Criteria")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define custom criteria
    print("\n2. Defining custom refinement criteria...")

    def has_code_example(args, pred):
        """Check if answer includes code example."""
        code_indicators = ["```", "def ", "class ", "import ", "print("]
        return 1.0 if any(ind in pred.answer for ind in code_indicators) else 0.0

    def appropriate_length(args, pred):
        """Check if answer has appropriate length (50-150 words)."""
        word_count = len(pred.answer.split())
        if 50 <= word_count <= 150:
            return 1.0
        elif word_count < 50:
            return word_count / 50
        else:
            excess = word_count - 150
            return max(0.0, 1.0 - (excess / 150))

    print("   ✓ Custom criteria defined")

    # Define module
    print("\n3. Defining code explanation module...")
    code_qa = dspy.Predict("question -> answer")
    print("   ✓ Module defined")

    # Wrap with custom criteria
    print("\n4. Creating Refine with custom criteria...")
    refiner = MelleaRefine(
        module=code_qa,
        N=3,
        requirements=[has_code_example, appropriate_length, "Must be clear"],
        threshold=0.85,
    )
    print("   ✓ Refine wrapper created")

    # Use it
    print("\n5. Asking coding question...")
    question = "How do you create a list in Python?"
    result = refiner(question=question)

    print(f"\n   Answer:\n{result.answer}")
    print("   ✓ Answer refined with custom criteria")

    print("\n" + "=" * 70)


def comparison_with_bestofn():
    """Example 5: Comparison with BestOfN."""
    print("\n" + "=" * 70)
    print("Example 5: Refine vs BestOfN")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define module
    qa = dspy.Predict("question -> answer")
    question = "What is artificial intelligence?"

    # Test with Refine
    print("\n2. Using Refine (iterative improvement)...")
    refiner = MelleaRefine(
        module=qa,
        N=3,
        requirements=["Must be under 50 words", "Must be professional"],
        threshold=0.9,
    )

    result_refine = refiner(question=question)
    print(f"   Refine result: {result_refine.answer[:100]}...")

    # Import BestOfN for comparison
    from mellea_dspy import MelleaBestOfN

    # Test with BestOfN
    print("\n3. Using BestOfN (parallel generation)...")
    best_of_3 = MelleaBestOfN(
        module=qa,
        N=3,
        requirements=["Must be under 50 words", "Must be professional"],
        threshold=0.9,
    )

    result_best = best_of_3(question=question)
    print(f"   BestOfN result: {result_best.answer[:100]}...")

    print("\n4. Key differences:")
    print("   Refine:")
    print("   - Iterative: Each attempt builds on previous")
    print("   - Good for: Gradual improvement, refinement")
    print("   - Use when: Quality matters more than diversity")
    print("\n   BestOfN:")
    print("   - Parallel: Independent attempts")
    print("   - Good for: Exploring variations, selection")
    print("   - Use when: Want best from diverse options")

    print("\n" + "=" * 70)


def refinement_strategies_example():
    """Example 6: Different refinement strategies."""
    print("\n" + "=" * 70)
    print("Example 6: Refinement Strategies")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define module
    summarizer = dspy.Predict("text -> summary")

    text = """
    Deep learning is a subset of machine learning that uses neural networks
    with multiple layers. These networks can learn hierarchical representations
    of data, making them particularly effective for tasks like image recognition,
    natural language processing, and speech recognition. The key advantage of
    deep learning is its ability to automatically learn features from raw data.
    """

    # Strategy 1: Strict refinement (high threshold, few iterations)
    print("\n2. Strategy 1: Strict refinement...")
    strict_refiner = MelleaRefine(
        module=summarizer,
        N=2,
        requirements=["Must be under 30 words", "Must be professional"],
        threshold=0.95,
    )

    result1 = strict_refiner(text=text)
    print(f"   Result: {result1.summary}")
    print(f"   Words: {len(result1.summary.split())}")

    # Strategy 2: Gradual refinement (moderate threshold, more iterations)
    print("\n3. Strategy 2: Gradual refinement...")
    gradual_refiner = MelleaRefine(
        module=summarizer,
        N=5,
        requirements=["Must be under 40 words", "Must mention key concepts"],
        threshold=0.8,
    )

    result2 = gradual_refiner(text=text)
    print(f"   Result: {result2.summary}")
    print(f"   Words: {len(result2.summary.split())}")

    # Strategy 3: Quality-focused (multiple requirements, high threshold)
    print("\n4. Strategy 3: Quality-focused...")
    quality_refiner = MelleaRefine(
        module=summarizer,
        N=4,
        requirements=[
            "Must be under 50 words",
            "Must be detailed",
            "Must be professional",
            "Must mention neural networks",
        ],
        threshold=0.9,
    )

    result3 = quality_refiner(text=text)
    print(f"   Result: {result3.summary}")
    print(f"   Words: {len(result3.summary.split())}")

    print("\n5. Strategy recommendations:")
    print("   - Strict: When you need precise output quickly")
    print("   - Gradual: When quality improvement is gradual")
    print("   - Quality: When multiple criteria must be met")

    print("\n" + "=" * 70)


def main():
    """Run all Refine examples."""
    print("\n" + "=" * 70)
    print("MELLEA + DSPY: REFINE VERIFICATION EXAMPLES")
    print("=" * 70)

    try:
        # Run examples
        basic_refine_example()
        iterative_improvement_example()
        quality_improvement_example()
        custom_refinement_example()
        comparison_with_bestofn()
        refinement_strategies_example()

        # Summary
        print("\n" + "=" * 70)
        print("✅ All Refine examples completed!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Refine iteratively improves outputs")
        print("  • Each iteration builds on previous attempt")
        print("  • Requirements guide refinement process")
        print("  • Threshold controls when to stop refining")
        print("  • Custom criteria enable complex refinement")
        print("  • Different from BestOfN (iterative vs parallel)")
        print("\nBest Practices:")
        print("  • Use N=3-5 for good refinement")
        print("  • Set high threshold (0.85-0.95) for quality")
        print("  • Combine multiple requirements")
        print("  • Use custom callables for complex checks")
        print("  • Choose Refine for gradual improvement")
        print("  • Choose BestOfN for diverse options")
        print("\nWhen to Use Refine:")
        print("  • Need iterative improvement")
        print("  • Quality matters more than diversity")
        print("  • Want to build on previous attempts")
        print("  • Have clear refinement criteria")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
