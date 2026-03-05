"""Example 10: Hybrid Approach - Wrappers vs Direct DSPy.

This example demonstrates both approaches for using Mellea requirements
with DSPy's BestOfN and Refine:

1. High-level wrappers (MelleaBestOfN, MelleaRefine) - Recommended for most users
2. Direct DSPy with create_reward_fn() - For advanced users who need more control

Both approaches use the same underlying requirement-to-reward conversion,
so you can choose based on your needs.
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaBestOfN, MelleaLM, MelleaRefine, create_reward_fn


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def approach_1_wrapper_classes():
    """Approach 1: Using high-level wrapper classes (Recommended)."""
    print_section("Approach 1: High-Level Wrappers (Recommended)")

    print("1. Setting up Mellea and DSPy...")
    m = start_session()
    lm = MelleaLM(mellea_session=m)
    dspy.configure(lm=lm)
    print("   ✓ Setup complete\n")

    print("2. Defining QA module...")
    qa = dspy.ChainOfThought("question -> answer")
    print("   ✓ Module defined\n")

    print("3. Using MelleaBestOfN wrapper...")
    print("   - Simple, Mellea-native API")
    print("   - One-step initialization")
    print("   - Automatic requirement conversion\n")

    # Simple one-step initialization
    best_of_5 = MelleaBestOfN(
        module=qa,
        N=5,
        requirements=["Must be under 30 words", "Must mention Belgium"],
        threshold=0.8,
    )

    result = best_of_5(question="What is the capital of Belgium?")
    print(f"   Answer: {result.answer}")
    print(f"   Word count: {len(result.answer.split())} words")
    print("   ✓ Simple and clean!\n")


def approach_2_direct_dspy():
    """Approach 2: Using direct DSPy with create_reward_fn() (Advanced)."""
    print_section("Approach 2: Direct DSPy with create_reward_fn() (Advanced)")

    print("1. Setting up Mellea and DSPy...")
    m = start_session()
    lm = MelleaLM(mellea_session=m)
    dspy.configure(lm=lm)
    print("   ✓ Setup complete\n")

    print("2. Defining QA module...")
    qa = dspy.ChainOfThought("question -> answer")
    print("   ✓ Module defined\n")

    print("3. Creating reward function with create_reward_fn()...")
    print("   - Two-step process")
    print("   - More control over reward function")
    print("   - Direct access to DSPy features\n")

    # Step 1: Create reward function
    reward_fn = create_reward_fn(
        requirements=["Must be under 30 words", "Must mention Belgium"],
        strategy="average",
    )
    print("   ✓ Reward function created\n")

    # Step 2: Use native DSPy
    print("4. Using native dspy.BestOfN...")
    best_of_5 = dspy.BestOfN(module=qa, N=5, reward_fn=reward_fn, threshold=0.8)

    result = best_of_5(question="What is the capital of Belgium?")
    print(f"   Answer: {result.answer}")
    print(f"   Word count: {len(result.answer.split())} words")
    print("   ✓ More flexible, but more steps\n")


def comparison_example():
    """Side-by-side comparison of both approaches."""
    print_section("Side-by-Side Comparison")

    m = start_session()
    lm = MelleaLM(mellea_session=m)
    dspy.configure(lm=lm)

    qa = dspy.ChainOfThought("question -> answer")
    requirements = ["Must be under 20 words", "Must be professional"]

    print("Question: What is machine learning?\n")

    # Approach 1: Wrapper
    print("Approach 1 (Wrapper):")
    print("```python")
    print("best_of_3 = MelleaBestOfN(")
    print("    module=qa,")
    print("    N=3,")
    print('    requirements=["Must be under 20 words", "Must be professional"],')
    print("    threshold=0.8")
    print(")")
    print("```")

    best_of_3_wrapper = MelleaBestOfN(
        module=qa, N=3, requirements=requirements, threshold=0.8
    )
    result1 = best_of_3_wrapper(question="What is machine learning?")
    print(f"Result: {result1.answer}")
    print(f"Words: {len(result1.answer.split())}\n")

    # Approach 2: Direct DSPy
    print("Approach 2 (Direct DSPy):")
    print("```python")
    print("reward_fn = create_reward_fn(")
    print('    requirements=["Must be under 20 words", "Must be professional"],')
    print('    strategy="average"')
    print(")")
    print("best_of_3 = dspy.BestOfN(")
    print("    module=qa, N=3, reward_fn=reward_fn, threshold=0.8")
    print(")")
    print("```")

    reward_fn = create_reward_fn(requirements=requirements, strategy="average")
    best_of_3_direct = dspy.BestOfN(module=qa, N=3, reward_fn=reward_fn, threshold=0.8)
    result2 = best_of_3_direct(question="What is machine learning?")
    print(f"Result: {result2.answer}")
    print(f"Words: {len(result2.answer.split())}\n")


def advanced_customization():
    """Example showing when direct DSPy approach is useful."""
    print_section("Advanced Customization with Direct DSPy")

    print("When you need more control, use create_reward_fn() + native DSPy:\n")

    m = start_session()
    lm = MelleaLM(mellea_session=m)
    dspy.configure(lm=lm)

    qa = dspy.ChainOfThought("question -> answer")

    print("1. Custom reward function with multiple strategies...")

    # Create multiple reward functions with different strategies
    reward_fn_strict = create_reward_fn(
        requirements=["Must be under 15 words", "Must mention 'algorithm'"],
        strategy="min",
    )

    reward_fn_balanced = create_reward_fn(
        requirements=["Must be under 15 words", "Must mention 'algorithm'"],
        strategy="average",
    )

    print("   ✓ Created strict (min) and balanced (average) reward functions\n")

    print("2. Using strict reward function...")
    best_of_3_strict = dspy.BestOfN(
        module=qa, N=3, reward_fn=reward_fn_strict, threshold=0.9
    )
    result_strict = best_of_3_strict(question="What is machine learning?")
    print(f"   Result: {result_strict.answer}")
    print(f"   Words: {len(result_strict.answer.split())}\n")

    print("3. Using balanced reward function...")
    best_of_3_balanced = dspy.BestOfN(
        module=qa, N=3, reward_fn=reward_fn_balanced, threshold=0.8
    )
    result_balanced = best_of_3_balanced(question="What is machine learning?")
    print(f"   Result: {result_balanced.answer}")
    print(f"   Words: {len(result_balanced.answer.split())}\n")

    print("   ✓ Direct DSPy gives you fine-grained control!")


def refine_comparison():
    """Compare wrapper vs direct approach for Refine."""
    print_section("Refine: Wrapper vs Direct DSPy")

    m = start_session()
    lm = MelleaLM(mellea_session=m)
    dspy.configure(lm=lm)

    summarizer = dspy.Predict("text -> summary")
    text = "Artificial intelligence is transforming industries worldwide."

    print("1. Using MelleaRefine wrapper...")
    refiner_wrapper = MelleaRefine(
        module=summarizer,
        N=3,
        requirements=["Must be under 10 words", "Must be concise"],
        threshold=0.9,
    )
    result1 = refiner_wrapper(text=text)
    print(f"   Summary: {result1.summary}")
    print(f"   Words: {len(result1.summary.split())}\n")

    print("2. Using dspy.Refine with create_reward_fn()...")
    reward_fn = create_reward_fn(
        requirements=["Must be under 10 words", "Must be concise"], strategy="average"
    )
    refiner_direct = dspy.Refine(
        module=summarizer, N=3, reward_fn=reward_fn, threshold=0.9
    )
    result2 = refiner_direct(text=text)
    print(f"   Summary: {result2.summary}")
    print(f"   Words: {len(result2.summary.split())}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MELLEA + DSPY: HYBRID APPROACH EXAMPLES")
    print("=" * 70)

    try:
        # Run all examples
        approach_1_wrapper_classes()
        approach_2_direct_dspy()
        comparison_example()
        advanced_customization()
        refine_comparison()

        # Summary
        print_section("Summary: When to Use Each Approach")

        print("✅ Use MelleaBestOfN/MelleaRefine (Approach 1) when:")
        print("   • You want simple, clean code")
        print("   • You're new to DSPy")
        print("   • You want Mellea-native API")
        print("   • You don't need fine-grained control\n")

        print("✅ Use create_reward_fn() + dspy.BestOfN/Refine (Approach 2) when:")
        print("   • You need advanced customization")
        print("   • You want direct access to DSPy features")
        print("   • You're building complex pipelines")
        print("   • You need multiple reward functions\n")

        print("💡 Both approaches use the same underlying conversion!")
        print("   Choose based on your needs and comfort level.\n")

        print("=" * 70)
        print("✅ All hybrid approach examples completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
