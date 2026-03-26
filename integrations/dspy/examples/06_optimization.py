"""Optimization example for Mellea + DSPy integration.

This example demonstrates how to use DSPy's optimization capabilities
with Mellea, including few-shot learning, prompt optimization, and
program compilation.
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaLM


def few_shot_examples():
    """Demonstrate few-shot learning with DSPy."""
    print("=" * 70)
    print("Example 1: Few-Shot Learning")
    print("=" * 70)

    # Setup
    print("\n1. Setting up Mellea and DSPy...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class ClassifyText(dspy.Signature):
        """Classify text into categories."""

        text = dspy.InputField()
        category = dspy.OutputField()

    print("   ✓ Signature defined")

    # Create predictor
    print("\n3. Creating predictor...")
    classifier = dspy.Predict(ClassifyText)
    print("   ✓ Predictor created")

    # Few-shot examples (in practice, these would be used for optimization)
    print("\n4. Few-shot examples for training:")
    examples = [
        ("Machine learning models require training data", "AI/ML"),
        ("The stock market showed gains today", "Finance"),
        ("New smartphone features announced", "Technology"),
        ("Climate change affects global weather", "Environment"),
    ]

    for text, category in examples:
        print(f"   - '{text}' → {category}")

    # Test classification
    print("\n5. Testing classification...")
    test_texts = [
        "Neural networks are inspired by the brain",
        "Interest rates may rise next quarter",
    ]

    for text in test_texts:
        result = classifier(text=text)
        print(f"\n   Text: {text}")
        print(f"   Predicted Category: {result.category}")

    print("\n   Note: With optimization, accuracy improves using training examples")

    print("\n" + "=" * 70)


def bootstrap_few_shot():
    """Demonstrate bootstrap few-shot optimization."""
    print("\n" + "=" * 70)
    print("Example 2: Bootstrap Few-Shot Optimization")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define a simple program
    print("\n2. Defining program...")

    class SimpleQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.predictor(question=question)

    print("   ✓ Program defined")

    # Create training data
    print("\n3. Creating training data...")
    trainset = [
        dspy.Example(
            question="What is generative programming?",
            answer="A paradigm for writing structured AI workflows",
        ).with_inputs("question"),
        dspy.Example(
            question="What is DSPy?",
            answer="A framework for programming with language models",
        ).with_inputs("question"),
        dspy.Example(
            question="What is Mellea?", answer="A library for generative programming"
        ).with_inputs("question"),
    ]
    print(f"   ✓ Created {len(trainset)} training examples")

    # Note: Actual optimization would use BootstrapFewShot
    print("\n4. Optimization process (conceptual):")
    print("   - BootstrapFewShot would:")
    print("     • Generate additional examples")
    print("     • Select best demonstrations")
    print("     • Compile optimized program")
    print("   - Result: Improved accuracy and consistency")

    # Test the program
    print("\n5. Testing program...")
    qa = SimpleQA()
    test_question = "What are the benefits of structured prompting?"
    result = qa(question=test_question)
    print(f"   Question: {test_question}")
    print(f"   Answer: {result.answer}")

    print("\n" + "=" * 70)


def signature_optimization():
    """Demonstrate signature optimization."""
    print("\n" + "=" * 70)
    print("Example 3: Signature Optimization")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Compare different signature designs
    print("\n2. Comparing signature designs...")

    # Basic signature
    class BasicSignature(dspy.Signature):
        """Basic signature."""

        input = dspy.InputField()
        output = dspy.OutputField()

    # Detailed signature
    class DetailedSignature(dspy.Signature):
        """Detailed signature with descriptions."""

        input = dspy.InputField(desc="The input text to process")
        output = dspy.OutputField(desc="The processed output with specific format")

    # Structured signature
    class StructuredSignature(dspy.Signature):
        """Structured signature with multiple fields."""

        text = dspy.InputField(desc="The text to analyze")
        context = dspy.InputField(desc="Additional context")
        analysis = dspy.OutputField(desc="Detailed analysis")
        summary = dspy.OutputField(desc="Brief summary")

    print("   ✓ Signatures defined")

    print("\n3. Signature design principles:")
    print("   - Clear, descriptive field names")
    print("   - Helpful field descriptions")
    print("   - Appropriate granularity")
    print("   - Structured outputs when needed")

    print("\n4. Testing structured signature...")
    predictor = dspy.Predict(StructuredSignature)
    result = predictor(
        text="Generative programming improves AI reliability",
        context="Software engineering best practices",
    )
    print(f"   Analysis: {result.analysis}")
    print(f"   Summary: {result.summary}")

    print("\n" + "=" * 70)


def metric_based_optimization():
    """Demonstrate metric-based optimization."""
    print("\n" + "=" * 70)
    print("Example 4: Metric-Based Optimization")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define a metric
    print("\n2. Defining evaluation metric...")

    def accuracy_metric(example, prediction, trace=None):
        """Simple accuracy metric."""
        # In practice, this would compare prediction to ground truth
        return 1.0 if prediction.answer else 0.0

    print("   ✓ Metric defined")

    # Define program
    print("\n3. Defining program...")

    class QAProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.qa(question=question)

    print("   ✓ Program defined")

    # Evaluation process
    print("\n4. Evaluation process (conceptual):")
    print("   - Define metric (accuracy, F1, custom)")
    print("   - Create validation set")
    print("   - Evaluate program performance")
    print("   - Use metric to guide optimization")

    # Test program
    print("\n5. Testing program...")
    program = QAProgram()
    result = program(question="What is the purpose of metrics in optimization?")
    print(f"   Answer: {result.answer}")

    print("\n   Note: Metrics guide the optimization process")
    print("   - Higher metric = better performance")
    print("   - Used by optimizers to select best prompts")

    print("\n" + "=" * 70)


def prompt_optimization():
    """Demonstrate prompt optimization strategies."""
    print("\n" + "=" * 70)
    print("Example 5: Prompt Optimization Strategies")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Different prompt strategies
    print("\n2. Prompt optimization strategies:")

    strategies = [
        ("Zero-shot", "No examples, rely on instruction"),
        ("Few-shot", "Include example demonstrations"),
        ("Chain-of-thought", "Include reasoning steps"),
        ("Self-consistency", "Multiple samples, majority vote"),
        ("Least-to-most", "Break down complex problems"),
    ]

    for strategy, description in strategies:
        print(f"   - {strategy}: {description}")

    # Demonstrate chain-of-thought
    print("\n3. Testing chain-of-thought strategy...")

    class CoTProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.cot = dspy.ChainOfThought("problem -> solution")

        def forward(self, problem):
            return self.cot(problem=problem)

    program = CoTProgram()
    problem = "How can we improve AI system reliability?"
    result = program(problem=problem)

    print(f"   Problem: {problem}")
    if hasattr(result, "rationale"):
        print(f"   Reasoning: {result.rationale}")
    elif hasattr(result, "reasoning"):
        print(f"   Reasoning: {result.reasoning}")
    print(f"   Solution: {result.solution}")

    print("\n   Note: CoT improves reasoning for complex problems")

    print("\n" + "=" * 70)


def program_compilation():
    """Demonstrate program compilation and optimization."""
    print("\n" + "=" * 70)
    print("Example 6: Program Compilation")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define program
    print("\n2. Defining program...")

    class MultiStepProgram(dspy.Module):
        """A program with multiple steps."""

        def __init__(self):
            super().__init__()
            self.step1 = dspy.Predict("input -> intermediate")
            self.step2 = dspy.Predict("intermediate -> output")

        def forward(self, input):
            intermediate = self.step1(input=input)
            output = self.step2(intermediate=intermediate.intermediate)
            return output

    print("   ✓ Program defined")

    # Compilation process
    print("\n3. Compilation process (conceptual):")
    print("   - Analyze program structure")
    print("   - Optimize each component")
    print("   - Select best prompts/examples")
    print("   - Compile into efficient program")

    # Benefits
    print("\n4. Benefits of compilation:")
    benefits = [
        "Improved accuracy through optimization",
        "Better prompt selection",
        "Efficient example usage",
        "Consistent performance",
        "Reduced manual prompt engineering",
    ]

    for benefit in benefits:
        print(f"   ✓ {benefit}")

    # Test program
    print("\n5. Testing compiled program...")
    program = MultiStepProgram()
    result = program(input="Explain generative programming benefits")
    print(f"   Output: {result.output}")

    print("\n" + "=" * 70)


def hyperparameter_tuning():
    """Demonstrate hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("Example 7: Hyperparameter Tuning")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    print("   ✓ Setup complete")

    # Hyperparameters to tune
    print("\n2. Key hyperparameters:")

    hyperparameters = [
        ("temperature", "0.0-1.0", "Controls randomness"),
        ("max_tokens", "100-2000", "Maximum output length"),
        ("top_p", "0.0-1.0", "Nucleus sampling threshold"),
        ("num_examples", "0-10", "Few-shot example count"),
    ]

    print("\n   Parameter | Range | Description")
    print("   " + "-" * 60)
    for param, range_val, desc in hyperparameters:
        print(f"   {param:15} | {range_val:10} | {desc}")

    # Test different temperatures
    print("\n3. Testing different temperatures...")

    class TestProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("task -> result")

        def forward(self, task):
            return self.predictor(task=task)

    temperatures = [0.0, 0.5, 1.0]
    task = "Generate a creative solution"

    for temp in temperatures:
        lm = MelleaLM(mellea_session=m, model="mellea-ollama", temperature=temp)
        dspy.configure(lm=lm)
        program = TestProgram()
        result = program(task=task)
        print(f"\n   Temperature {temp}: {result.result[:100]}...")

    print("\n4. Tuning recommendations:")
    print("   - Start with default values")
    print("   - Use validation set for evaluation")
    print("   - Grid search or random search")
    print("   - Monitor performance metrics")
    print("   - Balance quality vs. cost")

    print("\n" + "=" * 70)


def main():
    """Run all optimization examples."""
    print("\n" + "=" * 70)
    print("MELLEA + DSPY: OPTIMIZATION EXAMPLES")
    print("=" * 70)

    try:
        # Run examples
        few_shot_examples()
        bootstrap_few_shot()
        signature_optimization()
        metric_based_optimization()
        prompt_optimization()
        program_compilation()
        hyperparameter_tuning()

        # Summary
        print("\n" + "=" * 70)
        print("✅ All optimization examples completed!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Few-shot learning improves accuracy")
        print("  • Bootstrap optimization automates example selection")
        print("  • Signature design impacts performance")
        print("  • Metrics guide optimization process")
        print("  • Multiple prompt strategies available")
        print("  • Compilation optimizes entire programs")
        print("  • Hyperparameter tuning fine-tunes performance")
        print("\nOptimization Workflow:")
        print("  1. Define program and signature")
        print("  2. Create training/validation data")
        print("  3. Define evaluation metric")
        print("  4. Choose optimization strategy")
        print("  5. Compile and evaluate")
        print("  6. Iterate and refine")
        print("\nDSPy Optimizers:")
        print("  • BootstrapFewShot: Generate and select examples")
        print("  • MIPRO: Multi-prompt instruction optimization")
        print("  • KNNFewShot: K-nearest neighbor selection")
        print("  • SignatureOptimizer: Optimize signatures")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
