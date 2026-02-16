"""DSPy modules example for Mellea + DSPy integration.

This example demonstrates how to build custom DSPy modules and programs
using Mellea as the backend, including composition and reusable components.
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaLM


def simple_module():
    """Demonstrate creating a simple DSPy module."""
    print("=" * 70)
    print("Example 1: Simple DSPy Module")
    print("=" * 70)

    # Setup
    print("\n1. Setting up Mellea and DSPy...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define a simple module
    print("\n2. Defining a simple module...")

    class SimpleQA(dspy.Module):
        """A simple question-answering module."""

        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("question -> answer")

        def forward(self, question):
            """Process a question and return an answer."""
            return self.predictor(question=question)

    print("   ✓ Module defined")

    # Use the module
    print("\n3. Using the module...")
    qa = SimpleQA()

    questions = ["What is DSPy?", "How does Mellea integrate with DSPy?"]

    for question in questions:
        print(f"\n   Q: {question}")
        result = qa(question=question)
        print(f"   A: {result.answer}")

    print("\n" + "=" * 70)


def multi_step_module():
    """Demonstrate a multi-step reasoning module."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-Step Reasoning Module")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define multi-step module
    print("\n2. Defining multi-step module...")

    class MultiStepReasoning(dspy.Module):
        """A module that performs multi-step reasoning."""

        def __init__(self):
            super().__init__()
            self.analyze = dspy.ChainOfThought("question -> analysis")
            self.synthesize = dspy.Predict("analysis -> conclusion")

        def forward(self, question):
            """Perform multi-step reasoning."""
            # Step 1: Analyze the question
            analysis_result = self.analyze(question=question)

            # Step 2: Synthesize conclusion
            conclusion_result = self.synthesize(analysis=analysis_result.analysis)

            # Build result with available attributes
            result_dict = {
                "analysis": analysis_result.analysis,
                "conclusion": conclusion_result.conclusion,
            }

            # Add rationale if available
            if hasattr(analysis_result, "rationale"):
                result_dict["rationale"] = analysis_result.rationale
            elif hasattr(analysis_result, "reasoning"):
                result_dict["rationale"] = analysis_result.reasoning

            return dspy.Prediction(**result_dict)

    print("   ✓ Module defined")

    # Use the module
    print("\n3. Using multi-step reasoning...")
    reasoner = MultiStepReasoning()

    question = (
        "What are the advantages of using structured prompting over ad-hoc prompts?"
    )
    print(f"\n   Question: {question}")

    result = reasoner(question=question)
    print(f"\n   Analysis: {result.analysis}")
    if hasattr(result, "rationale"):
        print(f"   Rationale: {result.rationale}")
    print(f"   Conclusion: {result.conclusion}")

    print("\n" + "=" * 70)


def composable_modules():
    """Demonstrate composing multiple modules."""
    print("\n" + "=" * 70)
    print("Example 3: Composable Modules")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define component modules
    print("\n2. Defining component modules...")

    class Extractor(dspy.Module):
        """Extract key information from text."""

        def __init__(self):
            super().__init__()
            self.extract = dspy.Predict("text -> key_points")

        def forward(self, text):
            return self.extract(text=text)

    class Summarizer(dspy.Module):
        """Summarize key points."""

        def __init__(self):
            super().__init__()
            self.summarize = dspy.Predict("key_points -> summary")

        def forward(self, key_points):
            return self.summarize(key_points=key_points)

    class Pipeline(dspy.Module):
        """Compose extractor and summarizer."""

        def __init__(self):
            super().__init__()
            self.extractor = Extractor()
            self.summarizer = Summarizer()

        def forward(self, text):
            # Extract key points
            extraction = self.extractor(text=text)

            # Summarize key points
            summary = self.summarizer(key_points=extraction.key_points)

            return dspy.Prediction(
                key_points=extraction.key_points, summary=summary.summary
            )

    print("   ✓ Modules defined")

    # Use the pipeline
    print("\n3. Using the composed pipeline...")
    pipeline = Pipeline()

    text = """
    Generative programming is a paradigm that replaces brittle prompts with
    structured, maintainable code. It provides better reliability through
    requirements validation, improved efficiency through sampling strategies,
    and enhanced maintainability through modular design. The approach enables
    inference-time scaling and supports multiple model backends.
    """

    print(f"\n   Input text: {text.strip()[:100]}...")
    result = pipeline(text=text)
    print(f"\n   Key Points: {result.key_points}")
    print(f"   Summary: {result.summary}")

    print("\n" + "=" * 70)


def conditional_module():
    """Demonstrate a module with conditional logic."""
    print("\n" + "=" * 70)
    print("Example 4: Conditional Module")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define conditional module
    print("\n2. Defining conditional module...")

    class AdaptiveQA(dspy.Module):
        """QA module that adapts based on question complexity."""

        def __init__(self):
            super().__init__()
            self.classifier = dspy.Predict("question -> complexity")
            self.simple_qa = dspy.Predict("question -> answer")
            self.complex_qa = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            """Answer question with appropriate strategy."""
            # Classify question complexity
            classification = self.classifier(question=question)
            complexity = classification.complexity.lower()

            # Choose appropriate strategy
            if "simple" in complexity or "easy" in complexity:
                result = self.simple_qa(question=question)
                strategy = "simple"
            else:
                result = self.complex_qa(question=question)
                strategy = "complex (chain of thought)"

            return dspy.Prediction(
                answer=result.answer, strategy=strategy, complexity=complexity
            )

    print("   ✓ Module defined")

    # Use the module
    print("\n3. Using adaptive QA...")
    qa = AdaptiveQA()

    questions = [
        "What is 2+2?",
        "Explain the relationship between generative programming and software reliability",
    ]

    for question in questions:
        print(f"\n   Question: {question}")
        result = qa(question=question)
        print(f"   Complexity: {result.complexity}")
        print(f"   Strategy: {result.strategy}")
        print(f"   Answer: {result.answer}")

    print("\n" + "=" * 70)


def stateful_module():
    """Demonstrate a stateful module with memory."""
    print("\n" + "=" * 70)
    print("Example 5: Stateful Module")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define stateful module
    print("\n2. Defining stateful module...")

    class ConversationalQA(dspy.Module):
        """QA module that maintains conversation history."""

        def __init__(self):
            super().__init__()
            self.qa = dspy.Predict("context, question -> answer")
            self.context_str = ""  # Store as simple string

        def forward(self, question):
            """Answer question with conversation context."""
            # Get answer with current context
            result = self.qa(
                context=self.context_str if self.context_str else "No previous context",
                question=question,
            )

            # Update context string
            self.context_str += f"Q: {question}\nA: {result.answer}\n"

            return result

        def reset(self):
            """Reset conversation history."""
            self.context_str = ""

    print("   ✓ Module defined")

    # Use the module
    print("\n3. Using conversational QA...")
    conv_qa = ConversationalQA()

    conversation = [
        "What is Mellea?",
        "What are its main features?",
        "How does it compare to other frameworks?",
    ]

    for i, question in enumerate(conversation, 1):
        print(f"\n   Turn {i}")
        print(f"   Q: {question}")
        result = conv_qa(question=question)
        print(f"   A: {result.answer}")

    print("\n   Note: Module maintains context across turns")

    print("\n" + "=" * 70)


def reusable_module():
    """Demonstrate creating reusable module components."""
    print("\n" + "=" * 70)
    print("Example 6: Reusable Module Components")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define reusable components
    print("\n2. Defining reusable components...")

    class Validator(dspy.Module):
        """Reusable validation component."""

        def __init__(self):
            super().__init__()
            self.validate = dspy.Predict("text -> is_valid, reason")

        def forward(self, text):
            return self.validate(text=text)

    class Formatter(dspy.Module):
        """Reusable formatting component."""

        def __init__(self):
            super().__init__()
            self.format = dspy.Predict("text, style -> formatted_text")

        def forward(self, text, style="professional"):
            return self.format(text=text, style=style)

    class ValidatedFormatter(dspy.Module):
        """Compose validator and formatter."""

        def __init__(self):
            super().__init__()
            self.validator = Validator()
            self.formatter = Formatter()

        def forward(self, text, style="professional"):
            # Validate first
            validation = self.validator(text=text)

            if "valid" in validation.is_valid.lower():
                # Format if valid
                formatted = self.formatter(text=text, style=style)
                return dspy.Prediction(
                    formatted_text=formatted.formatted_text,
                    is_valid=True,
                    reason=validation.reason,
                )
            else:
                return dspy.Prediction(
                    formatted_text=text, is_valid=False, reason=validation.reason
                )

    print("   ✓ Components defined")

    # Use the composed module
    print("\n3. Using validated formatter...")
    formatter = ValidatedFormatter()

    texts = [
        ("This is a valid professional text", "professional"),
        ("casual text here", "formal"),
    ]

    for text, style in texts:
        print(f"\n   Input: {text}")
        print(f"   Style: {style}")
        result = formatter(text=text, style=style)
        print(f"   Valid: {result.is_valid}")
        print(f"   Output: {result.formatted_text}")

    print("\n" + "=" * 70)


def main():
    """Run all DSPy module examples."""
    print("\n" + "=" * 70)
    print("MELLEA + DSPY: MODULES AND COMPOSITION EXAMPLES")
    print("=" * 70)

    try:
        # Run examples
        simple_module()
        multi_step_module()
        composable_modules()
        conditional_module()
        stateful_module()
        reusable_module()

        # Summary
        print("\n" + "=" * 70)
        print("✅ All DSPy module examples completed!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Modules encapsulate reusable logic")
        print("  • Multi-step modules enable complex workflows")
        print("  • Composition creates powerful pipelines")
        print("  • Conditional logic adapts to inputs")
        print("  • Stateful modules maintain context")
        print("  • Reusable components promote modularity")
        print("\nBest Practices:")
        print("  • Keep modules focused and single-purpose")
        print("  • Use composition over inheritance")
        print("  • Document module inputs and outputs")
        print("  • Test modules independently")
        print("  • Design for reusability")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
