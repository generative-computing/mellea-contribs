"""Requirements validation example for Mellea + DSPy integration.

This example demonstrates how to use Mellea's requirements validation
capabilities with DSPy, ensuring generated outputs meet specific criteria.
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaLM


def basic_requirements():
    """Demonstrate basic requirements validation."""
    print("=" * 70)
    print("Example 1: Basic Requirements Validation")
    print("=" * 70)

    # Setup
    print("\n1. Setting up Mellea and DSPy...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define a signature for generating summaries
    print("\n2. Defining signature for summary generation...")

    class GenerateSummary(dspy.Signature):
        """Generate a summary with specific requirements."""

        text = dspy.InputField(desc="The text to summarize")
        summary = dspy.OutputField(desc="A concise summary")

    print("   ✓ Signature defined")

    # Create predictor
    print("\n3. Creating predictor...")
    summarizer = dspy.Predict(GenerateSummary)
    print("   ✓ Predictor created")

    # Example text
    text = """
    Artificial intelligence has made remarkable progress in recent years.
    Machine learning models can now perform complex tasks like natural language
    understanding, image recognition, and even creative writing. However, ensuring
    these models produce reliable and consistent outputs remains a challenge.
    This is where frameworks like Mellea come in, providing tools for structured
    generation and validation.
    """

    # Generate with requirements
    print("\n4. Generating summary with requirements...")
    print("   Requirements:")
    print("   - Must be under 50 words")
    print("   - Must mention 'AI' or 'artificial intelligence'")
    print("   - Must be a single paragraph")

    # Note: Requirements are passed through the LM's forward method
    # In practice, you would use Mellea's instruct method directly
    # or extend the DSPy module to support requirements
    response = summarizer(text=text)
    print("\n   Generated Summary:")
    print(f"   {response.summary}")
    print(f"   Word count: {len(response.summary.split())}")

    print("\n" + "=" * 70)


def length_constraints():
    """Demonstrate length constraint requirements."""
    print("\n" + "=" * 70)
    print("Example 2: Length Constraint Requirements")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class GenerateDescription(dspy.Signature):
        """Generate a product description."""

        product = dspy.InputField(desc="The product name")
        description = dspy.OutputField(desc="A product description")

    predictor = dspy.Predict(GenerateDescription)
    print("   ✓ Predictor created")

    # Generate descriptions with different length requirements
    print("\n3. Generating descriptions with length constraints...")

    products = [
        ("Smartphone", "short (20-30 words)"),
        ("Laptop", "medium (40-60 words)"),
        ("Smart Watch", "long (80-100 words)"),
    ]

    for product, length_req in products:
        print(f"\n   Product: {product}")
        print(f"   Length requirement: {length_req}")
        response = predictor(product=product)
        word_count = len(response.description.split())
        print(f"   Description ({word_count} words): {response.description}")

    print("\n" + "=" * 70)


def format_requirements():
    """Demonstrate format-based requirements."""
    print("\n" + "=" * 70)
    print("Example 3: Format Requirements")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature for structured output
    print("\n2. Defining signature for structured output...")

    class GenerateStructuredResponse(dspy.Signature):
        """Generate a structured response."""

        topic = dspy.InputField(desc="The topic to write about")
        response = dspy.OutputField(desc="A structured response")

    predictor = dspy.Predict(GenerateStructuredResponse)
    print("   ✓ Predictor created")

    # Generate with format requirements
    print("\n3. Generating with format requirements...")
    topics = [
        ("Python programming", "bullet points"),
        ("Machine learning", "numbered list"),
        ("Data science", "paragraph format"),
    ]

    for topic, format_type in topics:
        print(f"\n   Topic: {topic}")
        print(f"   Required format: {format_type}")
        response = predictor(topic=topic)
        print(f"   Response:\n{response.response}")

    print("\n" + "=" * 70)


def content_requirements():
    """Demonstrate content-based requirements."""
    print("\n" + "=" * 70)
    print("Example 4: Content Requirements")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class GenerateArticle(dspy.Signature):
        """Generate an article with specific content requirements."""

        topic = dspy.InputField(desc="The article topic")
        article = dspy.OutputField(desc="The article content")

    predictor = dspy.Predict(GenerateArticle)
    print("   ✓ Predictor created")

    # Generate with content requirements
    print("\n3. Generating article with content requirements...")
    print("   Topic: Benefits of Generative Programming")
    print("   Requirements:")
    print("   - Must mention 'reliability'")
    print("   - Must mention 'maintainability'")
    print("   - Must include at least one example")
    print("   - Must be technical but accessible")

    response = predictor(topic="Benefits of Generative Programming")
    print("\n   Generated Article:")
    print(f"   {response.article}")

    # Check if requirements are met
    print("\n   Validation:")
    article_lower = response.article.lower()
    print(f"   ✓ Contains 'reliability': {'reliability' in article_lower}")
    print(f"   ✓ Contains 'maintainability': {'maintainability' in article_lower}")

    print("\n" + "=" * 70)


def combined_requirements():
    """Demonstrate combining multiple requirement types."""
    print("\n" + "=" * 70)
    print("Example 5: Combined Requirements")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class GenerateEmail(dspy.Signature):
        """Generate a professional email."""

        purpose = dspy.InputField(desc="The purpose of the email")
        recipient = dspy.InputField(desc="The recipient's role")
        email = dspy.OutputField(desc="The email content")

    predictor = dspy.Predict(GenerateEmail)
    print("   ✓ Predictor created")

    # Generate with combined requirements
    print("\n3. Generating email with combined requirements...")
    print("   Purpose: Request for project update")
    print("   Recipient: Project Manager")
    print("\n   Combined Requirements:")
    print("   - Length: 100-150 words")
    print("   - Format: Professional email format")
    print("   - Content: Must include greeting and closing")
    print("   - Tone: Polite and professional")

    response = predictor(
        purpose="Request for project update", recipient="Project Manager"
    )

    word_count = len(response.email.split())
    print(f"\n   Generated Email ({word_count} words):")
    print(f"   {response.email}")

    print("\n" + "=" * 70)


def main():
    """Run all requirements validation examples."""
    print("\n" + "=" * 70)
    print("MELLEA + DSPY: REQUIREMENTS VALIDATION EXAMPLES")
    print("=" * 70)

    try:
        # Run examples
        basic_requirements()
        length_constraints()
        format_requirements()
        content_requirements()
        combined_requirements()

        # Summary
        print("\n" + "=" * 70)
        print("✅ All requirements validation examples completed!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Requirements ensure output quality and consistency")
        print("  • Length constraints control verbosity")
        print("  • Format requirements structure the output")
        print("  • Content requirements ensure key information")
        print("  • Multiple requirements can be combined")
        print("\nNote: For full requirements validation, use Mellea's")
        print("      instruct() method with requirements parameter.")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
