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

    # Example text
    text = """
    Artificial intelligence has made remarkable progress in recent years.
    Machine learning models can now perform complex tasks like natural language
    understanding, image recognition, and even creative writing. However, ensuring
    these models produce reliable and consistent outputs remains a challenge.
    This is where frameworks like Mellea come in, providing tools for structured
    generation and validation.
    """

    # Generate with requirements using forward method
    print("\n2. Generating summary with requirements...")
    print("   Requirements:")
    print("   - Must be concise")
    print("   - Must mention 'AI' or 'artificial intelligence'")
    print("   - Must be a single paragraph")

    requirements = [
        "be concise and under 50 words",
        "mention 'AI' or 'artificial intelligence'",
        "write as a single paragraph",
    ]

    response = lm.forward(
        prompt=f"Summarize the following text:\n\n{text}", requirements=requirements
    )

    summary = response.choices[0].message.content
    print("\n   Generated Summary:")
    print(f"   {summary}")
    print(f"   Word count: {len(summary.split())}")

    # Validate requirements were met
    print("\n   Validation:")
    print(f"   ✓ Word count under 50: {len(summary.split()) <= 50}")
    print(
        f"   ✓ Mentions AI: {'ai' in summary.lower() or 'artificial intelligence' in summary.lower()}"
    )

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

    # Generate descriptions with different length requirements
    print("\n2. Generating descriptions with length constraints...")

    products = [
        (
            "Smartphone",
            "20-30 words",
            ["keep response between 20-30 words", "be descriptive"],
        ),
        (
            "Laptop",
            "40-60 words",
            ["keep response between 40-60 words", "mention key features"],
        ),
        (
            "Smart Watch",
            "80-100 words",
            ["keep response between 80-100 words", "be comprehensive"],
        ),
    ]

    for product, length_desc, requirements in products:
        print(f"\n   Product: {product}")
        print(f"   Length requirement: {length_desc}")

        response = lm.forward(
            prompt=f"Write a product description for: {product}",
            requirements=requirements,
        )

        description = response.choices[0].message.content
        word_count = len(description.split())
        print(f"   Description ({word_count} words): {description}")

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

    # Generate with format requirements
    print("\n2. Generating with format requirements...")
    topics = [
        (
            "Python programming",
            "bullet points",
            ["use bullet points", "list 3-5 key points"],
        ),
        ("Machine learning", "numbered list", ["use numbered list", "include 4 items"]),
        (
            "Data science",
            "paragraph format",
            ["write as a single paragraph", "be concise"],
        ),
    ]

    for topic, format_desc, requirements in topics:
        print(f"\n   Topic: {topic}")
        print(f"   Required format: {format_desc}")

        response = lm.forward(prompt=f"Write about {topic}", requirements=requirements)

        content = response.choices[0].message.content
        print(f"   Response:\n{content}")

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

    # Generate with content requirements
    print("\n2. Generating article with content requirements...")
    print("   Topic: Benefits of Generative Programming")
    print("   Requirements:")
    print("   - Must mention 'reliability'")
    print("   - Must mention 'maintainability'")
    print("   - Must include at least one example")
    print("   - Must be technical but accessible")

    requirements = [
        "mention 'reliability' in the response",
        "mention 'maintainability' in the response",
        "include at least one concrete example",
        "be technical but accessible to developers",
    ]

    response = lm.forward(
        prompt="Write an article about the benefits of generative programming",
        requirements=requirements,
    )

    article = response.choices[0].message.content
    print("\n   Generated Article:")
    print(f"   {article}")

    # Check if requirements are met
    print("\n   Validation:")
    article_lower = article.lower()
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

    # Generate with combined requirements
    print("\n2. Generating email with combined requirements...")
    print("   Purpose: Request for project update")
    print("   Recipient: Project Manager")
    print("\n   Combined Requirements:")
    print("   - Length: 100-150 words")
    print("   - Format: Professional email format")
    print("   - Content: Must include greeting and closing")
    print("   - Tone: Polite and professional")

    requirements = [
        "keep response between 100-150 words",
        "use professional email format with greeting and closing",
        "include a polite greeting",
        "include a professional closing",
        "maintain a polite and professional tone throughout",
    ]

    response = lm.forward(
        prompt="Write a professional email to a Project Manager requesting a project update",
        requirements=requirements,
    )

    email = response.choices[0].message.content
    word_count = len(email.split())
    print(f"\n   Generated Email ({word_count} words):")
    print(f"   {email}")

    print("\n" + "=" * 70)


def requirements_with_lm_instance():
    """Demonstrate setting requirements on LM instance."""
    print("\n" + "=" * 70)
    print("Example 6: Requirements on LM Instance")
    print("=" * 70)

    # Setup with requirements on LM
    print("\n1. Setting up LM with default requirements...")
    m = start_session()

    # Set requirements at LM level - these apply to all generations
    default_requirements = ["be concise", "use clear language", "be helpful"]

    lm = MelleaLM(
        mellea_session=m, model="mellea-ollama", requirements=default_requirements
    )
    dspy.configure(lm=lm)
    print("   ✓ LM configured with default requirements")

    # Use with DSPy - requirements automatically applied
    print("\n2. Using DSPy with LM-level requirements...")
    qa = dspy.Predict("question -> answer")

    questions = ["What is Python?", "Explain machine learning"]

    for question in questions:
        print(f"\n   Question: {question}")
        response = qa(question=question)
        print(f"   Answer: {response.answer}")
        print(f"   (Requirements automatically applied: {default_requirements})")

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
        requirements_with_lm_instance()

        # Summary
        print("\n" + "=" * 70)
        print("✅ All requirements validation examples completed!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Requirements ensure output quality and consistency")
        print("  • Pass requirements via forward() method for per-call control")
        print("  • Set requirements on LM instance for default behavior")
        print("  • Length constraints control verbosity")
        print("  • Format requirements structure the output")
        print("  • Content requirements ensure key information")
        print("  • Multiple requirements can be combined")
        print("\nUsage Patterns:")
        print("  1. Per-call: lm.forward(prompt=..., requirements=[...])")
        print("  2. LM-level: MelleaLM(mellea_session=m, requirements=[...])")
        print("  3. Override: Call-level requirements override LM-level")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
