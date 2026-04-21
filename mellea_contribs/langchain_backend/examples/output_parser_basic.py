"""Basic example demonstrating MelleaOutputParser usage.

This example shows how to use the MelleaOutputParser with simple validation
requirements in a LangChain chain.
"""

from langchain_core.prompts import ChatPromptTemplate
from mellea import start_session

from mellea_langchain import MelleaChatModel, MelleaOutputParser


def simple_validate(fn, description):
    """Create a simple validator function with description.

    This is a helper that mimics mellea.stdlib.requirements.simple_validate.
    In production, use the actual mellea function.
    """

    def validator(text):
        return fn(text)

    validator.description = description
    validator.__doc__ = description
    return validator


def example_basic_parser():
    """Example 1: Basic output parser with simple requirements."""
    print("=" * 70)
    print("Example 1: Basic Output Parser")
    print("=" * 70)

    # Create Mellea session and chat model
    m = start_session()
    model = MelleaChatModel(mellea_session=m)

    # Create parser with validation requirements
    parser = MelleaOutputParser(
        requirements=[
            simple_validate(lambda x: len(x.split()) < 100, "Must be under 100 words"),
            simple_validate(lambda x: x.strip() == x, "Must not have leading/trailing whitespace"),
            simple_validate(
                lambda x: "AI" in x or "artificial intelligence" in x.lower(),
                "Must mention AI or artificial intelligence",
            ),
        ]
    )

    # Create a simple chain
    prompt = ChatPromptTemplate.from_template(
        "Write a brief summary about {topic} in 2-3 sentences."
    )

    chain = prompt | model | parser

    # Test with valid input
    print("\n1. Testing with valid input...")
    try:
        result = chain.invoke({"topic": "artificial intelligence"})
        print(f"✓ Success! Output:\n{result}\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")


def example_non_strict_mode():
    """Example 2: Non-strict mode returns text even on validation failure."""
    print("=" * 70)
    print("Example 2: Non-Strict Mode")
    print("=" * 70)

    m = start_session()
    model = MelleaChatModel(mellea_session=m)

    # Create parser in non-strict mode
    parser = MelleaOutputParser(
        requirements=[
            simple_validate(
                lambda x: len(x) < 50,  # Very strict limit
                "Must be under 50 characters",
            ),
        ],
        strict=False,  # Won't raise exception on failure
    )

    prompt = ChatPromptTemplate.from_template("Write a detailed explanation about {topic}.")

    chain = prompt | model | parser

    print("\n1. Testing with likely-to-fail requirement...")
    print("   (Non-strict mode will return text anyway)")
    try:
        result = chain.invoke({"topic": "machine learning"})
        print(f"✓ Returned text (may not meet requirements):\n{result}\n")
    except Exception as e:
        print(f"✗ Unexpected error: {e}\n")


def example_format_instructions():
    """Example 3: Using format instructions in prompts."""
    print("=" * 70)
    print("Example 3: Format Instructions")
    print("=" * 70)

    m = start_session()
    model = MelleaChatModel(mellea_session=m)

    # Create parser
    parser = MelleaOutputParser(
        requirements=[
            simple_validate(lambda x: len(x.split()) < 50, "Must be under 50 words"),
            simple_validate(lambda x: x[0].isupper(), "Must start with capital letter"),
            simple_validate(lambda x: x.endswith("."), "Must end with period"),
        ]
    )

    # Get format instructions
    instructions = parser.get_format_instructions()
    print("\n1. Format instructions:")
    print(instructions)

    # Use instructions in prompt
    prompt = ChatPromptTemplate.from_template("Write about {topic}.\n\n{format_instructions}")

    chain = prompt | model | parser

    print("\n2. Testing with format instructions in prompt...")
    try:
        result = chain.invoke({"topic": "quantum computing", "format_instructions": instructions})
        print(f"✓ Success! Output:\n{result}\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")


def example_custom_validators():
    """Example 4: Custom validation functions."""
    print("=" * 70)
    print("Example 4: Custom Validators")
    print("=" * 70)

    m = start_session()
    model = MelleaChatModel(mellea_session=m)

    # Define custom validation functions
    def has_bullet_points(text):
        """Check if text contains bullet points."""
        return "•" in text or "-" in text or "*" in text

    def has_multiple_sentences(text):
        """Check if text has multiple sentences."""
        return text.count(".") >= 2

    def word_count_in_range(text):
        """Check if word count is between 30 and 100."""
        word_count = len(text.split())
        return 30 <= word_count <= 100

    # Create parser with custom validators
    parser = MelleaOutputParser(
        requirements=[
            simple_validate(has_bullet_points, "Must contain bullet points"),
            simple_validate(has_multiple_sentences, "Must have at least 2 sentences"),
            simple_validate(word_count_in_range, "Must be between 30-100 words"),
        ]
    )

    prompt = ChatPromptTemplate.from_template(
        "Write a brief overview of {topic} with bullet points."
    )

    chain = prompt | model | parser

    print("\n1. Testing with custom validators...")
    try:
        result = chain.invoke({"topic": "renewable energy"})
        print(f"✓ Success! Output:\n{result}\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")


def example_error_handling():
    """Example 5: Handling validation errors."""
    print("=" * 70)
    print("Example 5: Error Handling")
    print("=" * 70)

    m = start_session()
    model = MelleaChatModel(mellea_session=m)

    # Create parser with strict requirements
    parser = MelleaOutputParser(
        requirements=[
            simple_validate(
                lambda x: len(x) < 20,  # Very strict
                "Must be under 20 characters",
            ),
        ],
        strict=True,
    )

    prompt = ChatPromptTemplate.from_template("Explain {topic}")
    chain = prompt | model | parser

    print("\n1. Testing with strict requirement (likely to fail)...")
    try:
        result = chain.invoke({"topic": "blockchain technology"})
        print(f"✓ Unexpected success: {result}\n")
    except Exception as e:
        print("✗ Expected failure caught:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)[:200]}...\n")

        # In production, you might retry with relaxed requirements
        print("2. Retrying with relaxed requirements...")
        relaxed_parser = MelleaOutputParser(
            requirements=[
                simple_validate(lambda x: len(x) < 200, "Under 200 chars"),
            ],
            strict=False,
        )
        relaxed_chain = prompt | model | relaxed_parser
        result = relaxed_chain.invoke({"topic": "blockchain technology"})
        print(f"✓ Success with relaxed requirements:\n{result}\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Mellea Output Parser Examples")
    print("=" * 70 + "\n")

    # Run examples
    example_basic_parser()
    example_non_strict_mode()
    example_format_instructions()
    example_custom_validators()
    example_error_handling()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
