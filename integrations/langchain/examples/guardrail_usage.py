"""Example demonstrating MelleaGuardrail usage.

This example shows how to use MelleaGuardrail for independent validation
and composition of multiple guardrails.
"""

from mellea_langchain import MelleaGuardrail


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


def example_basic_guardrail():
    """Example 1: Basic guardrail validation."""
    print("=" * 70)
    print("Example 1: Basic Guardrail Validation")
    print("=" * 70)

    # Create guardrail with requirements
    guardrail = MelleaGuardrail(
        requirements=[
            simple_validate(lambda x: len(x) > 50, "Must be at least 50 characters"),
            simple_validate(lambda x: len(x) < 500, "Must be under 500 characters"),
            simple_validate(lambda x: x.strip() == x, "Must not have leading/trailing whitespace"),
        ],
        name="length_and_format_check",
    )

    # Test with valid text
    print("\n1. Testing with valid text...")
    valid_text = "This is a valid message that meets all the requirements. " * 2
    result = guardrail.validate(valid_text)

    print(f"   Passed: {result.passed}")
    print(f"   Errors: {result.errors}")
    print(f"   Metadata: {result.metadata}")

    # Test with invalid text (too short)
    print("\n2. Testing with invalid text (too short)...")
    invalid_text = "Too short"
    result = guardrail.validate(invalid_text)

    print(f"   Passed: {result.passed}")
    print(f"   Errors: {result.errors}")
    print(f"   Failed requirements: {result.metadata['failed_requirements']}")


def example_composing_guardrails():
    """Example 2: Composing multiple guardrails."""
    print("\n" + "=" * 70)
    print("Example 2: Composing Guardrails")
    print("=" * 70)

    # Create specialized guardrails
    length_guardrail = MelleaGuardrail(
        requirements=[
            simple_validate(lambda x: 50 < len(x) < 500, "50-500 characters"),
        ],
        name="length_check",
    )

    format_guardrail = MelleaGuardrail(
        requirements=[
            simple_validate(lambda x: x.strip() == x, "No extra whitespace"),
            simple_validate(lambda x: not x.isupper(), "Not all uppercase"),
        ],
        name="format_check",
    )

    content_guardrail = MelleaGuardrail(
        requirements=[
            simple_validate(
                lambda x: "AI" in x or "artificial intelligence" in x.lower(), "Mentions AI"
            ),
        ],
        name="content_check",
    )

    # Compose using method
    print("\n1. Composing using compose() method...")
    combined = length_guardrail.compose(format_guardrail).compose(content_guardrail)
    print(f"   Combined guardrail: {combined}")
    print(f"   Total requirements: {len(combined.requirements)}")

    # Compose using & operator
    print("\n2. Composing using & operator...")
    combined_alt = length_guardrail & format_guardrail & content_guardrail
    print(f"   Combined guardrail: {combined_alt}")

    # Test combined guardrail
    print("\n3. Testing combined guardrail...")
    test_text = "Artificial intelligence is transforming how we interact with technology. " * 2
    result = combined.validate(test_text)

    print(f"   Passed: {result.passed}")
    if result.passed:
        print(f"   ✓ All {result.metadata['total_requirements']} requirements met!")
    else:
        print(f"   ✗ Failed requirements: {result.errors}")


def example_validation_metadata():
    """Example 3: Using validation metadata."""
    print("\n" + "=" * 70)
    print("Example 3: Validation Metadata")
    print("=" * 70)

    guardrail = MelleaGuardrail(
        requirements=[
            simple_validate(lambda x: len(x.split()) > 10, "At least 10 words"),
            simple_validate(lambda x: len(x.split()) < 100, "Under 100 words"),
            simple_validate(lambda x: any(c.isupper() for c in x), "Contains uppercase"),
            simple_validate(lambda x: "." in x, "Contains period"),
        ],
        name="comprehensive_check",
    )

    test_text = "This is a test message with proper formatting. It has multiple sentences."
    result = guardrail.validate(test_text)

    print("\n1. Validation result:")
    print(f"   Passed: {result.passed}")
    print(f"   Text length: {len(result.text)} characters")

    print("\n2. Detailed metadata:")
    for key, value in result.metadata.items():
        print(f"   {key}: {value}")

    print("\n3. Individual requirement results:")
    if "passed_requirement_list" in result.metadata:
        for req in result.metadata["passed_requirement_list"]:
            print(f"   ✓ {req}")
    for error in result.errors:
        print(f"   ✗ {error}")


def example_error_recovery():
    """Example 4: Error recovery with guardrails."""
    print("\n" + "=" * 70)
    print("Example 4: Error Recovery")
    print("=" * 70)

    # Strict guardrail
    strict_guardrail = MelleaGuardrail(
        requirements=[
            simple_validate(lambda x: len(x) < 100, "Under 100 characters"),
            simple_validate(lambda x: x.islower(), "All lowercase"),
            simple_validate(lambda x: "test" in x, "Contains 'test'"),
        ],
        name="strict_check",
    )

    # Relaxed guardrail
    relaxed_guardrail = MelleaGuardrail(
        requirements=[
            simple_validate(lambda x: len(x) < 500, "Under 500 characters"),
            simple_validate(lambda x: len(x) > 10, "At least 10 characters"),
        ],
        name="relaxed_check",
    )

    test_text = "This is a Test message that might not meet all requirements."

    print("\n1. Testing with strict guardrail...")
    result = strict_guardrail.validate(test_text)

    if not result.passed:
        print(f"   ✗ Strict validation failed: {len(result.errors)} errors")
        for error in result.errors:
            print(f"      - {error}")

        print("\n2. Falling back to relaxed guardrail...")
        result = relaxed_guardrail.validate(test_text)

        if result.passed:
            print("   ✓ Relaxed validation passed!")
            print(f"   Text accepted: {result.text[:50]}...")
        else:
            print("   ✗ Even relaxed validation failed")
    else:
        print("   ✓ Strict validation passed!")


def example_custom_validators():
    """Example 5: Custom validation functions."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Validators")
    print("=" * 70)

    # Define custom validators
    def has_email_format(text):
        """Check if text contains email-like pattern."""
        return "@" in text and "." in text

    def has_phone_number(text):
        """Check if text contains phone number pattern."""
        import re

        return bool(re.search(r"\d{3}[-.]?\d{3}[-.]?\d{4}", text))

    def professional_tone(text):
        """Check for professional language."""
        casual_words = ["hey", "gonna", "wanna", "yeah"]
        return not any(word in text.lower() for word in casual_words)

    # Create guardrail with custom validators
    contact_guardrail = MelleaGuardrail(
        requirements=[
            simple_validate(has_email_format, "Must contain email address"),
            simple_validate(has_phone_number, "Must contain phone number"),
            simple_validate(professional_tone, "Must use professional tone"),
        ],
        name="contact_info_check",
    )

    # Test with valid contact info
    print("\n1. Testing with valid contact information...")
    valid_contact = "Please contact me at john.doe@example.com or call 555-123-4567."
    result = contact_guardrail.validate(valid_contact)

    print(f"   Passed: {result.passed}")
    if result.passed:
        print("   ✓ All contact requirements met!")
    else:
        print(f"   ✗ Missing: {result.errors}")

    # Test with invalid contact info
    print("\n2. Testing with invalid contact information...")
    invalid_contact = "Hey, gonna send you the info later, yeah?"
    result = contact_guardrail.validate(invalid_contact)

    print(f"   Passed: {result.passed}")
    print(f"   Errors: {result.errors}")


def example_progressive_validation():
    """Example 6: Progressive validation with multiple stages."""
    print("\n" + "=" * 70)
    print("Example 6: Progressive Validation")
    print("=" * 70)

    # Stage 1: Basic format check
    stage1 = MelleaGuardrail(
        requirements=[
            simple_validate(lambda x: len(x) > 0, "Not empty"),
            simple_validate(lambda x: x.strip() == x, "No extra whitespace"),
        ],
        name="stage1_format",
    )

    # Stage 2: Length check
    stage2 = MelleaGuardrail(
        requirements=[
            simple_validate(lambda x: len(x) > 20, "At least 20 characters"),
            simple_validate(lambda x: len(x) < 1000, "Under 1000 characters"),
        ],
        name="stage2_length",
    )

    # Stage 3: Content check
    stage3 = MelleaGuardrail(
        requirements=[
            simple_validate(lambda x: any(c.isupper() for c in x), "Has uppercase"),
            simple_validate(lambda x: any(c.islower() for c in x), "Has lowercase"),
            simple_validate(lambda x: "." in x or "!" in x or "?" in x, "Has punctuation"),
        ],
        name="stage3_content",
    )

    test_text = "This is a properly formatted message with good content!"

    print("\n1. Running progressive validation...")
    stages = [stage1, stage2, stage3]

    for i, stage in enumerate(stages, 1):
        result = stage.validate(test_text)
        print(f"\n   Stage {i} ({stage.name}):")
        print(f"   Passed: {result.passed}")

        if not result.passed:
            print(f"   ✗ Failed at stage {i}")
            print(f"   Errors: {result.errors}")
            break
        else:
            print(f"   ✓ Stage {i} passed")
    else:
        print(f"\n   ✓ All {len(stages)} stages passed!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Mellea Guardrail Examples")
    print("=" * 70)

    # Run examples
    example_basic_guardrail()
    example_composing_guardrails()
    example_validation_metadata()
    example_error_recovery()
    example_custom_validators()
    example_progressive_validation()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
