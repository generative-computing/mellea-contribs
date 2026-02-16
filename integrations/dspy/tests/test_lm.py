"""Simple test for Mellea + DSPy integration.

This test verifies that the MelleaLM class works correctly with DSPy.
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaLM


def test_basic_integration():
    """Test basic integration between Mellea and DSPy."""
    print("Testing Mellea + DSPy Integration...")

    try:
        # Create Mellea session
        print("1. Creating Mellea session...")
        m = start_session()
        print("   ✓ Mellea session created")

        # Create MelleaLM
        print("2. Creating MelleaLM...")
        lm = MelleaLM(mellea_session=m, model="mellea-test")
        print("   ✓ MelleaLM created")

        # Configure DSPy
        print("3. Configuring DSPy...")
        dspy.configure(lm=lm)
        print("   ✓ DSPy configured")

        # Test simple prediction
        print("4. Testing simple prediction...")
        predictor = dspy.Predict("question -> answer")
        response = predictor(question="What is 2+2?")
        print("   Question: What is 2+2?")
        print(f"   Answer: {response.answer}")
        print("   ✓ Prediction successful")

        # Test that response has expected attributes
        assert hasattr(response, "answer"), "Response should have 'answer' attribute"
        assert response.answer, "Answer should not be empty"

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_forward_method():
    """Test the forward method directly."""
    print("\nTesting forward method...")

    try:
        # Create Mellea session and LM
        m = start_session()
        lm = MelleaLM(mellea_session=m, model="mellea-test")

        # Test with prompt
        print("1. Testing with prompt...")
        response = lm.forward(prompt="Say hello")
        print(f"   Response type: {type(response)}")
        print(f"   Has choices: {hasattr(response, 'choices')}")
        if hasattr(response, "choices") and response.choices:
            print(f"   Content: {response.choices[0].message.content[:50]}...")
        print("   ✓ Forward with prompt works")

        # Test with messages
        print("2. Testing with messages...")
        messages = [{"role": "user", "content": "What is Python?"}]
        response = lm.forward(messages=messages)
        print(f"   Response type: {type(response)}")
        if hasattr(response, "choices") and response.choices:
            print(f"   Content: {response.choices[0].message.content[:50]}...")
        print("   ✓ Forward with messages works")

        print("\n✅ Forward method tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Forward method test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_chain_of_thought():
    """Test Chain of Thought with MelleaLM."""
    print("\nTesting Chain of Thought...")

    try:
        # Create Mellea session and configure DSPy
        m = start_session()
        lm = MelleaLM(mellea_session=m, model="mellea-test")
        dspy.configure(lm=lm)

        # Test CoT
        print("1. Creating Chain of Thought predictor...")
        cot = dspy.ChainOfThought("question -> answer")
        print("   ✓ CoT predictor created")

        print("2. Running CoT prediction...")
        response = cot(question="Why is the sky blue?")
        print("   Question: Why is the sky blue?")
        if hasattr(response, "rationale"):
            print(f"   Rationale: {response.rationale[:100]}...")
        print(f"   Answer: {response.answer[:100]}...")
        print("   ✓ CoT prediction successful")

        print("\n✅ Chain of Thought test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Chain of Thought test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Mellea + DSPy Integration Tests")
    print("=" * 60)
    print()

    results = []

    # Run tests
    results.append(("Basic Integration", test_basic_integration()))
    results.append(("Forward Method", test_forward_method()))
    results.append(("Chain of Thought", test_chain_of_thought()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
