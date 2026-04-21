"""Async operations example for Mellea + DSPy integration.

This example demonstrates how to use async operations with Mellea and DSPy
for improved performance and concurrent processing.
"""

import asyncio
import time

import dspy
from mellea import start_session
from mellea_dspy import MelleaLM


async def basic_async():
    """Demonstrate basic async generation."""
    print("=" * 70)
    print("Example 1: Basic Async Generation")
    print("=" * 70)

    # Setup
    print("\n1. Setting up Mellea and DSPy...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class AnswerQuestion(dspy.Signature):
        """Answer a question."""

        question = dspy.InputField()
        answer = dspy.OutputField()

    # Create predictor (will be used via LM's aforward method)
    dspy.Predict(AnswerQuestion)
    print("   ✓ Predictor created")

    # Async generation
    print("\n3. Generating answer asynchronously...")
    question = "What is the benefit of async operations?"
    print(f"   Question: {question}")

    # Use the LM's aforward method directly
    response = await lm.aforward(
        prompt=f"Question: {question}\nAnswer:", model_options={}
    )

    content = response.choices[0].message.content
    print(f"   Answer: {content}")

    print("\n" + "=" * 70)


async def concurrent_generation():
    """Demonstrate concurrent generation with asyncio."""
    print("\n" + "=" * 70)
    print("Example 2: Concurrent Generation")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class Translate(dspy.Signature):
        """Translate text to a target language."""

        text = dspy.InputField()
        language = dspy.InputField()
        translation = dspy.OutputField()

    print("   ✓ Signature defined")

    # Concurrent translations
    print("\n3. Performing concurrent translations...")
    text = "Hello, how are you?"
    languages = ["Spanish", "French", "German", "Italian"]

    async def translate_to_language(lang):
        """Translate to a specific language."""
        prompt = f"Translate '{text}' to {lang}"
        response = await lm.aforward(prompt=prompt, model_options={})
        return lang, response.choices[0].message.content

    # Measure time for concurrent execution
    start_time = time.time()

    # Run translations concurrently
    tasks = [translate_to_language(lang) for lang in languages]
    results = await asyncio.gather(*tasks)

    concurrent_time = time.time() - start_time

    print(f"\n   Original text: {text}")
    print("   Translations:")
    for lang, translation in results:
        print(f"   - {lang}: {translation}")

    print(f"\n   Concurrent execution time: {concurrent_time:.2f}s")
    print(f"   Note: Sequential would take ~{concurrent_time * len(languages):.2f}s")

    print("\n" + "=" * 70)


async def batch_processing():
    """Demonstrate batch processing with async."""
    print("\n" + "=" * 70)
    print("Example 3: Batch Processing")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class Summarize(dspy.Signature):
        """Summarize text."""

        text = dspy.InputField()
        summary = dspy.OutputField()

    print("   ✓ Signature defined")

    # Batch of texts to summarize
    print("\n3. Processing batch of texts...")
    texts = [
        "Artificial intelligence is transforming how we work and live.",
        "Machine learning models require large amounts of data for training.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning has achieved remarkable results in image recognition.",
        "Generative AI can create new content based on learned patterns.",
    ]

    async def summarize_text(text, index):
        """Summarize a single text."""
        prompt = f"Summarize in one sentence: {text}"
        response = await lm.aforward(prompt=prompt, model_options={})
        return index, response.choices[0].message.content

    # Process batch
    start_time = time.time()

    tasks = [summarize_text(text, i) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks)

    batch_time = time.time() - start_time

    print(f"\n   Processed {len(texts)} texts in {batch_time:.2f}s")
    print(f"   Average time per text: {batch_time / len(texts):.2f}s")
    print("\n   Results:")
    for index, summary in sorted(results):
        print(f"   {index + 1}. {summary}")

    print("\n" + "=" * 70)


async def async_chain_of_thought():
    """Demonstrate async chain of thought reasoning."""
    print("\n" + "=" * 70)
    print("Example 4: Async Chain of Thought")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class ReasonAndAnswer(dspy.Signature):
        """Reason about a question and provide an answer."""

        question = dspy.InputField()
        reasoning = dspy.OutputField()
        answer = dspy.OutputField()

    print("   ✓ Signature defined")

    # Async reasoning
    print("\n3. Performing async reasoning...")
    questions = [
        "Why is async processing beneficial for AI applications?",
        "How does concurrent execution improve throughput?",
        "What are the trade-offs of parallel processing?",
    ]

    async def reason_about_question(question):
        """Reason about a question asynchronously."""
        prompt = f"Question: {question}\nProvide reasoning and answer."
        response = await lm.aforward(prompt=prompt, model_options={})
        return question, response.choices[0].message.content

    # Process questions concurrently
    tasks = [reason_about_question(q) for q in questions]
    results = await asyncio.gather(*tasks)

    print("\n   Results:")
    for i, (question, response) in enumerate(results, 1):
        print(f"\n   Question {i}: {question}")
        print(f"   Response: {response[:200]}...")

    print("\n" + "=" * 70)


async def async_with_error_handling():
    """Demonstrate async operations with error handling."""
    print("\n" + "=" * 70)
    print("Example 5: Async with Error Handling")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Define signature
    print("\n2. Defining signature...")

    class ProcessText(dspy.Signature):
        """Process text with potential errors."""

        text = dspy.InputField()
        result = dspy.OutputField()

    print("   ✓ Signature defined")

    # Process with error handling
    print("\n3. Processing with error handling...")

    async def safe_process(text, index):
        """Process text with error handling."""
        try:
            prompt = f"Process: {text}"
            response = await lm.aforward(prompt=prompt, model_options={})
            return {
                "index": index,
                "status": "success",
                "result": response.choices[0].message.content,
            }
        except Exception as e:
            return {"index": index, "status": "error", "error": str(e)}

    texts = ["Valid text to process", "Another valid text", "Yet another text"]

    tasks = [safe_process(text, i) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n   Results:")
    for result in results:
        if isinstance(result, dict):
            if result["status"] == "success":
                print(f"   ✓ Text {result['index']}: Success")
            else:
                print(f"   ✗ Text {result['index']}: Error - {result['error']}")
        else:
            print(f"   ✗ Unexpected error: {result}")

    print("\n   Note: Error handling ensures robustness in production")

    print("\n" + "=" * 70)


async def async_streaming_simulation():
    """Demonstrate simulated streaming with async."""
    print("\n" + "=" * 70)
    print("Example 6: Async Streaming Simulation")
    print("=" * 70)

    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")

    # Simulate streaming
    print("\n2. Simulating streaming generation...")
    print("   Note: This simulates streaming by processing chunks")

    async def generate_with_progress(prompt):
        """Generate with progress updates."""
        print(f"\n   Generating response for: {prompt[:50]}...")

        # Simulate progress
        for i in range(3):
            await asyncio.sleep(0.5)
            print(f"   Progress: {(i + 1) * 33}%")

        response = await lm.aforward(prompt=prompt, model_options={})
        return response.choices[0].message.content

    prompt = "Explain the benefits of async programming in AI applications"
    result = await generate_with_progress(prompt)

    print(f"\n   Final result: {result[:150]}...")

    print("\n" + "=" * 70)


async def run_all_examples():
    """Run all async examples in a single async context."""
    # Run async examples
    await basic_async()
    await concurrent_generation()
    await batch_processing()
    await async_chain_of_thought()
    await async_with_error_handling()
    await async_streaming_simulation()


def main():
    """Run all async examples."""
    print("\n" + "=" * 70)
    print("MELLEA + DSPY: ASYNC OPERATIONS EXAMPLES")
    print("=" * 70)

    try:
        # Run all examples in a single async context
        asyncio.run(run_all_examples())

        # Summary
        print("\n" + "=" * 70)
        print("✅ All async operation examples completed!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Async operations improve performance")
        print("  • Concurrent execution processes multiple requests")
        print("  • Batch processing handles large datasets efficiently")
        print("  • Error handling ensures robustness")
        print("  • Async enables responsive applications")
        print("\nBest Practices:")
        print("  • Use asyncio.gather() for concurrent tasks")
        print("  • Implement proper error handling")
        print("  • Monitor and limit concurrency")
        print("  • Consider rate limits and quotas")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
