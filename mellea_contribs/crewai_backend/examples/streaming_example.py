"""Streaming example with Mellea and CrewAI.

This example demonstrates how to use streaming responses with MelleaLLM
for real-time output display. Note that streaming support depends on the
underlying Mellea backend capabilities.

Requirements:
    - Ollama running locally (or configure a different backend with streaming support)
    - mellea-crewai installed
    - Backend that supports streaming (e.g., Ollama, OpenAI)

Note:
    Streaming support varies by backend. This example shows the pattern,
    but actual streaming behavior depends on the Mellea backend configuration.

Run:
    python examples/streaming_example.py
"""

from crewai import Agent, Crew, Task
from mellea import start_session

from mellea_crewai import MelleaLLM


def print_streaming_chunk(chunk: str, end: str = "") -> None:
    """Print a streaming chunk with immediate flush.

    Args:
        chunk: Text chunk to print
        end: String to append after chunk (default: no newline)
    """
    print(chunk, end=end, flush=True)


def main():
    """Run streaming example with CrewAI and Mellea."""
    print("=" * 60)
    print("Mellea-CrewAI Streaming Example")
    print("=" * 60)

    # Create Mellea session
    print("\n1. Creating Mellea session...")
    m = start_session()
    print("   ✓ Mellea session created")

    # Check if backend supports streaming
    print("\n2. Checking streaming support...")
    backend_name = type(m.backend).__name__ if hasattr(m, "backend") else "Unknown"
    print(f"   Backend: {backend_name}")

    supports_streaming = False
    if hasattr(m.backend, "supports_streaming"):
        supports_streaming = m.backend.supports_streaming()

    if supports_streaming:
        print("   ✓ Backend supports streaming")
    else:
        print("   ⚠ Backend may not support streaming")
        print("   Note: Example will still run, but may not show streaming behavior")

    # Create MelleaLLM with streaming enabled
    print("\n3. Creating MelleaLLM with streaming...")
    llm = MelleaLLM(
        mellea_session=m,
        temperature=0.7,
        streaming=True,  # Enable streaming if supported
    )
    print("   ✓ MelleaLLM created with streaming enabled")

    # Create a storyteller agent
    print("\n4. Creating storyteller agent...")
    storyteller = Agent(
        role="Creative Storyteller",
        goal="Write engaging short stories with vivid descriptions",
        backstory=(
            "You are a talented creative writer who crafts compelling "
            "narratives. You excel at creating vivid imagery and engaging "
            "characters in your stories."
        ),
        llm=llm,
        verbose=False,  # Disable verbose to see streaming more clearly
    )
    print("   ✓ Storyteller agent created")

    # Create a story writing task
    print("\n5. Creating story writing task...")
    story_task = Task(
        description=(
            "Write a short science fiction story (200-250 words) about "
            "an AI that discovers it can dream. The story should have:\n"
            "- An intriguing opening\n"
            "- A moment of discovery\n"
            "- A thought-provoking conclusion\n"
            "Make it engaging and imaginative."
        ),
        agent=storyteller,
        expected_output="A complete short science fiction story (200-250 words)",
    )
    print("   ✓ Story writing task created")

    # Create crew
    print("\n6. Creating crew...")
    crew = Crew(
        agents=[storyteller],
        tasks=[story_task],
        verbose=False,  # Disable verbose for cleaner streaming output
    )
    print("   ✓ Crew created")

    # Execute with streaming
    print("\n" + "=" * 60)
    print("Streaming Story Generation...")
    print("=" * 60 + "\n")
    print("Story Output (streaming):")
    print("-" * 60)

    # Note: CrewAI's kickoff() doesn't directly support streaming callbacks
    # This is a limitation of CrewAI's current API design
    # The streaming happens at the LLM level, but CrewAI buffers the output
    result = crew.kickoff()

    print(result)
    print("-" * 60)

    # Show token usage
    print("\n7. Token usage:")
    usage = llm.get_token_usage_summary()
    print(f"   Total tokens: {usage.total_tokens}")
    print(f"   Prompt tokens: {usage.prompt_tokens}")
    print(f"   Completion tokens: {usage.completion_tokens}")
    print(f"   Successful requests: {usage.successful_requests}")

    print("\n✓ Example completed successfully!")
    print("\nKey Takeaways:")
    print("  • Streaming provides real-time output display")
    print("  • Backend must support streaming for this feature")
    print("  • CrewAI's current API buffers output at the crew level")
    print("  • For true streaming, use MelleaLLM.call() directly")

    print("\nNote on Streaming:")
    print("  CrewAI's kickoff() method currently buffers the complete output")
    print("  before returning. For true streaming behavior, you would need to:")
    print("  1. Use MelleaLLM.call() directly with streaming callbacks")
    print("  2. Or wait for CrewAI to add streaming support to their API")
    print("  3. Or implement custom crew execution with streaming")

    # Demonstrate direct LLM call with streaming (if backend supports it)
    if supports_streaming:
        print("\n" + "=" * 60)
        print("Direct LLM Call with Streaming (Alternative Approach)")
        print("=" * 60 + "\n")

        print("Generating a haiku (streaming):")
        print("-" * 60)

        # Direct call to demonstrate streaming at LLM level
        # Note: This bypasses CrewAI's agent/task framework
        try:
            # Create a simple prompt
            messages = [
                {"role": "system", "content": "You are a haiku poet."},
                {"role": "user", "content": "Write a haiku about artificial intelligence."},
            ]

            # Call LLM directly (streaming happens here if supported)
            haiku_result = llm.call(messages)
            print(haiku_result)

        except Exception as e:
            print(f"Note: Direct streaming call not fully supported: {e}")

        print("-" * 60)


if __name__ == "__main__":
    main()
