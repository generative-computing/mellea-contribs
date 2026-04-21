"""Example demonstrating CrewAI validators with Mellea requirements.

This example shows how to use Mellea requirements as CrewAI guardrails
for task output validation with automatic retry on validation failure.

Requirements:
    - Ollama running locally (or configure a different backend)
    - mellea-crewai installed

Run:
    python examples/validator_usage_example.py
"""

import logging

from crewai import Agent, Crew, Task
from mellea import start_session

from mellea_crewai import (
    MelleaLLM,
    create_guardrail,
    create_guardrails,
)

# Configure logging to show validation details
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")


# Helper function to create simple validators
def simple_validate(fn, description):
    """Create a simple validator function with description."""
    validator = fn
    validator.description = description
    return validator


def example_1_basic_guardrail():
    """Example 1: Basic guardrail from Mellea requirement."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Guardrail from Mellea Requirement")
    print("=" * 70)

    # Setup
    m = start_session()
    llm = MelleaLLM(mellea_session=m)

    # Create Mellea requirement
    print("\n1. Creating validation function...")
    word_count_req = simple_validate(
        lambda x: 50 <= len(x.split()) <= 150, "Must be between 50-150 words"
    )
    print("   ✓ Requirement created: Must be between 50-150 words")

    # Convert to CrewAI guardrail
    print("\n2. Converting to CrewAI guardrail...")
    word_count_guard = create_guardrail(word_count_req)
    print("   ✓ Guardrail created")

    # Create agent and task
    print("\n3. Creating agent and task...")
    agent = Agent(
        role="Writer",
        goal="Write quality content",
        backstory="You are a professional writer",
        llm=llm,
        verbose=False,
    )

    task = Task(
        description="Write a brief summary about artificial intelligence",
        expected_output="A well-written summary",
        agent=agent,
        guardrails=[word_count_guard],
        guardrail_max_retries=3,
    )
    print("   ✓ Task created with guardrail")

    # Execute
    print("\n4. Executing task...")
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("Result:")
    print("=" * 70)
    print(result.raw)
    print("=" * 70)
    print(f"Word count: {len(result.raw.split())} words")


def example_2_multiple_guardrails():
    """Example 2: Multiple guardrails with Mellea requirements."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Guardrails")
    print("=" * 70)

    # Setup
    m = start_session()
    llm = MelleaLLM(mellea_session=m)

    # Create multiple requirements
    print("\n1. Creating multiple requirements...")
    requirements = [
        simple_validate(lambda x: 50 <= len(x.split()) <= 150, "Must be between 50-150 words"),
        simple_validate(
            lambda x: "AI" in x or "artificial intelligence" in x.lower(),
            "Must mention AI or artificial intelligence",
        ),
        simple_validate(lambda x: x.strip() == x, "Must not have leading/trailing whitespace"),
    ]
    print(f"   ✓ Created {len(requirements)} requirements")

    # Convert to guardrails
    print("\n2. Converting to CrewAI guardrails...")
    guardrails = create_guardrails(requirements)
    print(f"   ✓ Created {len(guardrails)} guardrails")

    # Create agent and task
    print("\n3. Creating agent and task...")
    agent = Agent(
        role="Technical Writer",
        goal="Write accurate technical content",
        backstory="You are an expert technical writer",
        llm=llm,
        verbose=False,
    )

    task = Task(
        description="Write a summary about artificial intelligence and its applications",
        expected_output="Technical summary",
        agent=agent,
        guardrails=guardrails,
        guardrail_max_retries=3,
    )
    print("   ✓ Task created with multiple guardrails")

    # Execute
    print("\n4. Executing task...")
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("Result:")
    print("=" * 70)
    print(result.raw)
    print("=" * 70)


def main():
    """Run all validator examples."""
    print("\n" + "=" * 70)
    print("CrewAI Validators with Mellea Requirements - Examples")
    print("=" * 70)

    try:
        # Run examples
        example_1_basic_guardrail()
        example_2_multiple_guardrails()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

        print("\nKey Takeaways:")
        print("  • Mellea requirements can be converted to CrewAI guardrails")
        print("  • Multiple guardrails can be applied sequentially")
        print("  • Guardrails integrate with CrewAI's retry mechanism")
        print("  • Logging shows validation details and failure reasons")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("  • Ollama is running (or configure a different backend)")
        print("  • mellea-crewai is installed")
        print("  • All dependencies are available")


if __name__ == "__main__":
    main()
