"""Basic usage example of Mellea with CrewAI.

This example demonstrates how to use MelleaLLM with CrewAI agents.

Requirements:
    - Ollama running locally (or configure a different backend)
    - mellea-crewai installed

Run:
    python examples/basic_usage.py
"""

from crewai import Agent, Crew, Task
from mellea import start_session

from mellea_crewai import MelleaLLM


def main():
    """Run basic CrewAI example with Mellea."""
    print("=" * 60)
    print("Mellea-CrewAI Basic Usage Example")
    print("=" * 60)

    # Create Mellea session (uses Ollama by default)
    print("\n1. Creating Mellea session...")
    m = start_session()
    print("   ✓ Mellea session created")

    # Create CrewAI LLM
    print("\n2. Creating MelleaLLM...")
    llm = MelleaLLM(mellea_session=m, temperature=0.7)
    print("   ✓ MelleaLLM created")

    # Create a researcher agent
    print("\n3. Creating researcher agent...")
    researcher = Agent(
        role="AI Researcher",
        goal="Research and analyze AI trends",
        backstory=(
            "You are an expert AI researcher with deep knowledge of "
            "machine learning, natural language processing, and generative AI. "
            "You stay up-to-date with the latest developments in the field."
        ),
        llm=llm,
        verbose=True,
    )
    print("   ✓ Researcher agent created")

    # Create a research task
    print("\n4. Creating research task...")
    research_task = Task(
        description=(
            "Research the current state of large language models (LLMs). "
            "Focus on recent developments, key capabilities, and limitations. "
            "Provide a concise summary of your findings."
        ),
        agent=researcher,
        expected_output="A concise summary of LLM developments (200-300 words)",
    )
    print("   ✓ Research task created")

    # Create crew and execute
    print("\n5. Creating crew and executing task...")
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Executing Crew...")
    print("=" * 60 + "\n")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # Show token usage
    print("\n6. Token usage:")
    usage = llm.get_token_usage_summary()
    print(f"   Total tokens: {usage.total_tokens}")
    print(f"   Prompt tokens: {usage.prompt_tokens}")
    print(f"   Completion tokens: {usage.completion_tokens}")
    print(f"   Successful requests: {usage.successful_requests}")

    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()
