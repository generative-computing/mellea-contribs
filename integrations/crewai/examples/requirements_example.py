"""Requirements and validation example with Mellea and CrewAI.

This example demonstrates how to use Mellea's requirements and sampling
strategies with CrewAI agents to ensure validated outputs.

Requirements:
    - Ollama running locally (or configure a different backend)
    - mellea-crewai installed

Run:
    python examples/requirements_example.py
"""

from crewai import Agent, Crew, Task
from mellea import start_session
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_crewai import MelleaLLM


def main():
    """Run requirements validation example with CrewAI."""
    print("=" * 60)
    print("Mellea-CrewAI Requirements & Validation Example")
    print("=" * 60)

    # Create Mellea session
    print("\n1. Creating Mellea session...")
    m = start_session()
    print("   ✓ Mellea session created")

    # Define requirements for validated outputs
    print("\n2. Defining requirements...")
    requirements = [
        req("The email should have a professional salutation"),
        req("The email should be concise (under 200 words)"),
        req("Include a clear call-to-action"),
        check("Do not use overly casual language"),
    ]
    print("   ✓ Requirements defined:")
    for i, r in enumerate(requirements, 1):
        print(f"      {i}. {r}")

    # Create MelleaLLM with requirements and sampling strategy
    print("\n3. Creating MelleaLLM with validation...")
    llm = MelleaLLM(
        mellea_session=m,
        requirements=requirements,
        strategy=RejectionSamplingStrategy(loop_budget=5),
        return_sampling_results=True,
        temperature=0.7,
    )
    print("   ✓ MelleaLLM created with:")
    print("      - Requirements validation enabled")
    print("      - RejectionSamplingStrategy (max 5 attempts)")
    print("      - Sampling results tracking enabled")

    # Create a professional writer agent
    print("\n4. Creating professional writer agent...")
    writer = Agent(
        role="Professional Email Writer",
        goal="Write clear, professional, and effective emails",
        backstory=(
            "You are an experienced business communication specialist. "
            "You excel at writing professional emails that are clear, "
            "concise, and achieve their intended purpose."
        ),
        llm=llm,
        verbose=True,
    )
    print("   ✓ Writer agent created")

    # Create an email writing task
    print("\n5. Creating email writing task...")
    email_task = Task(
        description=(
            "Write a professional email to the engineering team about "
            "an upcoming code review session. The session is scheduled "
            "for next Tuesday at 2 PM. Emphasize the importance of "
            "preparing their code beforehand and being ready to discuss "
            "their implementation decisions."
        ),
        agent=writer,
        expected_output="A professional email meeting all requirements",
    )
    print("   ✓ Email writing task created")

    # Create crew and execute
    print("\n6. Creating crew and executing task...")
    crew = Crew(
        agents=[writer],
        tasks=[email_task],
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Executing Crew with Validation...")
    print("=" * 60 + "\n")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Validated Result:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # Show validation statistics
    print("\n7. Validation Statistics:")
    usage = llm.get_token_usage_summary()
    print(f"   Total tokens: {usage.total_tokens}")
    print(f"   Successful requests: {usage.successful_requests}")
    print("   Note: Multiple requests may indicate retry attempts")

    print("\n✓ Example completed successfully!")
    print("\nKey Takeaways:")
    print("  • Requirements ensure output quality")
    print("  • Sampling strategies handle validation failures")
    print("  • Mellea automatically retries until requirements are met")
    print("  • Works seamlessly with CrewAI agents")


if __name__ == "__main__":
    main()
