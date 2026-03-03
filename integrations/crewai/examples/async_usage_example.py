"""Async usage example with Mellea and CrewAI.

This example demonstrates how to use MelleaLLM's async capabilities with CrewAI
agents for concurrent execution and improved performance.

Requirements:
    - Ollama running locally (or configure a different backend)
    - mellea-crewai installed

Run:
    python examples/async_usage_example.py
"""

import asyncio
import time

from crewai import Agent, Crew, Task
from mellea import start_session

from mellea_crewai import MelleaLLM


async def main():
    """Run async CrewAI example with Mellea."""
    print("=" * 60)
    print("Mellea-CrewAI Async Usage Example")
    print("=" * 60)

    # Create Mellea session
    print("\n1. Creating Mellea session...")
    m = start_session()
    print("   ✓ Mellea session created")

    # Create MelleaLLM
    print("\n2. Creating MelleaLLM...")
    llm = MelleaLLM(mellea_session=m, temperature=0.7)
    print("   ✓ MelleaLLM created")

    # Create multiple agents for concurrent execution
    print("\n3. Creating multiple agents...")
    
    researcher = Agent(
        role="AI Researcher",
        goal="Research AI topics quickly and thoroughly",
        backstory=(
            "You are an expert AI researcher who can quickly analyze "
            "and summarize complex topics."
        ),
        llm=llm,
        verbose=True,
    )
    
    analyst = Agent(
        role="Data Analyst",
        goal="Analyze trends and provide insights",
        backstory=(
            "You are a skilled data analyst who can identify patterns "
            "and trends in information."
        ),
        llm=llm,
        verbose=True,
    )
    
    writer = Agent(
        role="Technical Writer",
        goal="Write clear and concise technical content",
        backstory=(
            "You are an experienced technical writer who can explain "
            "complex concepts in simple terms."
        ),
        llm=llm,
        verbose=True,
    )
    
    print("   ✓ Created 3 agents: Researcher, Analyst, Writer")

    # Create tasks for each agent
    print("\n4. Creating tasks...")
    
    research_task = Task(
        description=(
            "Research the current state of transformer models in NLP. "
            "Focus on recent architectures and their key innovations."
        ),
        agent=researcher,
        expected_output="A brief summary of transformer model developments (150 words)",
    )
    
    analysis_task = Task(
        description=(
            "Analyze the adoption trends of large language models in enterprise. "
            "Identify key factors driving adoption."
        ),
        agent=analyst,
        expected_output="Analysis of LLM adoption trends (150 words)",
    )
    
    writing_task = Task(
        description=(
            "Write a brief explanation of how attention mechanisms work "
            "in neural networks for a general audience."
        ),
        agent=writer,
        expected_output="Clear explanation of attention mechanisms (150 words)",
    )
    
    print("   ✓ Created 3 tasks")

    # Execute tasks sequentially (for comparison)
    print("\n" + "=" * 60)
    print("Sequential Execution (for comparison)")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    crew_sequential = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        verbose=False,
    )
    
    print("Executing tasks sequentially...")
    result_sequential = crew_sequential.kickoff()
    
    sequential_time = time.time() - start_time
    print(f"\n✓ Sequential execution completed in {sequential_time:.2f} seconds")

    # Execute tasks concurrently using async
    print("\n" + "=" * 60)
    print("Concurrent Execution (using async)")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    # Create separate crews for concurrent execution
    crew1 = Crew(agents=[researcher], tasks=[research_task], verbose=False)
    crew2 = Crew(agents=[analyst], tasks=[analysis_task], verbose=False)
    crew3 = Crew(agents=[writer], tasks=[writing_task], verbose=False)
    
    print("Executing tasks concurrently...")
    
    # Run crews concurrently
    results = await asyncio.gather(
        asyncio.to_thread(crew1.kickoff),
        asyncio.to_thread(crew2.kickoff),
        asyncio.to_thread(crew3.kickoff),
    )
    
    concurrent_time = time.time() - start_time
    print(f"\n✓ Concurrent execution completed in {concurrent_time:.2f} seconds")

    # Display results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    
    print("\n[Research Task Result]")
    print(results[0])
    
    print("\n[Analysis Task Result]")
    print(results[1])
    
    print("\n[Writing Task Result]")
    print(results[2])
    
    print("\n" + "=" * 60)

    # Show performance comparison
    print("\n5. Performance Comparison:")
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
    print(f"   Sequential time: {sequential_time:.2f}s")
    print(f"   Concurrent time: {concurrent_time:.2f}s")
    print(f"   Speedup: {speedup:.2f}x")
    
    # Show token usage
    print("\n6. Token usage:")
    usage = llm.get_token_usage_summary()
    print(f"   Total tokens: {usage.total_tokens}")
    print(f"   Prompt tokens: {usage.prompt_tokens}")
    print(f"   Completion tokens: {usage.completion_tokens}")
    print(f"   Successful requests: {usage.successful_requests}")

    print("\n✓ Example completed successfully!")
    print("\nKey Takeaways:")
    print("  • Async execution enables concurrent task processing")
    print("  • Multiple agents can work simultaneously")
    print("  • Significant performance improvements for independent tasks")
    print("  • MelleaLLM supports both sync and async operations")
    print("  • Use asyncio.gather() for concurrent crew execution")


if __name__ == "__main__":
    asyncio.run(main())

# Made with Bob
