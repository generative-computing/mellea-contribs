"""Multi-agent crew example with Mellea and CrewAI.

This example demonstrates how to create a multi-agent crew where agents
collaborate on a complex task. It shows agent coordination, task dependencies,
and how different agents can use different LLM configurations.

Requirements:
    - Ollama running locally (or configure a different backend)
    - mellea-crewai installed

Run:
    python examples/multi_agent_crew_example.py
"""

from crewai import Agent, Crew, Task
from mellea import start_session
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_crewai import MelleaLLM


def main():
    """Run multi-agent crew example with Mellea."""
    print("=" * 60)
    print("Mellea-CrewAI Multi-Agent Crew Example")
    print("=" * 60)

    # Create Mellea session
    print("\n1. Creating Mellea session...")
    m = start_session()
    print("   ✓ Mellea session created")

    # Create different LLM configurations for different agents
    print("\n2. Creating LLM configurations...")

    # Researcher LLM - focused on accuracy and detail
    researcher_llm = MelleaLLM(
        mellea_session=m,
        temperature=0.3,  # Lower temperature for more focused research
    )
    print("   ✓ Researcher LLM created (temperature=0.3)")

    # Analyst LLM - with requirements for structured output
    analyst_llm = MelleaLLM(
        mellea_session=m,
        temperature=0.5,
        requirements=[
            req("Include specific data points or statistics"),
            req("Provide clear conclusions"),
            check("Avoid speculation without evidence"),
        ],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )
    print("   ✓ Analyst LLM created (temperature=0.5, with requirements)")

    # Writer LLM - creative but professional
    writer_llm = MelleaLLM(
        mellea_session=m,
        temperature=0.7,  # Higher temperature for more creative writing
        requirements=[
            req("Use professional tone"),
            req("Include clear section headings"),
            req("Keep paragraphs concise (3-4 sentences)"),
        ],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )
    print("   ✓ Writer LLM created (temperature=0.7, with requirements)")

    # Editor LLM - strict quality control
    editor_llm = MelleaLLM(
        mellea_session=m,
        temperature=0.2,  # Very low temperature for consistent editing
        requirements=[
            req("Identify any factual inconsistencies"),
            req("Check for clarity and readability"),
            check("Do not add new information"),
        ],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )
    print("   ✓ Editor LLM created (temperature=0.2, with requirements)")

    # Create specialized agents
    print("\n3. Creating specialized agents...")

    researcher = Agent(
        role="Senior AI Researcher",
        goal="Conduct thorough research on AI topics and gather accurate information",
        backstory=(
            "You are a senior AI researcher with 10+ years of experience in "
            "machine learning and natural language processing. You excel at "
            "finding and synthesizing information from multiple sources. "
            "You prioritize accuracy and cite your reasoning."
        ),
        llm=researcher_llm,
        verbose=True,
    )
    print("   ✓ Senior AI Researcher created")

    analyst = Agent(
        role="Data Analyst",
        goal="Analyze research findings and extract key insights",
        backstory=(
            "You are a skilled data analyst who specializes in AI/ML trends. "
            "You can identify patterns, compare approaches, and provide "
            "data-driven insights. You always back your conclusions with evidence."
        ),
        llm=analyst_llm,
        verbose=True,
    )
    print("   ✓ Data Analyst created")

    writer = Agent(
        role="Technical Content Writer",
        goal="Transform research and analysis into clear, engaging content",
        backstory=(
            "You are an experienced technical writer who can explain complex "
            "AI concepts to diverse audiences. You create well-structured, "
            "readable content that maintains technical accuracy while being "
            "accessible."
        ),
        llm=writer_llm,
        verbose=True,
    )
    print("   ✓ Technical Content Writer created")

    editor = Agent(
        role="Senior Editor",
        goal="Review and refine content for quality and consistency",
        backstory=(
            "You are a meticulous senior editor with expertise in technical "
            "content. You ensure clarity, consistency, and accuracy. You catch "
            "errors and improve readability without changing the core message."
        ),
        llm=editor_llm,
        verbose=True,
    )
    print("   ✓ Senior Editor created")

    # Create collaborative tasks with dependencies
    print("\n4. Creating collaborative tasks...")

    research_task = Task(
        description=(
            "Research the topic: 'Recent advances in retrieval-augmented generation (RAG)'\n"
            "Focus on:\n"
            "- Key innovations in the past year\n"
            "- Major challenges and solutions\n"
            "- Practical applications\n"
            "Provide a comprehensive research summary (300-400 words)."
        ),
        agent=researcher,
        expected_output="Comprehensive research summary on RAG advances",
    )
    print("   ✓ Research task created")

    analysis_task = Task(
        description=(
            "Analyze the research findings on RAG advances.\n"
            "Identify:\n"
            "- The 3 most significant innovations\n"
            "- Key trends and patterns\n"
            "- Implications for practitioners\n"
            "Provide structured analysis with clear conclusions (250-300 words)."
        ),
        agent=analyst,
        expected_output="Structured analysis with key insights and conclusions",
        context=[research_task],  # Depends on research task
    )
    print("   ✓ Analysis task created (depends on research)")

    writing_task = Task(
        description=(
            "Write a blog post based on the research and analysis.\n"
            "Structure:\n"
            "- Engaging introduction\n"
            "- Main findings (3 sections)\n"
            "- Practical implications\n"
            "- Conclusion\n"
            "Target audience: ML engineers and data scientists.\n"
            "Length: 400-500 words."
        ),
        agent=writer,
        expected_output="Well-structured blog post on RAG advances",
        context=[research_task, analysis_task],  # Depends on both previous tasks
    )
    print("   ✓ Writing task created (depends on research and analysis)")

    editing_task = Task(
        description=(
            "Review and edit the blog post for:\n"
            "- Technical accuracy\n"
            "- Clarity and readability\n"
            "- Consistency in terminology\n"
            "- Grammar and style\n"
            "Provide the final edited version with a brief summary of changes made."
        ),
        agent=editor,
        expected_output="Final edited blog post with change summary",
        context=[writing_task],  # Depends on writing task
    )
    print("   ✓ Editing task created (depends on writing)")

    # Create and execute crew
    print("\n5. Creating crew with task dependencies...")
    crew = Crew(
        agents=[researcher, analyst, writer, editor],
        tasks=[research_task, analysis_task, writing_task, editing_task],
        verbose=True,
    )
    print("   ✓ Crew created with 4 agents and 4 sequential tasks")

    print("\n" + "=" * 60)
    print("Executing Multi-Agent Crew...")
    print("=" * 60 + "\n")
    print("Task Flow:")
    print("  1. Researcher → Research RAG advances")
    print("  2. Analyst → Analyze findings (uses research)")
    print("  3. Writer → Write blog post (uses research + analysis)")
    print("  4. Editor → Edit and finalize (uses blog post)")
    print()

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Final Result:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # Show token usage for each agent
    print("\n6. Token usage by agent:")
    print(f"   Researcher: {researcher_llm.get_token_usage_summary().total_tokens} tokens")
    print(f"   Analyst: {analyst_llm.get_token_usage_summary().total_tokens} tokens")
    print(f"   Writer: {writer_llm.get_token_usage_summary().total_tokens} tokens")
    print(f"   Editor: {editor_llm.get_token_usage_summary().total_tokens} tokens")

    total_tokens = (
        researcher_llm.get_token_usage_summary().total_tokens
        + analyst_llm.get_token_usage_summary().total_tokens
        + writer_llm.get_token_usage_summary().total_tokens
        + editor_llm.get_token_usage_summary().total_tokens
    )
    print(f"   Total: {total_tokens} tokens")

    print("\n✓ Example completed successfully!")
    print("\nKey Takeaways:")
    print("  • Multiple agents can collaborate on complex tasks")
    print("  • Each agent can have different LLM configurations")
    print("  • Task dependencies ensure proper execution order")
    print("  • Requirements can be tailored to each agent's role")
    print("  • Different temperatures suit different agent roles")
    print("  • Context parameter enables information flow between tasks")


if __name__ == "__main__":
    main()

# Made with Bob
