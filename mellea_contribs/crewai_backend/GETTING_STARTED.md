# Getting Started with Mellea-CrewAI Integration

This guide will help you get started with using Mellea as an LLM provider for CrewAI.

## Prerequisites

- Python 3.10 or higher
- Ollama installed and running (or access to another LLM backend)
- Basic familiarity with CrewAI and Mellea

## Installation

### Option 1: Install from PyPI (when published)

```bash
pip install mellea-crewai
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/generative-computing/mellea.git
cd mellea/integrations/crewai

# Install in development mode
pip install -e .
```

### Option 3: Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Start Ollama (if using local backend)

```bash
ollama serve
```

### 2. Create Your First Agent

Create a file `my_first_agent.py`:

```python
from mellea import start_session
from mellea_crewai import MelleaLLM
from crewai import Agent, Task, Crew

# Create Mellea session
m = start_session()  # Uses Ollama by default

# Create CrewAI LLM
llm = MelleaLLM(mellea_session=m)

# Create an agent
agent = Agent(
    role="Research Assistant",
    goal="Help users find information",
    backstory="You are a helpful research assistant",
    llm=llm
)

# Create a task
task = Task(
    description="Research the benefits of generative programming",
    agent=agent,
    expected_output="A summary of key benefits"
)

# Create and run crew
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()

print(result)
```

### 3. Run Your Agent

```bash
python my_first_agent.py
```

## Using Different Backends

### OpenAI Backend

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea_crewai import MelleaLLM

# Configure OpenAI backend
backend = OpenAIBackend(
    model_id="gpt-4",
    api_key="your-api-key"  # Or set OPENAI_API_KEY env var
)

# Create session and LLM
m = MelleaSession(backend=backend)
llm = MelleaLLM(mellea_session=m)

# Use with CrewAI as before
```

### WatsonX Backend

```python
from mellea import MelleaSession
from mellea.backends.watsonx import WatsonXBackend
from mellea_crewai import MelleaLLM

backend = WatsonXBackend(
    model_id="ibm/granite-13b-chat-v2",
    api_key="your-api-key",
    project_id="your-project-id"
)

m = MelleaSession(backend=backend)
llm = MelleaLLM(mellea_session=m)
```

### HuggingFace Backend

```python
from mellea import MelleaSession
from mellea.backends.huggingface import HuggingFaceBackend
from mellea_crewai import MelleaLLM

backend = HuggingFaceBackend(
    model_id="meta-llama/Llama-2-7b-chat-hf"
)

m = MelleaSession(backend=backend)
llm = MelleaLLM(mellea_session=m)
```

## Using Requirements and Validation

One of Mellea's unique features is requirements-based validation:

```python
from mellea import start_session
from mellea.stdlib.requirements import req, check
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea_crewai import MelleaLLM

# Create session
m = start_session()

# Define requirements
requirements = [
    req("The response should be professional"),
    req("Include specific examples"),
    req("Keep it under 300 words"),
    check("Do not use jargon")
]

# Create LLM with validation
llm = MelleaLLM(
    mellea_session=m,
    requirements=requirements,
    strategy=RejectionSamplingStrategy(loop_budget=5),
    return_sampling_results=True
)

# Use with CrewAI - outputs will be validated!
agent = Agent(
    role="Professional Writer",
    goal="Write clear, professional content",
    llm=llm
)
```

## Custom Validation Functions

You can create custom validation logic:

```python
from mellea.stdlib.requirements import req, simple_validate

def validate_length(text: str) -> bool:
    """Validate text is between 100-500 characters."""
    return 100 <= len(text) <= 500

def validate_no_urls(text: str) -> bool:
    """Validate text contains no URLs."""
    return "http://" not in text and "https://" not in text

requirements = [
    req("Length between 100-500 chars", 
        validation_fn=simple_validate(validate_length)),
    req("No URLs allowed",
        validation_fn=simple_validate(validate_no_urls))
]

llm = MelleaLLM(
    mellea_session=m,
    requirements=requirements,
    strategy=RejectionSamplingStrategy(loop_budget=3)
)

## Using Task Guardrails

CrewAI supports guardrails for task output validation. You can convert Mellea requirements into CrewAI-compatible guardrails:

### Basic Guardrail Usage

```python
from mellea.stdlib.requirements import simple_validate
from mellea_crewai import create_guardrail, MelleaLLM
from crewai import Agent, Task, Crew

# Create Mellea session
m = start_session()
llm = MelleaLLM(mellea_session=m)

# Create a Mellea requirement
word_count_req = simple_validate(
    lambda x: 50 <= len(x.split()) <= 150,
    "Must be between 50-150 words"
)

# Convert to CrewAI guardrail
guardrail = create_guardrail(word_count_req)

# Create agent and task with guardrail
agent = Agent(
    role="Writer",
    goal="Write concise content",
    llm=llm
)

task = Task(
    description="Write a summary about AI",
    expected_output="Brief AI summary",
    agent=agent,
    guardrails=[guardrail],
    guardrail_max_retries=3  # Retry up to 3 times if validation fails
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

### Multiple Guardrails

Apply multiple validation checks sequentially:

```python
from mellea_crewai import create_guardrails

# Define multiple requirements
requirements = [
    simple_validate(
        lambda x: 50 <= len(x.split()) <= 150,
        "Must be between 50-150 words"
    ),
    simple_validate(
        lambda x: "AI" in x or "artificial intelligence" in x.lower(),
        "Must mention AI"
    ),
    simple_validate(
        lambda x: x.strip() == x,
        "No extra whitespace"
    ),
]

# Convert all to guardrails
guardrails = create_guardrails(requirements)

# Use in task
task = Task(
    description="Write about AI",
    expected_output="AI article",
    agent=agent,
    guardrails=guardrails,
    guardrail_max_retries=3
)
```

### Using Helper Functions

Mellea-CrewAI provides helper functions for common validation patterns:

```python
from mellea_crewai import (
    word_count_guardrail,
    contains_keywords_guardrail,
    no_profanity_guardrail,
    format_output_guardrail,
)

task = Task(
    description="Write a professional blog post about AI",
    expected_output="Professional blog post",
    agent=agent,
    guardrails=[
        word_count_guardrail(min_words=100, max_words=500),
        contains_keywords_guardrail(["AI", "machine learning"]),
        no_profanity_guardrail(),
        format_output_guardrail(strip_whitespace=True, capitalize_first=True),
    ],
    guardrail_max_retries=3
)
```

### Mixing Mellea Requirements with Custom Guardrails

You can combine Mellea requirements with custom CrewAI guardrails:

```python
from typing import Tuple, Any
from crewai import TaskOutput

# Mellea requirement
length_req = simple_validate(lambda x: len(x) > 100, "At least 100 characters")

# Custom CrewAI guardrail
def custom_format_check(result: TaskOutput) -> Tuple[bool, Any]:
    """Custom formatting validation."""
    text = result.raw
    if not text[0].isupper():
        return (False, "Must start with capital letter")
    if not text.endswith("."):
        return (False, "Must end with period")
    return (True, text)

# Combine both
task = Task(
    description="Write content",
    expected_output="Formatted content",
    agent=agent,
    guardrails=[
        create_guardrail(length_req),  # From Mellea
        custom_format_check,            # Custom
    ],
    guardrail_max_retries=3
)
```

### Available Helper Functions

- **`word_count_guardrail(min_words, max_words)`**: Validate word count
  ```python
  word_count_guardrail(min_words=50, max_words=200)
  ```

- **`contains_keywords_guardrail(keywords, case_sensitive=False)`**: Check for required keywords
  ```python
  contains_keywords_guardrail(["AI", "machine learning"], case_sensitive=False)
  ```

- **`no_profanity_guardrail(profanity_list=None)`**: Filter inappropriate content
  ```python
  no_profanity_guardrail(["badword1", "badword2"])
  ```

- **`format_output_guardrail(strip_whitespace=True, capitalize_first=True)`**: Format output
  ```python
  format_output_guardrail(strip_whitespace=True, capitalize_first=True)
  ```

See [`examples/validator_usage_example.py`](examples/validator_usage_example.py) for complete examples.
```

## Working with Tools

CrewAI tools work seamlessly with Mellea:

```python
from crewai.tools import tool
from crewai import Agent

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Your search implementation
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Create agent with tools
agent = Agent(
    role="Research Assistant",
    goal="Help with research and calculations",
    tools=[search_web, calculate],
    llm=llm
)
```

## Multi-Agent Crews

Create crews with multiple agents using different configurations:

```python
# Researcher with powerful model
researcher_llm = MelleaLLM(
    mellea_session=MelleaSession(
        backend=OpenAIBackend(model_id="gpt-4")
    )
)

researcher = Agent(
    role="Senior Researcher",
    goal="Conduct thorough research",
    llm=researcher_llm
)

# Writer with local model
writer_llm = MelleaLLM(
    mellea_session=start_session(),  # Ollama
    requirements=[
        req("Professional tone"),
        req("Clear structure")
    ]
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging content",
    llm=writer_llm
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process="sequential"
)
```

## Monitoring and Debugging

### Token Usage

Track token usage across your crew:

```python
result = crew.kickoff()

# Check token usage
usage = llm.get_token_usage_summary()
print(f"Total tokens: {usage.total_tokens}")
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Requests: {usage.successful_requests}")
```

### Verbose Mode

Enable verbose mode to see what's happening:

```python
agent = Agent(
    role="Assistant",
    goal="Help users",
    llm=llm,
    verbose=True  # Enable verbose output
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True  # Enable crew verbose output
)
```

### Validation Results

When using requirements, check validation results:

```python
llm = MelleaLLM(
    mellea_session=m,
    requirements=requirements,
    return_sampling_results=True  # Enable result tracking
)

# After execution, check if requirements were met
# (This information is logged during execution)
```

## Common Patterns

### Pattern 1: Development vs Production

```python
import os

# Use different backends for dev/prod
if os.getenv("ENVIRONMENT") == "production":
    backend = OpenAIBackend(model_id="gpt-4")
else:
    backend = OllamaModelBackend(model_id="llama2")

m = MelleaSession(backend=backend)
llm = MelleaLLM(mellea_session=m)
```

### Pattern 2: Conditional Requirements

```python
def create_llm(strict_mode: bool = False):
    requirements = [
        req("Be helpful and accurate")
    ]
    
    if strict_mode:
        requirements.extend([
            req("Cite sources"),
            req("Avoid speculation"),
            check("No personal opinions")
        ])
    
    return MelleaLLM(
        mellea_session=m,
        requirements=requirements if strict_mode else None
    )
```

### Pattern 3: Shared Session, Multiple LLMs

```python
# Create one session
m = start_session()

# Create multiple LLMs with different configs
creative_llm = MelleaLLM(mellea_session=m, temperature=0.9)
precise_llm = MelleaLLM(mellea_session=m, temperature=0.1)
validated_llm = MelleaLLM(
    mellea_session=m,
    requirements=[req("Be accurate")]
)

# Use with different agents
creative_agent = Agent(role="Creative Writer", llm=creative_llm)
analyst_agent = Agent(role="Data Analyst", llm=precise_llm)
reviewer_agent = Agent(role="Reviewer", llm=validated_llm)
```

## Troubleshooting

### Issue: "Import mellea_crewai could not be resolved"

**Solution**: Make sure you've installed the package:
```bash
pip install -e .
```

### Issue: "Ollama connection refused"

**Solution**: Start Ollama:
```bash
ollama serve
```

### Issue: "Requirements not being validated"

**Solution**: Make sure you're passing requirements to MelleaLLM:
```python
llm = MelleaLLM(
    mellea_session=m,
    requirements=[...],  # Don't forget this!
    strategy=RejectionSamplingStrategy(loop_budget=5)
)
```

### Issue: "Tool calls not working"

**Solution**: Ensure tools are properly defined and passed to the agent:
```python
@tool
def my_tool(arg: str) -> str:
    """Tool description."""  # Description is important!
    return result

agent = Agent(
    role="Assistant",
    tools=[my_tool],  # Pass tools here
    llm=llm
)
```

## Next Steps

- Explore the [examples/](examples/) directory for more examples
- Read the [Integration Plan](INTEGRATION_PLAN.md) for technical details
- Check out [Mellea documentation](https://mellea.ai/) for advanced features
- Join the [Mellea Discord](https://ibm.biz/mellea-discord) for support

## Additional Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [Mellea Documentation](https://mellea.ai/)
- [Example Code](examples/)
- [Test Suite](tests/)

---

**Need Help?** Open an issue on [GitHub](https://github.com/generative-computing/mellea/issues) or join our [Discord](https://ibm.biz/mellea-discord).