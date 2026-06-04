# Mellea Contribs — CrewAI

A CrewAI-compatible LLM implementation that wraps Mellea, enabling CrewAI agents to use Mellea's generative programming capabilities including requirements, validation, and sampling strategies.

## Features

- Standard synchronous and asynchronous LLM calls (`call` / `acall`)
- Mellea requirements and sampling strategies wired through CrewAI agents
- Convert Mellea requirements into CrewAI task guardrails
- Tool calling with CrewAI agents
- Backend-agnostic (Ollama, OpenAI, WatsonX, HuggingFace, …)
- Full CrewAI event-bus integration and token-usage tracking

## Installation

```bash
pip install mellea-contribs-crewai
```

## Quick Start

```python
from crewai import Agent, Crew, Task
from mellea import start_session
from mellea_contribs.crewai import MelleaLLM

m = start_session()
llm = MelleaLLM(mellea_session=m)

agent = Agent(
    role="Researcher",
    goal="Research AI trends thoroughly",
    backstory="You are an expert AI researcher",
    llm=llm,
)

task = Task(
    description="Research the latest trends in generative AI",
    agent=agent,
    expected_output="A comprehensive report on AI trends",
)

crew = Crew(agents=[agent], tasks=[task])
print(crew.kickoff())
```

## Requirements and Validation

```python
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea_contribs.crewai import MelleaLLM

llm = MelleaLLM(
    mellea_session=m,
    requirements=[
        req("The response should be professional"),
        req("Include specific examples"),
        check("Do not mention competitors"),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=5),
)
```

## Task Guardrails

```python
from mellea.stdlib.requirements import simple_validate
from mellea_contribs.crewai import create_guardrail, create_guardrails

word_count_req = simple_validate(
    lambda x: 50 <= len(x.split()) <= 150,
    "Must be between 50-150 words",
)
guardrail = create_guardrail(word_count_req)
```

## Public API

```python
from mellea_contribs.crewai import (
    MelleaLLM,
    CrewAIMessageConverter,
    CrewAIToolConverter,
    create_guardrail,
    create_guardrails,
)
```

## Layout

```
crewai/
├── pyproject.toml
├── README.md
├── OWNERS
├── tests/
├── examples/
└── mellea_contribs/crewai/
    ├── __init__.py
    ├── core/                # llm, message_conversion, tool_conversion, validators
    ├── backends/
    ├── formatters/
    ├── stdlib/
    └── helpers/
```

## Development

```bash
cd crewai
uv sync --all-extras
uv run pytest -m "not qualitative and not e2e"
```

## License

Apache-2.0.
