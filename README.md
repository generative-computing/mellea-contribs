<img src="https://github.com/generative-computing/mellea-contribs/raw/main/mellea-contribs.jpg" height=100>


# Mellea Contribs

`mellea-contribs` is an incubation point for contributions to the [Mellea](https://github.com/generative-computing/mellea) ecosystem. It provides reusable validation/repair components, utility tools, and framework integrations for building LLM-powered applications with Mellea.

## Installation

```bash
pip install mellea-contribs
```

With robustness testing support:

```bash
pip install "mellea-contribs[robustness]"
```

Requires Python >= 3.10.

## Requirements Library (`reqlib`)

Reusable validation and repair components for use with Mellea m-programs:

| Component | Description |
|---|---|
| `citation_exists` | Validates that LLM-generated legal citations exist in a case database |
| `import_repair` | Validates Python imports in LLM-generated code and provides repair feedback |
| `import_resolution` | Helpers for resolving undefined names and module import errors |
| `is_appellate_case` | Determines whether a legal case is appellate |
| `grounding_context_formatter` | Formats multi-section grounding context for LLM prompts — [docs](docs/grounding_context_formatter.mdx) |

## Tools

| Tool | Description |
|---|---|
| `double_round_robin` | Selection strategy for sampling across candidates — [docs](docs/double_round_robin.mdx) |
| `top_k` | Top-K filtering utility — [docs](docs/top_k.mdx) |
| `benchdrift_runner` | Integration with [BenchDrift](https://github.com/IBM/BenchDrift) for robustness testing — [docs](docs/ROBUSTNESS_TESTING.md) |

## Framework Integrations

Mellea backends for popular LLM orchestration frameworks. Each integration lives in `integrations/` with its own getting-started guide.

| Integration | Package | Guide |
|---|---|---|
| **DSPy** | `mellea-dspy` | [Getting Started](integrations/dspy/GETTING_STARTED.md) |
| **LangChain** | `mellea-langchain` | [Getting Started](integrations/langchain/GETTING_STARTED.md) |
| **CrewAI** | `mellea-crewai` | [Getting Started](integrations/crewai/GETTING_STARTED.md) |

### DSPy

Use Mellea as a DSPy `LM` backend with support for requirements validation, sampling strategies (BestOfN, Refine), and async operations:

```python
import dspy
from mellea import start_session
from mellea_dspy import MelleaLM

m = start_session()
lm = MelleaLM(mellea_session=m, model="mellea-ollama")
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")
response = qa(question="What is generative programming?")
```

### LangChain

Use Mellea as a LangChain `ChatModel` with full LCEL support:

```python
from mellea import start_session
from mellea_langchain import MelleaChatModel
from langchain_core.messages import HumanMessage

m = start_session()
chat_model = MelleaChatModel(mellea_session=m)
response = chat_model.invoke([HumanMessage(content="Hello!")])
```

### CrewAI

Use Mellea as a CrewAI LLM provider with requirements-based validation and task guardrails:

```python
from mellea import start_session
from mellea_crewai import MelleaLLM
from crewai import Agent, Task, Crew

m = start_session()
llm = MelleaLLM(mellea_session=m)

agent = Agent(role="Research Assistant", goal="Find information", llm=llm)
task = Task(description="Summarize generative programming", agent=agent, expected_output="Summary")
result = Crew(agents=[agent], tasks=[task]).kickoff()
```

## Contributing

Add your name to the `authors` list in `pyproject.toml` when contributing. See the dev dependency group for tooling setup (`ruff`, `mypy`, `pytest`).

```bash
uv sync --group dev
```
