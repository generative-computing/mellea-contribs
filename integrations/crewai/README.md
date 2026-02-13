# Mellea LLM for CrewAI

A CrewAI-compatible LLM implementation that wraps Mellea, enabling CrewAI agents to use Mellea's generative programming capabilities including requirements, validation, and sampling strategies.

## Overview

This integration allows CrewAI users to leverage Mellea's structured approach to generative programming while using familiar CrewAI APIs. It provides full support for:

- ✅ **Standard LLM Calls**: Synchronous and asynchronous generation
- ✅ **Requirements & Validation**: Mellea's requirements and sampling strategies
- ✅ **Tool Calling**: Function calling with CrewAI agents
- ✅ **Multi-Backend**: Works with Ollama, OpenAI, WatsonX, and more
- ✅ **Event Integration**: Full CrewAI event bus support
- ✅ **Token Tracking**: Automatic usage metrics

## Quick Start

### Installation

```bash
# Install from PyPI (once published)
pip install mellea-crewai

# Or install from source
cd integrations/crewai
pip install -e .
```

### Basic Usage

```python
from mellea import start_session
from mellea_crewai import MelleaLLM
from crewai import Agent, Task, Crew

# Create Mellea session (with any backend)
m = start_session()  # Uses Ollama by default

# Create CrewAI LLM
llm = MelleaLLM(mellea_session=m)

# Use with CrewAI agents
agent = Agent(
    role="Researcher",
    goal="Research AI trends thoroughly",
    backstory="You are an expert AI researcher",
    llm=llm
)

task = Task(
    description="Research the latest trends in generative AI",
    agent=agent,
    expected_output="A comprehensive report on AI trends"
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
print(result)
```

### Using with Different Mellea Backends

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.backends.ollama import OllamaModelBackend
from mellea_crewai import MelleaLLM

# Option 1: OpenAI backend
openai_backend = OpenAIBackend(model_id="gpt-4")
m = MelleaSession(backend=openai_backend)
llm = MelleaLLM(mellea_session=m)

# Option 2: Ollama backend
ollama_backend = OllamaModelBackend(model_id="llama2")
m = MelleaSession(backend=ollama_backend)
llm = MelleaLLM(mellea_session=m)

# Use the same CrewAI code regardless of backend
agent = Agent(role="Assistant", goal="Help users", llm=llm)
```

## Mellea Unique Features

### Requirements and Validation

Mellea's requirements system ensures LLM outputs meet specific criteria:

```python
from mellea.stdlib.requirements import req, check
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Define requirements
llm = MelleaLLM(
    mellea_session=m,
    requirements=[
        req("The response should be professional"),
        req("Include specific examples"),
        check("Do not mention competitors")
    ],
    strategy=RejectionSamplingStrategy(loop_budget=5),
    return_sampling_results=True
)

# Use with CrewAI agents
agent = Agent(
    role="Professional Writer",
    goal="Write professional content",
    llm=llm
)
```

### Custom Validation Functions

```python
from mellea.stdlib.requirements import req, simple_validate

# Custom validation
llm = MelleaLLM(
    mellea_session=m,
    requirements=[
        req("Use only lowercase", 
            validation_fn=simple_validate(lambda x: x.lower() == x)),
        req("Length between 100-500 characters",
            validation_fn=simple_validate(lambda x: 100 <= len(x) <= 500))
    ]
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              CrewAI Application                          │
│  (Agents, Tasks, Crews, Tools)                          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              MelleaLLM (BaseLLM)                        │
│  • Message conversion (CrewAI ↔ Mellea)                │
│  • Tool call handling                                   │
│  • Event emission                                       │
│  • Requirements & validation                            │
│  • Sampling strategies                                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 Mellea Session                           │
│  (chat, instruct, achat, ainstruct)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Mellea Backends                             │
│  Ollama | OpenAI | WatsonX | HuggingFace | ...         │
└─────────────────────────────────────────────────────────┘
```

## Features

### Interface Mapping

| CrewAI Method | Mellea Method | Notes |
|---------------|---------------|-------|
| `call()` | `chat()` or `instruct()` | Uses `instruct()` when requirements provided |
| `acall()` | `achat()` or `ainstruct()` | Async version |
| `supports_stop_words()` | Post-processing | Handled via `_apply_stop_words()` |
| `get_context_window_size()` | Backend-specific | Queries Mellea backend |
| `supports_multimodal()` | Backend-specific | Queries Mellea backend |

### Message Format Conversion

Automatic conversion between CrewAI and Mellea message formats:

```python
# CrewAI → Mellea
{"role": "system", "content": "..."} → Message(role="system", content="...")
{"role": "user", "content": "..."} → Message(role="user", content="...")
{"role": "assistant", "content": "..."} → Message(role="assistant", content="...")
```

### Tool Calling

Automatic tool conversion and execution:

```python
from crewai.tools import Tool

# Define CrewAI tool
@Tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Tools are automatically converted to Mellea format
agent = Agent(
    role="Researcher",
    tools=[search_web],
    llm=llm
)
```

### Event Integration

Full integration with CrewAI's event bus:
- `LLMCallStartedEvent`: When LLM call begins
- `LLMCallCompletedEvent`: When LLM call succeeds
- `LLMCallFailedEvent`: When LLM call fails
- `ToolUsageStartedEvent`: When tool execution begins
- `ToolUsageFinishedEvent`: When tool execution completes
- `ToolUsageErrorEvent`: When tool execution fails

## Project Structure

```
integrations/crewai/
├── README.md                          # This file
├── INTEGRATION_PLAN.md                # Detailed integration plan
├── GETTING_STARTED.md                 # Step-by-step guide
├── pyproject.toml                     # Package configuration
├── src/
│   └── mellea_crewai/
│       ├── __init__.py               # Package exports
│       ├── llm.py                    # MelleaLLM class
│       ├── message_conversion.py     # Message format conversion
│       └── tool_conversion.py        # Tool calling utilities
├── tests/
│   ├── test_llm.py                   # Core LLM tests
│   ├── test_message_conversion.py    # Message conversion tests
│   ├── test_tool_conversion.py       # Tool conversion tests
│   ├── test_requirements.py          # Requirements feature tests
│   └── test_sampling.py              # Sampling strategy tests
└── examples/
    ├── basic_usage.py                # Simple agent example
    ├── agent_with_tools.py           # Tool calling example
    ├── requirements_example.py       # Requirements validation
    └── multi_agent_crew.py           # Multi-agent crew example
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd integrations/crewai

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mellea_crewai --cov-report=html

# Run specific test file
pytest tests/test_llm.py

# Run with specific markers
pytest -m ollama  # Only Ollama tests
pytest -m "not qualitative"  # Skip qualitative tests
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/mellea_crewai
```

## Requirements

- Python >= 3.10
- mellea >= 0.3.0
- crewai >= 0.1.0
- pydantic >= 2.0.0

Optional dependencies:
- pytest >= 7.0.0 (for testing)
- ruff >= 0.1.0 (for linting)
- mypy >= 1.0.0 (for type checking)

## Use Cases

### 1. Multi-Backend CrewAI Applications

Easily switch between different model providers:

```python
# Development: Use Ollama
dev_llm = MelleaLLM(
    mellea_session=MelleaSession(backend=OllamaModelBackend())
)

# Production: Use OpenAI
prod_llm = MelleaLLM(
    mellea_session=MelleaSession(backend=OpenAIBackend())
)

# Same agent code works with both
agent = Agent(role="Assistant", llm=dev_llm)  # or prod_llm
```

### 2. Validated Agent Outputs

Ensure agent outputs meet specific criteria:

```python
llm = MelleaLLM(
    mellea_session=m,
    requirements=[
        req("Response must be under 500 words"),
        req("Must include actionable recommendations"),
        check("Do not include personal opinions")
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3)
)

agent = Agent(role="Analyst", llm=llm)
```

### 3. Multi-Agent Crews with Different Backends

```python
# Researcher uses powerful model
researcher_llm = MelleaLLM(
    mellea_session=MelleaSession(backend=OpenAIBackend(model_id="gpt-4"))
)

# Writer uses local model
writer_llm = MelleaLLM(
    mellea_session=MelleaSession(backend=OllamaModelBackend(model_id="llama2"))
)

researcher = Agent(role="Researcher", llm=researcher_llm)
writer = Agent(role="Writer", llm=writer_llm)

crew = Crew(agents=[researcher, writer], tasks=[...])
```

## Examples

See the `examples/` directory for complete examples:

- [`basic_usage.py`](examples/basic_usage.py): Simple agent with Mellea
- [`agent_with_tools.py`](examples/agent_with_tools.py): Tool calling
- [`requirements_example.py`](examples/requirements_example.py): Requirements validation
- [`multi_agent_crew.py`](examples/multi_agent_crew.py): Multi-agent crew

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Areas for Contribution

1. **Features**: Additional CrewAI integrations
2. **Testing**: More integration tests and examples
3. **Documentation**: Improve docs and add tutorials
4. **Performance**: Optimize critical paths
5. **Examples**: Create more usage examples

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../LICENSE) file for details.

## Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/generative-computing/mellea/issues)
- **Discord**: Join the [Mellea Discord](https://ibm.biz/mellea-discord)
- **Documentation**: Visit [mellea.ai](https://mellea.ai/)

## FAQ

### Q: Why use this instead of CrewAI's built-in LLMs?

**A**: This integration combines:
- Mellea's structured generative programming approach
- CrewAI's agent framework and tooling
- Ability to use any Mellea backend with CrewAI
- Requirements and validation capabilities

### Q: Does this support all CrewAI features?

**A**: It supports the core features: agents, tasks, crews, tools, and events. Some advanced features may have limitations.

### Q: Can I use Mellea's advanced features?

**A**: Yes! Configure your Mellea session with all desired features, then wrap it with `MelleaLLM`.

### Q: What about performance?

**A**: The overhead is minimal. The wrapper primarily handles message format conversion and event emission.

### Q: Which Mellea backends are supported?

**A**: All Mellea backends work, including Ollama, OpenAI, WatsonX, HuggingFace, and custom backends.

## Acknowledgments

- **Mellea Team**: For creating the generative programming framework
- **CrewAI Team**: For the powerful agent framework
- **Contributors**: Everyone who helps improve this integration

---

**Version**: 0.1.0  
**Status**: Alpha  
**Last Updated**: 2026-02-12