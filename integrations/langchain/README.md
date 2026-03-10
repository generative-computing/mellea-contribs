# Mellea Chat Model for LangChain

A LangChain-compatible chat model that wraps Mellea, enabling LangChain applications to use Mellea's generative programming capabilities as a standard chat model.

## Overview

This integration allows LangChain users to leverage Mellea's structured approach to generative programming while using familiar LangChain APIs. It provides full support for:

- ✅ **Chat Completion**: Standard synchronous and asynchronous generation
- ✅ **Requirements & Validation**: Mellea's requirements and sampling strategies
- ✅ **Tool Calling**: Function calling with LangChain agents
- ✅ **Chains**: Integration with LangChain chains
- ✅ **Agents**: Support for LangChain agents

**Note**: Streaming is not currently supported. The `stream()` and `astream()` methods will return the full response as a single chunk.

## Quick Start

### Installation

```bash
# Install from PyPI (once published)
pip install mellea-langchain

# Or install from source
cd langchain
pip install -e .
```

### Basic Usage

```python
from mellea import start_session
from mellea_langchain import MelleaChatModel
from langchain_core.messages import HumanMessage

# Create Mellea session (with any backend)
m = start_session()  # Uses Ollama by default

# Create LangChain chat model
chat_model = MelleaChatModel(mellea_session=m)

# Use like any LangChain chat model
response = chat_model.invoke([
    HumanMessage(content="What is generative programming?")
])
print(response.content)
```

### Using with Different Mellea Backends

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.backends.ollama import OllamaModelBackend
from mellea_langchain import MelleaChatModel

# Option 1: OpenAI backend
openai_backend = OpenAIBackend(model_id="gpt-4")
m = MelleaSession(backend=openai_backend)
chat_model = MelleaChatModel(mellea_session=m)

# Option 2: Ollama backend
ollama_backend = OllamaModelBackend(model_id="llama2")
m = MelleaSession(backend=ollama_backend)
chat_model = MelleaChatModel(mellea_session=m)

# Use the same LangChain interface regardless of backend
response = chat_model.invoke([HumanMessage(content="Hello!")])
```

### Async Example

```python
from langchain_core.messages import HumanMessage

# Async invoke
response = await chat_model.ainvoke([
    HumanMessage(content="Write a short story about AI")
])
print(response.content)
```

### Using with LangChain Chains

```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Create a chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains technical concepts."),
    ("user", "{input}")
])

chain = prompt | chat_model

# Run the chain
result = chain.invoke({"input": "What is generative programming?"})
print(result.content)
```

### Using Requirements and Validation

Mellea's requirements and sampling strategies ensure LLM outputs meet specific criteria:

```python
from mellea.stdlib.requirements import req, check
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Define requirements
requirements = [
    req("The email should have a salutation"),
    req("Use only lower-case letters"),
    check("Do not mention purple elephants"),
]

# Use with strategy
response = chat_model.invoke(
    [HumanMessage(content="Write an email to John about the meeting")],
    requirements=requirements,
    strategy=RejectionSamplingStrategy(loop_budget=5),
    return_sampling_results=True,
)

# Access validation results
if response.success:
    print("Requirements met:", response.result.content)
else:
    print("Requirements not met, using best attempt")
```

### Using with LangChain Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create a chain with requirements
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Write an email to {name} about: {topic}")
])

model_with_requirements = chat_model.bind(
    requirements=[
        req("The email should have a salutation"),
        req("The email should be professional"),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)

chain = prompt | model_with_requirements | StrOutputParser()
result = chain.invoke({"name": "Dr. Smith", "topic": "conference"})
print(result)
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│              LangChain Application                       │
│  (Chains, Agents, LangServe, etc.)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              MelleaChatModel                             │
│  (Implements BaseChatModel interface)                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • Message Conversion                            │   │
│  │  • Streaming Handler                             │   │
│  │  • Tool Call Handler                             │   │
│  │  • Callback Integration                          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 Mellea Session                           │
│  (Generative Programming Framework)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Mellea Backends                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  Ollama  │  │  OpenAI  │  │ WatsonX  │  ...        │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

### Key Features

#### 1. Seamless Integration

Works with any Mellea backend while providing a standard LangChain interface:

```python
# Same code works with different backends
backends = [
    OllamaModelBackend(model_id="llama2"),
    OpenAIBackend(model_id="gpt-4"),
    WatsonXBackend(model_id="granite-13b"),
]

for backend in backends:
    m = MelleaSession(backend=backend)
    chat_model = MelleaChatModel(mellea_session=m)
    # Use with any LangChain component
```

#### 2. Message Format Conversion

Automatically converts between LangChain and Mellea message formats:

```python
# LangChain → Mellea
HumanMessage(content="Hello") → Message(role="user", content="Hello")
AIMessage(content="Hi") → Message(role="assistant", content="Hi")
SystemMessage(content="...") → Message(role="system", content="...")
ToolMessage(...) → Message(role="tool", ...)
```

#### 3. Async Support

```python
# Async invoke
response = await chat_model.ainvoke([HumanMessage(content="Tell me a story")])
print(response.content)

# Async batch processing
responses = await chat_model.abatch([
    [HumanMessage(content="What is 2+2?")],
    [HumanMessage(content="What is the capital of France?")]
])
```

#### 4. Tool Calling for Agents

Automatic tool conversion and execution with LangChain agents:

```python
# Tools are automatically converted between formats
langchain_tools = [Tool(...), Tool(...)]
# MelleaChatModel handles the conversion internally
agent = create_tool_calling_agent(llm=chat_model, tools=langchain_tools)
```

#### 5. Callback Support

Full integration with LangChain's callback system:

```python
from langchain.callbacks import StdOutCallbackHandler

# Callbacks work automatically
response = chat_model.invoke(
    [HumanMessage(content="Hello")],
    config={"callbacks": [StdOutCallbackHandler()]}
)
```

## Project Structure

```
langchain/
├── README.md                          # This file
├── GETTING_STARTED.md                 # Quick start guide
├── pyproject.toml                     # Project configuration
├── src/
│   └── mellea_langchain/
│       ├── __init__.py               # Package exports
│       ├── chat_model.py             # MelleaChatModel class
│       ├── message_conversion.py     # Message format conversion
│       └── tool_conversion.py        # Tool calling utilities
├── tests/
│   ├── test_chat_model.py            # Chat model tests
│   ├── test_message_conversion.py    # Message conversion tests
│   └── test_tool_conversion.py       # Tool conversion tests
└── examples/
    ├── basic_usage.py                # Basic example
    ├── async_example.py              # Sync and async example
    ├── tools_example.py              # Tool calling example
    ├── langchain_chain.py            # Chain example
    └── requirements_strategy_example.py  # Requirements/validation example
```

## Features

### ✅ Implemented
- Chat completion (sync and async)
- Streaming (sync and async)
- Message conversion (all LangChain message types)
- Tool calling with LangChain agents
- Requirements and validation strategies
- Sampling results and retry logic
- LangChain chain integration
- Comprehensive test coverage

### 📋 Future Enhancements
- LangServe deployment examples
- Additional backend examples
- Performance benchmarks
- Advanced agent patterns

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd langchain

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mellea_langchain --cov-report=html

# Run specific test file
pytest tests/test_chat_model.py

# Run integration tests
pytest tests/integration/
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/mellea_langchain
```

## Documentation

- **[Getting Started Guide](GETTING_STARTED.md)**: Step-by-step setup and usage
- **Examples**: See the `examples/` directory for complete examples
- **Tests**: See the `tests/` directory for usage patterns

## Requirements

- Python >= 3.10
- mellea >= 0.3.0
- langchain-core >= 0.3.0
- pydantic >= 2.0.0

Optional dependencies:
- langchain >= 0.3.0 (for chains and agents)
- langserve >= 0.3.0 (for deployment)

## Use Cases

### 1. Structured Generative Programming in LangChain

Use Mellea's structured approach within LangChain applications:

```python
# Leverage Mellea's validation and sampling strategies
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = MelleaSession(backend=backend)
chat_model = MelleaChatModel(mellea_session=m)

# Use with LangChain while benefiting from Mellea's features
chain = prompt | chat_model
```

### 2. Multi-Backend LangChain Applications

Easily switch between different model providers:

```python
# Development: Use Ollama
dev_model = MelleaChatModel(
    mellea_session=MelleaSession(backend=OllamaModelBackend())
)

# Production: Use OpenAI
prod_model = MelleaChatModel(
    mellea_session=MelleaSession(backend=OpenAIBackend())
)

# Same chain works with both
chain = prompt | dev_model  # or prod_model
```

### 3. LangChain Agents with Mellea

Build agents that leverage Mellea's capabilities:

```python
# Create agent with Mellea backend
agent = create_tool_calling_agent(
    llm=MelleaChatModel(mellea_session=m),
    tools=tools,
    prompt=prompt
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

### Areas for Contribution

1. **Features**: Implement additional LangChain integrations
2. **Testing**: Add more integration tests and examples
3. **Documentation**: Improve docs and add tutorials
4. **Performance**: Optimize critical paths
5. **Examples**: Create more usage examples

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../LICENSE) file for details.

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/generative-computing/mellea-adapter/issues)
- **Discord**: Join the [Mellea Discord](https://ibm.biz/mellea-discord) for discussions
- **Documentation**: Visit [mellea.ai](https://mellea.ai/) for Mellea docs

## FAQ

### Q: Why use this instead of LangChain's built-in models?

**A**: This integration combines:
- Mellea's structured generative programming approach
- LangChain's extensive ecosystem and tooling
- Ability to use any Mellea backend with LangChain

### Q: Does this support all LangChain features?

**A**: It supports the core features: chat, tool calling, chains, agents, and LangServe. Note: Streaming is not currently supported - `stream()` and `astream()` methods return the full response as a single chunk.

### Q: Can I use Mellea's advanced features?

**A**: Yes! You configure your Mellea session with all desired features, then wrap it with `MelleaChatModel`.

### Q: What about performance?

**A**: The overhead is minimal. The wrapper primarily handles message format conversion.

### Q: Which Mellea backends are supported?

**A**: All Mellea backends work, including Ollama, OpenAI, WatsonX, HuggingFace, and custom backends.

## Examples

See the `examples/` directory for complete examples:

- [`basic_usage.py`](examples/basic_usage.py): Simple chat completion
- [`async_example.py`](examples/async_example.py): Synchronous and asynchronous usage
- [`tools_example.py`](examples/tools_example.py): Function calling with agents
- [`langchain_chain.py`](examples/langchain_chain.py): Using with chains
- [`requirements_strategy_example.py`](examples/requirements_strategy_example.py): Requirements and validation

## Acknowledgments

- **Mellea Team**: For creating the generative programming framework
- **LangChain Team**: For the comprehensive LLM application framework
- **Contributors**: Everyone who helps improve this integration