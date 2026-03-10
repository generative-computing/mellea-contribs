# Mellea Chat Model for LangChain

A LangChain-compatible chat model that wraps Mellea, enabling LangChain applications to use Mellea's generative programming capabilities as a standard chat model.

## Overview

This integration allows LangChain users to leverage Mellea's structured approach to generative programming while using familiar LangChain APIs. It provides full support for:

- ✅ **Chat Completion**: Standard synchronous and asynchronous generation
- ✅ **Streaming**: Real-time token-by-token generation
- ✅ **Requirements & Validation**: Mellea's requirements and sampling strategies
- ✅ **Tool Calling**: Function calling with LangChain agents
- ✅ **Chains**: Integration with LangChain chains
- ✅ **Agents**: Support for LangChain agents
- ✅ **Output Parsers & Guardrails**: Validate outputs with Mellea requirements

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

### Streaming Example

```python
from langchain_core.messages import HumanMessage

# Stream responses
for chunk in chat_model.stream([
    HumanMessage(content="Write a short story about AI")
]):
    print(chunk.content, end="", flush=True)
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
<<<<<<< HEAD
=======

## Output Parsers and Guardrails

The integration provides two components for validating LLM outputs using Mellea's requirements system:

### MelleaOutputParser

A LangChain `BaseOutputParser` that validates outputs using deterministic validation functions. Perfect for use in chains with automatic validation.

```python
from mellea_langchain import MelleaOutputParser
from mellea.stdlib.requirements import simple_validate

# Create parser with validation requirements
parser = MelleaOutputParser(
    requirements=[
        simple_validate(
            lambda x: len(x.split()) < 100,
            "Must be under 100 words"
        ),
        simple_validate(
            lambda x: "AI" in x,
            "Must mention AI"
        ),
    ]
)

# Use in a chain
chain = prompt | model | parser
result = chain.invoke({"topic": "artificial intelligence"})
```

**Key Features:**
- Stateless design (no session required)
- Fast, deterministic validation
- Raises `OutputParserException` on failure (strict mode)
- Returns text anyway in non-strict mode
- Provides format instructions for LLM

### MelleaGuardrail

An independent validation component that can be used standalone or composed with other guardrails.

```python
from mellea_langchain import MelleaGuardrail

# Create guardrail
guardrail = MelleaGuardrail(
    requirements=[
        simple_validate(lambda x: len(x) > 50, "At least 50 chars"),
        simple_validate(lambda x: len(x) < 500, "Under 500 chars"),
    ],
    name="length_check"
)

# Validate output
result = guardrail.validate(text)
if not result.passed:
    print(f"Validation failed: {result.errors}")
```

**Key Features:**
- Independent validation (not tied to chains)
- Returns detailed `ValidationResult`
- Composable with other guardrails
- Supports `&` operator for composition

### Composing Guardrails

```python
# Create specialized guardrails
tone_guard = MelleaGuardrail(
    requirements=[simple_validate(lambda x: x.islower(), "Lowercase only")]
)
length_guard = MelleaGuardrail(
    requirements=[simple_validate(lambda x: len(x) < 100, "Under 100 chars")]
)

# Compose using & operator
combined = tone_guard & length_guard

# Validate with combined requirements
result = combined.validate(text)
```

### Integration with OutputFixingParser

For automatic repair of validation failures:

```python
from langchain.output_parsers import OutputFixingParser

# Create base parser
base_parser = MelleaOutputParser(
    requirements=[
        simple_validate(lambda x: x.startswith("{"), "Must be JSON"),
    ]
)

# Wrap with OutputFixingParser for auto-repair
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=model
)

# Will automatically retry and fix on validation failure
chain = prompt | model | fixing_parser
result = chain.invoke({"topic": "AI"})
```

### LLM vs Deterministic Validation

**Deterministic Validation (Output Parser):**
- Use `simple_validate()` with custom functions
- Fast (<1ms), predictable
- No LLM calls required
- Perfect for format, length, pattern checks

**LLM Validation (During Generation):**
- Use `req()` or `check()` in `MelleaChatModel`
- Validates during generation
- Better for semantic requirements
- Uses Mellea's sampling strategies

```python
# Combine both approaches
model = MelleaChatModel(
    mellea_session=m,
    requirements=[req("Must be professional")]  # LLM validation
)

parser = MelleaOutputParser(
    requirements=[
        simple_validate(lambda x: len(x) < 500, "Under 500 chars")  # Deterministic
    ]
)

chain = prompt | model | parser  # Both validations applied
```

>>>>>>> aaec38e (adding missing changes)
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

#### 3. Full Streaming Support

```python
# Async streaming
async for chunk in chat_model.astream([HumanMessage(content="Tell me a story")]):
    print(chunk.content, end="")

# Sync streaming
for chunk in chat_model.stream([HumanMessage(content="Tell me a story")]):
    print(chunk.content, end="")
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
│       ├── tool_conversion.py        # Tool calling utilities
│       └── guardrails.py             # Output parsers and guardrails
├── tests/
│   ├── test_chat_model.py            # Chat model tests
│   ├── test_message_conversion.py    # Message conversion tests
│   ├── test_tool_conversion.py       # Tool conversion tests
│   └── test_guardrails.py            # Guardrails tests
└── examples/
    ├── basic_usage.py                # Basic example
    ├── streaming_example.py          # Streaming example
    ├── tools_example.py              # Tool calling example
    ├── langchain_chain.py            # Chain example
    ├── requirements_strategy_example.py  # Requirements/validation example
    ├── output_parser_basic.py        # Output parser examples
    └── guardrail_usage.py            # Guardrail examples
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
<<<<<<< HEAD
=======
- **Output parsers with requirement validation**
- **Guardrails for independent validation**
- **Requirement composition and reuse**
>>>>>>> aaec38e (adding missing changes)
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

**A**: It supports the core features: chat, streaming, tool calling, chains, agents, and LangServe. Some advanced features may have limitations.

### Q: Can I use Mellea's advanced features?

**A**: Yes! You configure your Mellea session with all desired features, then wrap it with `MelleaChatModel`.

### Q: What about performance?

**A**: The overhead is minimal. The wrapper primarily handles message format conversion.

### Q: Which Mellea backends are supported?

**A**: All Mellea backends work, including Ollama, OpenAI, WatsonX, HuggingFace, and custom backends.

## Examples

See the `examples/` directory for complete examples:

**Chat Model:**
- [`basic_usage.py`](examples/basic_usage.py): Simple chat completion
- [`streaming_example.py`](examples/streaming_example.py): Real-time streaming
- [`tools_example.py`](examples/tools_example.py): Function calling with agents
- [`langchain_chain.py`](examples/langchain_chain.py): Using with chains
- [`requirements_strategy_example.py`](examples/requirements_strategy_example.py): Requirements and validation

**Output Parsers & Guardrails:**
- [`output_parser_basic.py`](examples/output_parser_basic.py): Basic output parser usage
- [`guardrail_usage.py`](examples/guardrail_usage.py): Guardrail validation and composition

## Acknowledgments

- **Mellea Team**: For creating the generative programming framework
- **LangChain Team**: For the comprehensive LLM application framework
- **Contributors**: Everyone who helps improve this integration