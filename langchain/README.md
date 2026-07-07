# Mellea Chat Model for LangChain

A LangChain-compatible chat model that wraps Mellea, enabling LangChain applications to use Mellea's generative programming capabilities as a standard chat model.

## Overview

This integration allows LangChain users to leverage Mellea's structured approach to generative programming while using familiar LangChain APIs. It provides full support for:

- Chat Completion: Standard synchronous and asynchronous generation
- Requirements & Validation: Mellea's requirements and sampling strategies
- Tool Calling: Function calling with LangChain agents
- Chains: Integration with LangChain chains
- Agents: Support for LangChain agents
- Output Parsers & Guardrails: Validate outputs with Mellea requirements

**Note**: Streaming is not currently supported. The `stream()` and `astream()` methods will return the full response as a single chunk.

## Quick Start

### Installation

```bash
# Install from PyPI (once published)
pip install mellea-contribs-langchain

# Or install from source
cd langchain
pip install -e .
```

### Basic Usage

```python
from mellea import start_session
from mellea_contribs.langchain import MelleaChatModel
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
from mellea_contribs.langchain import MelleaChatModel

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

## Output Parsers and Guardrails

The integration provides two components for validating LLM outputs using Mellea's requirements system:

### MelleaOutputParser

A LangChain `BaseOutputParser` that validates outputs using deterministic validation functions. Perfect for use in chains with automatic validation.

```python
from mellea_contribs.langchain import MelleaOutputParser
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
from mellea_contribs.langchain import MelleaGuardrail

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

## Architecture

### Component Overview

```text
LangChain Application (Chains, Agents, LangServe, etc.)
    -> MelleaChatModel (BaseChatModel interface)
    -> Mellea Session (Generative Programming Framework)
    -> Mellea Backends (Ollama, OpenAI, WatsonX, ...)
```

### Key Features

#### 1. Seamless Integration

Works with any Mellea backend while providing a standard LangChain interface.

#### 2. Message Format Conversion

Automatically converts between LangChain and Mellea message formats.

#### 3. Async Support

```python
# Async invoke
response = await chat_model.ainvoke([HumanMessage(content="Tell me a story")])
```

#### 4. Tool Calling for Agents

Automatic tool conversion and execution with LangChain agents.

#### 5. Callback Support

Full integration with LangChain's callback system.

## Project Structure

```text
langchain/
├── README.md
├── pyproject.toml
├── mellea_contribs/
│   └── langchain/
│       ├── __init__.py
│       └── core/
│           ├── chat_model.py
│           ├── message_conversion.py
│           ├── tool_conversion.py
│           └── guardrails.py
├── tests/
└── examples/
```

## Features

### Implemented

- Chat completion (sync and async)
- Streaming (sync and async, single-chunk)
- Message conversion (all LangChain message types)
- Tool calling with LangChain agents
- Requirements and validation strategies
- Sampling results and retry logic
- LangChain chain integration
- Output parsers with requirement validation
- Guardrails for independent validation
- Requirement composition and reuse

## Development

### Running Tests

```bash
pytest
pytest tests/integration/
```

### Code Quality

```bash
ruff format .
ruff check .
mypy mellea_contribs/langchain
```

## Requirements

- Python >= 3.11
- mellea >= 0.3.2
- mellea-contribs-integration-core
- langchain-core >= 0.3.2
- pydantic >= 2.0.0

## License

This project is licensed under the Apache License 2.0.
