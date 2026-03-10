# Getting Started with Mellea Chat Model for LangChain

This guide will help you get started with using Mellea as a chat model in LangChain applications.

## Installation

### Prerequisites

- Python >= 3.10
- Mellea >= 0.3.0
- LangChain Core >= 0.3.0

### Install from Source

```bash
# Clone the repository
cd langchain

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# Install optional dependencies for development
pip install -e ".[dev]"
```

### Install Dependencies

```bash
# Install Mellea
pip install mellea

# Install LangChain
pip install langchain-core langchain

# Install a Mellea backend (e.g., for Ollama)
# Mellea comes with Ollama support by default
```

## Quick Start

### 1. Basic Chat Completion

```python
from mellea import start_session
from mellea_langchain import MelleaChatModel
from langchain_core.messages import HumanMessage

# Create Mellea session (uses Ollama by default)
m = start_session()

# Create LangChain chat model
chat_model = MelleaChatModel(mellea_session=m)

# Use it like any LangChain chat model
response = chat_model.invoke([
    HumanMessage(content="What is generative programming?")
])

print(response.content)
```

### 2. Using Different Mellea Backends

```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea_langchain import MelleaChatModel

# Use OpenAI backend
backend = OpenAIBackend(model_id="gpt-4")
m = MelleaSession(backend=backend)
chat_model = MelleaChatModel(mellea_session=m)

# Use the same LangChain interface
response = chat_model.invoke([HumanMessage(content="Hello!")])
```

### 3. Async Usage

```python
from langchain_core.messages import HumanMessage

# Async invoke
response = await chat_model.ainvoke([
    HumanMessage(content="Write a short poem")
])
print(response.content)

# Async batch processing
responses = await chat_model.abatch([
    [HumanMessage(content="What is 2+2?")],
    [HumanMessage(content="Count to 5")]
])
```

**Note**: Streaming is not currently supported. The `stream()` and `astream()` methods will return the full response as a single chunk.

### 4. Using with LangChain Chains

```python
from langchain.prompts import ChatPromptTemplate

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

# Create a chain using LCEL
chain = prompt | chat_model

# Run the chain
result = chain.invoke({"input": "Explain quantum computing"})
print(result.content)
```

### 5. Multi-turn Conversations

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="My name is Alice."),
]

# First turn
response1 = chat_model.invoke(messages)
print(f"Assistant: {response1.content}")

# Add response and continue conversation
messages.append(response1)
messages.append(HumanMessage(content="What's my name?"))

# Second turn
response2 = chat_model.invoke(messages)
print(f"Assistant: {response2.content}")
```

## Running Examples

The `examples/` directory contains several working examples:

```bash
# Basic usage
python examples/basic_usage.py

# Sync and async usage
python examples/async_example.py

# LangChain chains
python examples/langchain_chain.py
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mellea_langchain

# Run specific test file
pytest tests/test_chat_model.py -v
```

## Configuration

### Model Options

You can pass model options through kwargs:

```python
response = chat_model.invoke(
    [HumanMessage(content="Hello")],
    model_options={
        "temperature": 0.7,
        "max_tokens": 100,
    }
)
```

### Async by Default

```python
# Use async methods for better performance
response = await chat_model.ainvoke([
    HumanMessage(content="Your question here")
])
```

### Custom Model Name

```python
# Set a custom model name for identification
chat_model = MelleaChatModel(
    mellea_session=m,
    model_name="my-custom-mellea"
)
```

## Common Use Cases

### 1. Development with Ollama, Production with OpenAI

```python
import os

# Choose backend based on environment
if os.getenv("ENV") == "production":
    from mellea.backends.openai import OpenAIBackend
    backend = OpenAIBackend(model_id="gpt-4")
else:
    from mellea.backends.ollama import OllamaModelBackend
    backend = OllamaModelBackend(model_id="llama2")

m = MelleaSession(backend=backend)
chat_model = MelleaChatModel(mellea_session=m)

# Same code works in both environments
```

### 2. Leveraging Mellea's Validation

```python
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Configure Mellea with validation
m = MelleaSession(backend=backend)

# Use with LangChain while benefiting from Mellea's features
chat_model = MelleaChatModel(mellea_session=m)
```

### 3. Building LangChain Applications

```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Build complex chains
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("user", "{question}")
])

chain = prompt | chat_model

# Use in your application
result = chain.invoke({
    "domain": "machine learning",
    "question": "Explain gradient descent"
})
```

## Troubleshooting

### Import Errors

If you get import errors, make sure all dependencies are installed:

```bash
pip install mellea langchain-core langchain pydantic
```

### Ollama Not Running

If using Ollama backend, ensure Ollama is running:

```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull llama2
```

### Async Event Loop Issues

If you encounter event loop issues in Jupyter notebooks:

```python
import nest_asyncio
nest_asyncio.apply()
```

## Next Steps

- Explore the [examples/](examples/) directory for more use cases
- Read the [Technical Specification](TECHNICAL_SPECIFICATION.md) for implementation details
- Check the [Implementation Plan](IMPLEMENTATION_PLAN.md) for roadmap and features
- Join the [Mellea Discord](https://ibm.biz/mellea-discord) for support

## Additional Resources

- [Mellea Documentation](https://mellea.ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [GitHub Repository](https://github.com/generative-computing/mellea-adapter)

## Support

If you encounter issues:

1. Check the [examples/](examples/) directory
2. Review the [Technical Specification](TECHNICAL_SPECIFICATION.md)
3. Open an issue on [GitHub](https://github.com/generative-computing/mellea-adapter/issues)
4. Ask on [Discord](https://ibm.biz/mellea-discord)