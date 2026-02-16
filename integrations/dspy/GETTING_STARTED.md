# Getting Started with Mellea DSPy Integration

This guide will help you get started with using Mellea as a backend for DSPy.

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- [Ollama](https://ollama.com/) installed and running (for local models)

## Installation

### 1. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install -e .
```

### 2. Set Up Ollama

Install Ollama from [https://ollama.com/](https://ollama.com/) and pull a model:

```bash
# Pull IBM Granite model (recommended)
ollama pull granite4:micro

# Or use another model
ollama pull llama3.2:3b
```

## Quick Start

### Basic Usage

```python
import dspy
from mellea import start_session
from mellea_dspy import MelleaLM

# Create Mellea session
m = start_session()

# Create DSPy LM with Mellea backend
lm = MelleaLM(mellea_session=m, model="mellea-ollama")

# Configure DSPy
dspy.configure(lm=lm)

# Use with DSPy
qa = dspy.Predict("question -> answer")
response = qa(question="What is generative programming?")
print(response.answer)
```

### Chain of Thought

```python
# Use chain of thought for reasoning tasks
cot = dspy.ChainOfThought("question -> answer")
response = cot(question="Why is structured prompting better than ad-hoc prompts?")

print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")
```

### Custom Signatures

```python
class Summarize(dspy.Signature):
    """Summarize text into a concise summary."""
    text = dspy.InputField(desc="The text to summarize")
    summary = dspy.OutputField(desc="A concise summary")

summarizer = dspy.Predict(Summarize)
result = summarizer(text="Your long text here...")
print(result.summary)
```

## Running Examples

The `examples/` directory contains comprehensive examples:

```bash
# Basic usage patterns
uv run examples/01_basic_usage.py

# Requirements validation
uv run examples/02_requirements_validation.py

# Sampling strategies
uv run examples/03_sampling_strategies.py

# Async operations
uv run examples/04_async_operations.py

# DSPy modules
uv run examples/05_dspy_modules.py

# Optimization
uv run examples/06_optimization.py
```

## Configuration Options

### Model Selection

```python
# Use different backends
lm_ollama = MelleaLM(mellea_session=m, model="mellea-ollama")
lm_openai = MelleaLM(mellea_session=m, model="mellea-openai")
lm_anthropic = MelleaLM(mellea_session=m, model="mellea-anthropic")
```

### Generation Parameters

```python
lm = MelleaLM(
    mellea_session=m,
    model="mellea-ollama",
    temperature=0.7,      # Control randomness (0.0-1.0)
    max_tokens=2000       # Maximum output length
)
```

### Mellea Session Options

```python
# Default (Ollama)
m = start_session()

# With specific backend
m = start_session(backend="openai")

# With custom model
m = start_session(
    backend="ollama",
    model="granite4:micro"
)
```

## Key Features

### 1. **Structured Prompting**
Use DSPy signatures to define clear input/output structures:

```python
class QA(dspy.Signature):
    """Answer questions accurately."""
    question = dspy.InputField()
    answer = dspy.OutputField()

qa = dspy.Predict(QA)
```

### 2. **Requirements Validation**
Ensure outputs meet specific criteria (via Mellea's instruct method):

```python
# Pass requirements through model_options
response = lm(
    prompt="Generate a summary",
    requirements=["Must be under 100 words"],
    model_options={}
)
```

### 3. **Sampling Strategies**
Improve output quality with inference-time scaling:

```python
# Temperature control
lm_creative = MelleaLM(m, temperature=1.0)  # More creative
lm_focused = MelleaLM(m, temperature=0.0)   # More deterministic
```

### 4. **Async Operations**
Process multiple requests concurrently:

```python
import asyncio

async def generate():
    response = await lm.aforward(
        prompt="Generate response",
        model_options={}
    )
    return response

result = asyncio.run(generate())
```

### 5. **Modular Programs**
Build reusable DSPy modules:

```python
class MyProgram(dspy.Module):
    def __init__(self):
        self.predictor = dspy.Predict("input -> output")
    
    def forward(self, input):
        return self.predictor(input=input)

program = MyProgram()
result = program(input="test")
```

## Testing

Run the test suite:

```bash
uv run tests/test_lm.py
```

## Common Use Cases

### Question Answering
```python
qa = dspy.ChainOfThought("question -> answer")
response = qa(question="What is Mellea?")
```

### Text Classification
```python
classifier = dspy.Predict("text -> category")
response = classifier(text="AI article content")
```

### Summarization
```python
summarizer = dspy.Predict("text -> summary")
response = summarizer(text="Long article...")
```

### Code Generation
```python
coder = dspy.Predict("task -> code")
response = coder(task="Write a factorial function")
```

## Troubleshooting

### Connection Errors

**Problem:** `Connection refused` error

**Solution:**
```bash
# Ensure Ollama is running
ollama serve
```

### Model Not Found

**Problem:** Model not available

**Solution:**
```bash
# Pull the model
ollama pull granite4:micro
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'mellea_dspy'`

**Solution:**
```bash
# Install in development mode
uv sync
# or
pip install -e .
```

### Slow Generation

**Problem:** Generation takes too long

**Solutions:**
- Use async operations for concurrent requests
- Use smaller models (e.g., `granite4:micro`)
- Reduce `max_tokens` parameter
- Batch similar requests together

## Next Steps

1. **Explore Examples**: Check out the `examples/` directory for comprehensive demonstrations
2. **Read Documentation**: See [README.md](README.md) for detailed information
3. **Build Programs**: Start creating your own DSPy programs with Mellea
4. **Optimize**: Use DSPy's optimization tools to improve performance

## Resources

- [Mellea Documentation](https://mellea.ai/)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Mellea GitHub](https://github.com/generative-computing/mellea)
- [Examples Directory](examples/README.md)

## Support

For issues or questions:
- Check the [examples](examples/) directory
- Review the [README](README.md)
- Open an issue on GitHub

---

**Happy coding with Mellea + DSPy! 🚀**