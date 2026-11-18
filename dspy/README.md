# Mellea + DSPy Integration

A powerful integration that combines [Mellea](https://mellea.ai/)'s generative programming capabilities with [DSPy](https://github.com/stanfordnlp/dspy)'s structured prompting framework.

## Overview

This integration enables you to use Mellea as a backend for DSPy, bringing together:

- **Mellea's Features:**
  - Multi-backend support (Ollama, OpenAI, WatsonX, LiteLLM, HuggingFace)
  - Requirements validation
  - Sampling strategies for inference-time scaling
  - Generative programming patterns

- **DSPy's Features:**
  - Structured prompting with signatures
  - Modular program composition
  - Automatic optimization
  - Chain of thought reasoning

## Quick Start

### Installation

```bash
pip install "mellea-contribs[dspy] @ https://github.com/generative-computing/mellea-contribs/releases/latest/download/mellea_contribs-py3-none-any.whl"

# Ensure Ollama is running
ollama pull granite4:micro
```

See the [contribs README](../README.md) for details.

### Basic Usage

```python
import dspy
from mellea import start_session
from mellea_contribs.dspy import MelleaLM

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

## Examples

We provide comprehensive examples demonstrating all aspects of the integration:

### [01. Basic Usage](examples/01_basic_usage.py)
Learn the fundamentals:
- Simple predictions
- Chain of thought reasoning
- Custom signatures
- Multiple output fields

```bash
uv run examples/01_basic_usage.py
```

### [02. Requirements Validation](examples/02_requirements_validation.py)
Ensure output quality:
- Length constraints
- Format requirements
- Content validation
- Combined requirements

```bash
uv run examples/02_requirements_validation.py
```

### [03. Sampling Strategies](examples/03_sampling_strategies.py)
Improve output quality:
- Rejection sampling
- Best-of-N sampling
- Temperature control
- Ensemble methods
- Adaptive sampling

```bash
uv run examples/03_sampling_strategies.py
```

### [04. Async Operations](examples/04_async_operations.py)
Optimize performance:
- Concurrent generation
- Batch processing
- Async chain of thought
- Error handling

```bash
uv run examples/04_async_operations.py
```

### [05. DSPy Modules](examples/05_dspy_modules.py)
Build modular programs:
- Simple modules
- Multi-step reasoning
- Module composition
- Conditional logic
- Stateful modules

```bash
uv run examples/05_dspy_modules.py
```

### [06. Optimization](examples/06_optimization.py)
Improve program performance:
- Few-shot learning
- Bootstrap optimization
- Signature optimization
- Metric-based optimization
- Hyperparameter tuning

```bash
uv run examples/06_optimization.py
```

### [07. Ollama with Granite](examples/07_ollama_granite.py)
Configure specific models:
- OllamaBackend setup
- Custom model configuration
- Backend vs model parameter

```bash
uv run examples/07_ollama_granite.py
```

### [08. BestOfN Verification](examples/08_bestofn_verification.py)
Runtime verification with BestOfN:
- Generate N candidates, select best
- Automatic requirement-to-reward conversion
- Multiple requirement types
- Custom callable requirements
- Combination strategies

```bash
uv run examples/08_bestofn_verification.py
```

### [09. Refine Verification](examples/09_refine_verification.py)

Iterative refinement with requirements:
- Iterative output improvement
- Requirement-guided refinement
- Quality-focused strategies
- Custom refinement criteria
- Comparison with BestOfN

```bash
uv run examples/09_refine_verification.py
```

### [10. Hybrid Approach](examples/10_hybrid_approach.py)

Two ways to use Mellea requirements with DSPy:
- High-level wrappers (MelleaBestOfN, MelleaRefine)
- Direct DSPy with create_reward_fn()
- Side-by-side comparisons
- When to use each approach

```bash
uv run examples/10_hybrid_approach.py
```

### Runtime Verification with BestOfN and Refine

Two approaches for using Mellea requirements with DSPy verification:

**Approach 1: High-Level Wrappers (Recommended)**

```python
from mellea_contribs.dspy import MelleaBestOfN, MelleaRefine

# BestOfN: Generate N candidates, select best
qa = dspy.Predict("question -> answer")
best_of_5 = MelleaBestOfN(
    module=qa,
    N=5,
    requirements=[
        "Must be one word",
        "Must be a proper noun"
    ],
    threshold=0.8
)
result = best_of_5(question="What is the capital of Belgium?")
# Returns: "Brussels"

# Refine: Iteratively improve output
refiner = MelleaRefine(
    module=qa,
    N=3,
    requirements=[
        "Must be under 50 words",
        "Must be professional"
    ],
    threshold=0.9
)
result = refiner(question="What is Python?")
# Iteratively refines until requirements are met
```

**Approach 2: Direct DSPy with create_reward_fn() (Advanced)**

```python
import dspy
from mellea_contribs.dspy import create_reward_fn

# Create reward function from requirements
reward_fn = create_reward_fn(
    requirements=["Must be one word", "Must be a proper noun"],
    strategy="average"
)

# Use with native DSPy
qa = dspy.Predict("question -> answer")
best_of_5 = dspy.BestOfN(
    module=qa,
    N=5,
    reward_fn=reward_fn,
    threshold=0.8
)
result = best_of_5(question="What is the capital of Belgium?")
```

**When to Use Each Approach:**
- **Wrappers**: Simple API, Mellea-native, recommended for most users
- **Direct DSPy**: Advanced customization, fine-grained control, complex pipelines

**Key Features:**
- Automatic requirement-to-reward conversion
- Support for string and callable requirements
- Multiple combination strategies (average, min, product)
- Configurable thresholds and iterations
- Works with any DSPy module

**Requirement Types:**
- Length constraints: "Must be under 50 words"
- Content checks: "Must mention AI"
- Format requirements: "Must be in bullet points"
- Quality criteria: "Must be professional"
- Custom callables: `lambda args, pred: len(pred.answer) < 100`

**See [examples/README.md](examples/README.md) for detailed documentation.**

## Key Features

### Seamless Integration

```python
# Works with all DSPy modules
predictor = dspy.Predict("input -> output")
cot = dspy.ChainOfThought("question -> answer")
```

### Requirements Validation

```python
# Use Mellea's validation (through instruct method)
response = lm(
    prompt="Generate a summary",
    requirements=["Must be under 100 words", "Must mention key points"],
    model_options={}
)
```

### Async Support

```python
# Async generation for better performance
response = await lm.aforward(
    prompt="Generate response",
    model_options={}
)
```

### Multi-Model Support

The backend and model are chosen by the Mellea session. The `model` argument of `MelleaLM` is only metadata on response objects.

```python
# Ollama (default)
lm_ollama = MelleaLM(mellea_session=start_session(backend_name="ollama"))

# OpenAI
lm_openai = MelleaLM(mellea_session=start_session(backend_name="openai", model_id="gpt-4o"))
```

## Documentation

- **[Examples README](examples/README.md)** - Comprehensive examples guide
- **[API Reference](mellea_contribs/dspy/core/lm.py)** - MelleaLM class documentation

## Architecture

```
DSPy Application
       ↓
   MelleaLM (dspy.BaseLM)
       ↓
  Mellea Session
       ↓
  Model Backend (Ollama/OpenAI/WatsonX/LiteLLM/HF)
```

The `MelleaLM` class implements DSPy's `BaseLM` interface, translating between DSPy's expectations and Mellea's capabilities.

## Configuration

### Model Options

```python
lm = MelleaLM(
    mellea_session=m,
    temperature=0.7,
    max_tokens=2000
)
```

### Mellea Session

```python
# Default (Ollama)
m = start_session()

# With specific backend
m = start_session(backend_name="openai", model_id="gpt-4o")

# With a specific model
m = start_session(backend_name="ollama", model_id="granite4:micro")
```

## Learning Path

1. **Start Here:** [Basic Usage](examples/01_basic_usage.py)
2. **Quality Control:** [Requirements Validation](examples/02_requirements_validation.py)
3. **Better Outputs:** [Sampling Strategies](examples/03_sampling_strategies.py)
4. **Performance:** [Async Operations](examples/04_async_operations.py)
5. **Modularity:** [DSPy Modules](examples/05_dspy_modules.py)
6. **Optimization:** [Optimization](examples/06_optimization.py)
7. **Model Configuration:** [Ollama with Granite](examples/07_ollama_granite.py)
8. **Runtime Verification:** [BestOfN](examples/08_bestofn_verification.py), [Refine](examples/09_refine_verification.py)
9. **Choosing an Approach:** [Hybrid](examples/10_hybrid_approach.py)

## Layout convention

The source is laid out flat (`<core mirror>/...`); the wheel exposes the namespaced path:

- On disk:  `mellea_contribs/dspy/core/lm.py`
- Imports:  `from mellea_contribs.dspy.core.lm import MelleaLM`

The doubled segment (`mellea_contribs.dspy.core`) is deliberate — outer = subpackage; inner = mirror of mellea core. Public API is re-exported from `mellea_contribs.dspy`.

## Development

```bash
cd dspy
uv sync --all-extras
uv run pytest -m "not qualitative and not e2e"
```

## Use Cases

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

## Contributing

Contributions welcome! To add examples or improve the integration:

1. Create new examples in `examples/`
2. Follow existing patterns and structure
3. Update documentation
4. Test thoroughly
5. Submit a pull request

## Resources

- [Mellea Documentation](https://mellea.ai/)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Mellea GitHub](https://github.com/generative-computing/mellea)

## Troubleshooting

### Common Issues

**Connection Error:**
```bash
# Ensure Ollama is running
ollama serve
```

**Model Not Found:**
```bash
# Pull the model
ollama pull granite4:micro
```

**Import Error:**
```bash
# Install dependencies
uv sync
```

See [examples/README.md](examples/README.md) for more troubleshooting tips.

## License

Apache-2.0. See [LICENSE](../LICENSE).

---

**Built with Mellea and DSPy**
