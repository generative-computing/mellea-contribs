# Mellea + DSPy Integration Examples

This directory contains comprehensive examples demonstrating the Mellea + DSPy integration. Each example focuses on specific features and use cases.

## 📚 Examples Overview

### 01. Basic Usage (`01_basic_usage.py`)
**Fundamental usage patterns for Mellea + DSPy**

Learn the basics of using Mellea as a DSPy backend:
- Simple predictions with `dspy.Predict`
- Chain of thought reasoning with `dspy.ChainOfThought`
- Custom signatures for structured I/O
- Multiple output fields
- Basic configuration and setup

**Run it:**
```bash
uv run examples/01_basic_usage.py
```

**Key Concepts:**
- MelleaLM initialization
- DSPy configuration
- Signature definition
- Predictor creation

---

### 02. Requirements Validation (`02_requirements_validation.py`)
**Using Mellea's requirements validation with DSPy**

Demonstrates how to ensure generated outputs meet specific criteria:
- Basic requirements validation
- Length constraints
- Format requirements (bullet points, numbered lists, etc.)
- Content requirements (must include specific terms)
- Combined requirements

**Run it:**
```bash
uv run examples/02_requirements_validation.py
```

**Key Concepts:**
- Requirements specification
- Output validation
- Quality assurance
- Constraint enforcement

---

### 03. Sampling Strategies (`03_sampling_strategies.py`)
**Inference-time scaling with sampling strategies**

Explore different sampling strategies for improved output quality:
- Rejection sampling (retry until valid)
- Best-of-N sampling (generate multiple, select best)
- Temperature sampling (control creativity)
- Ensemble sampling (combine multiple outputs)
- Adaptive sampling (adjust based on confidence)

**Run it:**
```bash
uv run examples/03_sampling_strategies.py
```

**Key Concepts:**
- Sampling strategies
- Inference-time scaling
- Quality vs. compute trade-offs
- Confidence-based adaptation

---

### 04. Async Operations (`04_async_operations.py`)
**Asynchronous generation for better performance**

Learn how to use async operations for concurrent processing:
- Basic async generation
- Concurrent generation (multiple requests)
- Batch processing
- Async chain of thought
- Error handling in async context
- Streaming simulation

**Run it:**
```bash
uv run examples/04_async_operations.py
```

**Key Concepts:**
- Async/await patterns
- Concurrent execution with `asyncio.gather()`
- Performance optimization
- Error handling

---

### 05. DSPy Modules (`05_dspy_modules.py`)
**Building reusable DSPy modules and programs**

Create modular, composable AI programs:
- Simple modules
- Multi-step reasoning modules
- Composable modules (pipelines)
- Conditional modules (adaptive logic)
- Stateful modules (conversation history)
- Reusable components

**Run it:**
```bash
uv run examples/05_dspy_modules.py
```

**Key Concepts:**
- Module composition
- Program structure
- Reusability
- State management

---

### 06. Optimization (`06_optimization.py`)
**Optimizing DSPy programs with Mellea**

Advanced optimization techniques for better performance:
- Few-shot learning
- Bootstrap few-shot optimization
- Signature optimization
- Metric-based optimization
- Prompt optimization strategies
- Program compilation
- Hyperparameter tuning

**Run it:**
```bash
uv run examples/06_optimization.py
```

**Key Concepts:**
- Training data creation
- Evaluation metrics
- Optimization strategies
- Hyperparameter tuning

---

### 07. Ollama with Granite (`07_ollama_granite.py`)
**Using Mellea DSPy with Ollama backend and specific models**

Demonstrates how to configure Mellea with Ollama backend using a specific model:
- Proper OllamaBackend creation
- MelleaSession configuration with custom backend
- Understanding model parameter vs backend configuration
- Basic DSPy operations with custom model

**Run it:**
```bash
uv run examples/07_ollama_granite.py
```

**Key Concepts:**
- OllamaModelBackend initialization
- Backend configuration
- Model selection (backend vs constructor parameter)
- Custom model usage

**Important Note:**
The `model` parameter in `MelleaLM` constructor is only used for metadata in response objects. The actual model used for generation is determined by the backend configuration in the MelleaSession.

---

## 🚀 Quick Start

### Prerequisites

1. Install dependencies:
```bash
uv sync
```

2. Ensure Ollama is running with a model:
```bash
# Install Ollama from https://ollama.com/
ollama pull granite4:micro
```

### Running Examples

Run all examples:
```bash
# Run individually
uv run examples/01_basic_usage.py
uv run examples/02_requirements_validation.py
uv run examples/03_sampling_strategies.py
uv run examples/04_async_operations.py
uv run examples/05_dspy_modules.py
uv run examples/06_optimization.py
uv run examples/07_ollama_granite.py
```

Or run the original comprehensive example:
```bash
uv run example_mellea_dspy.py
```

---

## 📖 Learning Path

**Beginner:**
1. Start with `01_basic_usage.py` to understand fundamentals
2. Explore `02_requirements_validation.py` for quality control
3. Try `03_sampling_strategies.py` for better outputs

**Intermediate:**
4. Learn async patterns in `04_async_operations.py`
5. Build modular programs with `05_dspy_modules.py`

**Advanced:**
6. Master optimization in `06_optimization.py`
7. Combine techniques for production systems

---

## 🎯 Use Cases by Example

### Text Generation
- **Basic Usage**: Simple text generation
- **Requirements**: Controlled generation with constraints
- **Sampling**: High-quality creative text

### Question Answering
- **Basic Usage**: Simple Q&A
- **Modules**: Multi-step reasoning
- **Optimization**: Improved accuracy with few-shot

### Classification
- **Basic Usage**: Text classification
- **Optimization**: Few-shot classification
- **Sampling**: Confidence-based classification

### Summarization
- **Requirements**: Length-constrained summaries
- **Modules**: Multi-stage summarization
- **Async**: Batch summarization

### Code Generation
- **Requirements**: Valid code with constraints
- **Sampling**: Multiple candidates, select best
- **Modules**: Multi-step code generation

---

## 🔧 Configuration

### Model Selection

Change the model in any example:
```python
lm = MelleaLM(
    mellea_session=m,
    model="mellea-ollama",  # or "mellea-openai", "mellea-anthropic"
    temperature=0.7,
    max_tokens=1000
)
```

### Mellea Session Options

Configure Mellea session:
```python
from mellea import start_session

# Default (Ollama)
m = start_session()

# With specific backend
m = start_session(backend="openai")

# With custom options
m = start_session(
    backend="ollama",
    model="granite4:micro"
)
```

---

## 💡 Tips and Best Practices

### Performance
- Use async operations for concurrent requests
- Batch similar requests together
- Cache results when appropriate
- Monitor token usage

### Quality
- Start with clear signatures
- Use requirements for constraints
- Implement validation logic
- Test with diverse inputs

### Development
- Start simple, add complexity gradually
- Test modules independently
- Use meaningful variable names
- Document signatures and modules

### Production
- Implement error handling
- Add logging and monitoring
- Use appropriate sampling strategies
- Optimize with real data

---

## 🐛 Troubleshooting

### Common Issues

**"Connection refused" error:**
- Ensure Ollama is running: `ollama serve`
- Check if model is pulled: `ollama list`

**Slow generation:**
- Use async operations for concurrency
- Consider smaller models for faster inference
- Adjust max_tokens parameter

**Poor output quality:**
- Add more descriptive signatures
- Use requirements validation
- Try different sampling strategies
- Optimize with few-shot examples

**Import errors:**
- Run `uv sync` to install dependencies
- Ensure you're in the correct directory
- Check Python version (3.10+)

---

## 📚 Additional Resources

### Documentation
- [Mellea Documentation](https://mellea.ai/)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Integration README](../README_MELLEA_DSPY.md)

### Related Examples
- [Test Suite](../test_mellea_dspy.py) - Integration tests
- [Original Example](../example_mellea_dspy.py) - Comprehensive demo

### Community
- [Mellea GitHub](https://github.com/generative-computing/mellea)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)

---

## 🤝 Contributing

Have an example idea? Contributions welcome!

1. Create a new example file: `examples/0X_your_example.py`
2. Follow the existing format and structure
3. Add documentation to this README
4. Test thoroughly
5. Submit a pull request

---

## 📝 Example Template

Use this template for new examples:

```python
"""Example description.

This example demonstrates...
"""

import dspy
from mellea import start_session
from mellea_dspy import MelleaLM


def example_function():
    """Demonstrate specific feature."""
    print("=" * 70)
    print("Example: Feature Name")
    print("=" * 70)
    
    # Setup
    print("\n1. Setting up...")
    m = start_session()
    lm = MelleaLM(mellea_session=m, model="mellea-ollama")
    dspy.configure(lm=lm)
    print("   ✓ Setup complete")
    
    # Your example code here
    
    print("\n" + "=" * 70)


def main():
    """Run all examples."""
    try:
        example_function()
        print("\n✅ Example completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

---

## 📄 License

This integration follows the same license as the parent project.

---

**Happy coding with Mellea + DSPy! 🚀**