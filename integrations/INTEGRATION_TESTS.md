# Integration Tests Guide

This guide explains how to run integration tests for the Mellea framework integrations (DSPy, CrewAI, and LangChain).

## Overview

The integration tests verify that each framework integration works correctly with Mellea's generative programming capabilities. Each integration has its own test suite in its respective `tests/integration` directory.

## Quick Start

### 1. Install Dependencies

You have several options for installing dependencies:

#### Option A: Install all integration test dependencies at once
```bash
pip install -r requirements-integration-tests.txt
```

#### Option B: Install each integration individually
```bash
# DSPy
pip install -e dspy[dev]

# CrewAI
pip install -e crewai[dev]

# LangChain
pip install -e langchain[dev]
```

#### Option C: Use the test runner to install dependencies automatically
```bash
python run_integration_tests.py --install-deps
```

### 2. Run Integration Tests

#### Run all integration tests
```bash
python run_integration_tests.py
```

#### Run tests for specific framework(s)
```bash
python run_integration_tests.py --frameworks dspy
python run_integration_tests.py --frameworks dspy crewai
python run_integration_tests.py --frameworks langchain
```

#### Run with verbose output
```bash
python run_integration_tests.py -v
```

## Using pytest Directly

You can also run tests directly with pytest:

### Run all integration tests
```bash
pytest -m integration
```

### Run tests for a specific framework
```bash
# DSPy
pytest dspy/tests/integration -m integration

# CrewAI
pytest crewai/tests/integration -m integration

# LangChain
pytest langchain/tests/integration -m integration
```

### Run specific test files
```bash
pytest dspy/tests/integration/test_basic.py
pytest crewai/tests/integration/test_integration_crew.py
pytest langchain/tests/integration/test_langchain_integration.py
```

### Run tests with specific markers
```bash
# Run only tests that require Ollama
pytest -m "integration and ollama"

# Run integration tests but skip slow ones
pytest -m "integration and not slow"

# Run only LangChain streaming tests
pytest -m "integration and streaming"
```

## Test Markers

The following pytest markers are available across all integrations:

### General Markers
- `integration` - Integration tests requiring live Mellea session
- `unit` - Unit tests (fast, no external dependencies)
- `llm` - Tests that make LLM calls
- `slow` - Tests taking >5 minutes

### Backend-Specific Markers
- `ollama` - Tests requiring Ollama backend
- `openai` - Tests requiring OpenAI backend
- `watsonx` - Tests requiring Watsonx backend
- `huggingface` - Tests requiring HuggingFace backend
- `vllm` - Tests requiring vLLM backend
- `litellm` - Tests requiring LiteLLM backend

### Resource Requirements
- `requires_gpu` - Tests requiring GPU
- `requires_heavy_ram` - Tests requiring 48GB+ RAM
- `requires_api_key` - Tests requiring external API keys

### Framework-Specific Markers
- `dspy` - DSPy integration tests
- `crewai` - CrewAI integration tests
- `langchain` - LangChain integration tests
- `streaming` - Streaming functionality tests
- `tool_calling` - Tool calling tests
- `qualitative` - Tests checking LLM output quality

## Advanced Usage

### Pass custom pytest arguments
```bash
python run_integration_tests.py --pytest-args "-k test_basic -x"
python run_integration_tests.py --pytest-args "--maxfail=1 -v"
```

### Run with coverage reporting
```bash
pytest -m integration --cov=mellea_dspy --cov=mellea_crewai --cov=mellea_langchain --cov-report=html
```

### Run tests in parallel (requires pytest-xdist)
```bash
pip install pytest-xdist
pytest -m integration -n auto
```

### List available frameworks
```bash
python run_integration_tests.py --list
```

## Framework-Specific Details

### DSPy Integration Tests

**Location:** `dspy/tests/integration/`

**Key Test Files:**
- `test_basic.py` - Basic forward/aforward functionality
- `test_dspy_modules.py` - DSPy module integration
- `test_requirements_live.py` - Requirements validation
- `test_strategy_live.py` - Sampling strategies

**Example:**
```bash
# Run all DSPy integration tests
pytest dspy/tests/integration -m integration

# Run only basic tests
pytest dspy/tests/integration/test_basic.py -v
```

### CrewAI Integration Tests

**Location:** `crewai/tests/integration/`

**Key Test Files:**
- `test_integration_crew.py` - Full crew pipeline tests

**Markers:**
- Tests are marked with both `integration` and `llm`
- Real backend tests use `ollama` marker

**Example:**
```bash
# Run all CrewAI integration tests
pytest crewai/tests/integration -m integration

# Run only mock-based tests (skip real Ollama tests)
pytest crewai/tests/integration -m "integration and not ollama"
```

### LangChain Integration Tests

**Location:** `langchain/tests/integration/`

**Key Test Files:**
- `test_langchain_integration.py` - Comprehensive integration tests

**Test Groups:**
- Basic chat integration
- Streaming integration
- Message conversion
- Tool calling
- Requirements and strategy
- LangChain chain integration
- Error handling

**Example:**
```bash
# Run all LangChain integration tests
pytest langchain/tests/integration -m integration

# Run only streaming tests
pytest langchain/tests/integration -m "integration and streaming"
```

## Troubleshooting

### Tests fail with "No module named 'mellea'"
Install the core Mellea package:
```bash
pip install mellea>=0.3.0
```

### Tests fail with missing framework dependencies
Install the specific framework's dev dependencies:
```bash
pip install -e <framework>[dev]
```

### Tests hang or timeout
Some integration tests make real LLM calls which can be slow. Use the `--timeout` option:
```bash
pytest -m integration --timeout=300
```

### Skip slow tests
```bash
pytest -m "integration and not slow"
```

### Backend not available (e.g., Ollama)
Skip backend-specific tests:
```bash
pytest -m "integration and not ollama"
```

## Configuration

### pytest.ini
The root-level `pytest.ini` file contains common configuration for all integrations:
- Test discovery patterns
- Async support
- Markers
- Output formatting
- Logging configuration

### Individual pyproject.toml
Each integration has its own `pyproject.toml` with framework-specific settings.

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        framework: [dspy, crewai, langchain]
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e ${{ matrix.framework }}[dev]
      
      - name: Run integration tests
        run: |
          python run_integration_tests.py --frameworks ${{ matrix.framework }}
```

## Best Practices

1. **Run tests locally before committing** - Ensure your changes don't break existing functionality
2. **Use markers to filter tests** - Run only relevant tests during development
3. **Keep tests isolated** - Each test should be independent and not rely on others
4. **Mock external dependencies when possible** - Use mocks for faster unit tests, real backends for integration tests
5. **Document test requirements** - Clearly mark tests that need specific backends or resources

## Contributing

When adding new integration tests:

1. Place tests in the appropriate `tests/integration/` directory
2. Use appropriate markers (especially `integration`)
3. Follow existing test patterns and naming conventions
4. Add docstrings explaining what the test verifies
5. Update this documentation if adding new test categories or requirements

## Support

For issues or questions:
- Check the individual framework's README and GETTING_STARTED guides
- Review existing test files for examples
- Open an issue in the repository