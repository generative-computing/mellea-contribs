# Mellea Framework Integrations

This repository contains integrations for the Mellea generative programming framework with popular agentic frameworks.

## Available Integrations

### 🔷 DSPy Integration (`dspy/`)
Integration with DSPy for structured prompting and optimization.
- **Package:** `mellea-dspy`
- **Documentation:** [dspy/README.md](dspy/README.md)
- **Getting Started:** [dspy/GETTING_STARTED.md](dspy/GETTING_STARTED.md)

### 🤖 CrewAI Integration (`crewai/`)
Integration with CrewAI for multi-agent orchestration.
- **Package:** `mellea-crewai`
- **Documentation:** [crewai/README.md](crewai/README.md)
- **Getting Started:** [crewai/GETTING_STARTED.md](crewai/GETTING_STARTED.md)

### 🦜 LangChain Integration (`langchain/`)
Integration with LangChain for building LLM applications.
- **Package:** `mellea-langchain`
- **Documentation:** [langchain/README.md](langchain/README.md)
- **Getting Started:** [langchain/GETTING_STARTED.md](langchain/GETTING_STARTED.md)

## Quick Start

### Installation

Install a specific integration:

```bash
# DSPy
pip install -e dspy[dev]

# CrewAI
pip install -e crewai[dev]

# LangChain
pip install -e langchain[dev]
```

### Running Integration Tests

We provide a unified test runner for all integrations:

```bash
# Run all integration tests
python3 run_integration_tests.py

# Run tests for specific framework(s)
python3 run_integration_tests.py --frameworks dspy crewai

# Install dependencies and run tests
python3 run_integration_tests.py --install-deps

# Using Make (if available)
make test-all          # Run all tests
make test-dspy         # Run DSPy tests
make test-crewai       # Run CrewAI tests
make test-langchain    # Run LangChain tests
```

For detailed testing documentation, see [INTEGRATION_TESTS.md](INTEGRATION_TESTS.md).

## Repository Structure

```
.
├── dspy/                          # DSPy integration
│   ├── src/mellea_dspy/          # Source code
│   ├── tests/                     # Tests
│   │   ├── integration/          # Integration tests
│   │   └── unit/                 # Unit tests
│   ├── examples/                  # Example scripts
│   └── pyproject.toml            # Package configuration
│
├── crewai/                        # CrewAI integration
│   ├── src/mellea_crewai/        # Source code
│   ├── tests/                     # Tests
│   │   └── integration/          # Integration tests
│   ├── examples/                  # Example scripts
│   └── pyproject.toml            # Package configuration
│
├── langchain/                     # LangChain integration
│   ├── src/mellea_langchain/     # Source code
│   ├── tests/                     # Tests
│   │   └── integration/          # Integration tests
│   ├── examples/                  # Example scripts
│   └── pyproject.toml            # Package configuration
│
├── run_integration_tests.py      # Unified test runner
├── pytest.ini                     # Common pytest configuration
├── requirements-integration-tests.txt  # Test dependencies
├── INTEGRATION_TESTS.md          # Testing documentation
├── Makefile                       # Convenience commands
└── README.md                      # This file
```

## Development

### Setting Up Development Environment

1. Clone the repository
2. Install dependencies for the integration(s) you want to work on:
   ```bash
   pip install -e dspy[dev]
   pip install -e crewai[dev]
   pip install -e langchain[dev]
   ```

3. Run tests to verify setup:
   ```bash
   python3 run_integration_tests.py
   ```

### Running Tests

#### Using the Test Runner
```bash
# All tests
python3 run_integration_tests.py

# Specific framework
python3 run_integration_tests.py --frameworks dspy

# With verbose output
python3 run_integration_tests.py -v

# Pass custom pytest arguments
python3 run_integration_tests.py --pytest-args "-k test_basic -x"
```

#### Using pytest Directly
```bash
# All integration tests
pytest -m integration

# Specific framework
pytest dspy/tests/integration -m integration
pytest crewai/tests/integration -m integration
pytest langchain/tests/integration -m integration

# With markers
pytest -m "integration and not slow"
pytest -m "integration and ollama"
```

#### Using Make
```bash
make test-all          # Run all integration tests
make test-dspy         # Run DSPy tests only
make test-crewai       # Run CrewAI tests only
make test-langchain    # Run LangChain tests only
make list              # List available frameworks
make clean             # Clean test artifacts
```

### Test Markers

Common markers across all integrations:
- `integration` - Integration tests requiring live Mellea session
- `unit` - Unit tests (fast, no external dependencies)
- `slow` - Tests taking >5 minutes
- `ollama` - Tests requiring Ollama backend
- `openai` - Tests requiring OpenAI backend

See [INTEGRATION_TESTS.md](INTEGRATION_TESTS.md) for complete marker documentation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add/update tests
5. Run tests: `python3 run_integration_tests.py`
6. Submit a pull request

## Documentation

- **Integration Tests Guide:** [INTEGRATION_TESTS.md](INTEGRATION_TESTS.md)
- **DSPy Integration:** [dspy/README.md](dspy/README.md)
- **CrewAI Integration:** [crewai/README.md](crewai/README.md)
- **LangChain Integration:** [langchain/README.md](langchain/README.md)

## License

Each integration may have its own license. See individual integration directories for details.

## Support

For issues or questions:
- Check the individual integration's documentation
- Review [INTEGRATION_TESTS.md](INTEGRATION_TESTS.md) for testing help
- Open an issue in the repository

## Related Projects

- [Mellea](https://github.com/generative-computing/mellea) - Core Mellea framework
- [DSPy](https://github.com/stanfordnlp/dspy) - Structured prompting framework
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework