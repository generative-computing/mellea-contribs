<img src="https://github.com/generative-computing/mellea-contribs/raw/main/mellea-contribs.jpg" height=100>


# Mellea Contribs

The `mellea-contribs` repository is an incubation point for contributions to
the Mellea ecosystem.

# Mellea Contributions Directory Structure

This document explains the organization of the `mellea_contribs` directory and the requirements for running CI/CD.

## Directory Structure Overview

```
mellea_contribs/
├── mellea-integration-core/      # Core abstractions for framework integrations
├── crewai_backend/               # CrewAI integration with Mellea
├── dspy_backend/                 # DSPy integration with Mellea
├── langchain_backend/            # LangChain integration with Mellea
├── tools_package/                # Generative tools and utilities library
├── reqlib_package/               # Requirements library for validation
├── mcp_tools/                    # Expose MCP server tools as Mellea tools
└── __init__.py                   # Package initialization
```

## Subpackage Descriptions

### 1. **mellea-integration-core**
- **Purpose**: Core abstractions and utilities for building clean, maintainable integrations between Mellea and various AI frameworks
- **Key Features**:
  - Base integration class with common patterns
  - Message conversion utilities (convert between framework and Mellea formats)
  - Tool conversion and handling
  - Requirements and strategy support
  - Async/sync generation patterns
- **Dependencies**: `mellea>=0.3.2`
- **Python Version**: ≥3.11
- **CI Requirements**: No Ollama needed (skip_ollama=true)
- **Timeout**: 30 minutes

### 2. **crewai_backend**
- **Purpose**: Enables CrewAI agents to use Mellea's generative programming capabilities
- **Key Features**:
  - CrewAI agent integration with Mellea
  - Message and tool conversion for CrewAI format
  - Inherits from `MelleaIntegrationBase`
- **Dependencies**: `mellea>=0.3.0`, `crewai>=0.1.0`, `mellea-integration-core`
- **Python Version**: ≥3.11
- **CI Requirements**: Ollama support enabled
- **Timeout**: 90 minutes (extended due to complex integration tests)

### 3. **dspy_backend**
- **Purpose**: DSPy integration enabling structured prompting with generative programming
- **Key Features**:
  - DSPy module integration with Mellea
  - Structured prompting capabilities
  - Inherits from `MelleaIntegrationBase`
- **Dependencies**: `dspy>=3.1.3`, `mellea>=0.3.2`, `mellea-integration-core`
- **Python Version**: ≥3.11
- **CI Requirements**: Ollama support enabled
- **Timeout**: 30 minutes

### 4. **langchain_backend**
- **Purpose**: LangChain integration for using Mellea within LangChain applications
- **Key Features**:
  - LangChain language model integration with Mellea
  - Tool calling support
  - Inherits from `MelleaIntegrationBase`
- **Dependencies**: `langchain`, `mellea>=0.3.x`, `mellea-integration-core`
- **Python Version**: ≥3.11
- **CI Requirements**: Ollama support enabled
- **Timeout**: 30 minutes

### 5. **tools_package**
- **Purpose**: Incubating generative programming tools and utilities
- **Key Features**:
  - Various tools for generative programming
  - Robustness testing capabilities
  - Requirements validation and sampling strategies
- **Dependencies**: Multiple (see pyproject.toml for details)
- **Python Version**: ≥3.11
- **CI Requirements**: Ollama support enabled
- **Timeout**: 30 minutes

### 6. **reqlib_package**
- **Purpose**: Requirements library for validation and constraints in generative systems
- **Key Features**:
  - Requirement specification and validation
  - Integration with Mellea's validation framework
- **Dependencies**: `mellea>=0.3.x`
- **Python Version**: ≥3.11
- **CI Requirements**: No Ollama needed (skip_ollama=true)
- **Timeout**: 30 minutes

### 7. **mcp_tools**
- **Purpose**: Bridges MCP (Model Context Protocol) server tools into Mellea's native tool-calling system
- **Key Features**:
  - Discover tools from any MCP server via HTTP, SSE, or stdio
  - Wrap MCP tools as `MelleaTool` instances for use in agents and `react()` loops
  - Short-lived per-call sessions (no session lifetime management required)
- **Dependencies**: `mellea>=0.4.2`, `mcp>=1.27.0`, `httpx>=0.27`
- **Python Version**: ≥3.11
- **CI Requirements**: No Ollama needed (skip_ollama=true)
- **Timeout**: 30 minutes

## CI/CD Requirements

### Architecture

The CI/CD pipeline uses a two-stage workflow:

1. **discover-subpackages** (ci.yml): Discovers which subpackages have changed
2. **test** (quality-generic.yml): Runs tests for each changed subpackage

### Key Requirements

#### 1. **Environment Setup**
- **Python Versions**: Tests run on Python 3.11, 3.12, and 3.13
- **Package Manager**: Uses `uv` for fast dependency management
- **Virtual Environment**: `uv sync --all-extras` creates and manages the venv

#### 2. **Test Directory Structure**
Each subpackage must have a test directory with the following structure:
```
<subpackage>/
├── tests/          # OR
├── test/           # OR
└── test/integration/  # (optional, ignored in CI)
```

The CI pipeline looks for either `tests/` or `test/` directory. Integration tests in `test/integration/` are skipped.

#### 3. **Dependency Management**
- **Package Format**: Each subpackage is a Python package with `pyproject.toml`
- **Build System**: 
  - `crewai_backend`, `dspy_backend`, `langchain_backend`, `mellea-integration-core`: Use `hatchling`
  - `tools_package`: Uses `pdm-backend`
  - `reqlib_package`: Uses appropriate build backend
- **Local Path Dependencies**: Subpackages can reference each other via path-based dependencies in `pyproject.toml`:
  ```toml
  [tool.uv.sources]
  mellea-integration-core = { path = "../mellea-integration-core" }
  ```

#### 4. **Ollama Setup (for backends)**
Some subpackages require Ollama for testing:
- **Skip Ollama**: `mellea-integration-core`, `reqlib_package`, `mcp_tools`
- **Include Ollama**: `crewai_backend`, `dspy_backend`, `langchain_backend`, `tools_package`

When Ollama is enabled:
1. Ollama service is installed and started
2. Model `granite4:micro` is pulled for testing

#### 5. **Test Execution**
```bash
cd <subpackage>
uv sync --all-extras          # Install all dependencies including dev extras
uv run -- pytest -v tests/    # Run tests (or test/ if tests/ doesn't exist)
```

#### 6. **Pytest Configuration**
Each subpackage's `pyproject.toml` should include pytest markers for test organization:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: marks tests as requiring integration setup",
    "llm: marks tests making LLM calls",
    "slow: marks slow tests",
]
asyncio_mode = "auto"
```

#### 7. **Code Quality Tools**
Standard tools configured in each subpackage:
- **Ruff**: Linting and formatting (configured in `pyproject.toml`)
- **MyPy**: Type checking (configured in `pyproject.toml`)
- **Pytest**: Testing framework with coverage support

### CI Trigger Conditions

The CI pipeline triggers when:
- **Push to main**: Changes detected in `mellea_contribs/` directory
- **Pull Request**: Changes detected in `mellea_contribs/` directory
- **Workflow Changes**: Any change to `.github/workflows/ci.yml` or `.github/workflows/quality-generic.yml`

### CI Discovery Logic

The discovery step:
1. Gets changed files compared to base ref (PR base or HEAD~1 for push)
2. Extracts unique subpackage directories that changed
3. If workflow files changed, tests ALL subpackages
4. Creates a test matrix with custom settings per subpackage:
   - `skip_ollama`: Whether to skip Ollama installation
   - `timeout_minutes`: Custom timeout (default 30, extended to 90 for crewai_backend)

### CI Parallelization

- **Max Parallel Jobs**: 6 concurrent subpackages
- **Python Versions**: Tests run sequentially per Python version (max-parallel: 1)
- **Concurrency Groups**: Uses PR number or branch name for cancellation of in-progress jobs

## Setting Up a New Subpackage for CI

To add a new subpackage to the CI pipeline:

### 1. Create Package Structure
```
new_backend/
├── pyproject.toml
├── README.md
├── src/
│   └── mellea_<framework>/
│       ├── __init__.py
│       └── integration.py
└── tests/  # or test/
    ├── __init__.py
    ├── test_integration.py
    └── integration/
        └── test_advanced.py
```

### 2. Configure pyproject.toml
```toml
[project]
name = "mellea-<framework>"
version = "0.1.0"
description = "<Framework> integration for Mellea"
requires-python = ">=3.11"

dependencies = [
    "mellea>=0.3.0",
    "mellea-integration-core",
    "<framework>",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mellea_<framework>"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.uv.sources]
mellea-integration-core = { path = "../mellea-integration-core" }
```

### 3. Configure CI Settings
Update `.github/workflows/ci.yml` discover-subpackages step to add custom settings if needed:
```bash
skip_ollama: (if test("<new_backend>") then true else false end),
timeout_minutes: (if test("<new_backend>") then 60 else 30 end)
```

### 4. Verify Locally
```bash
cd new_backend
uv sync --all-extras
uv run -- pytest -v tests/
```

## Troubleshooting

### Common CI Failures

1. **"No tests/ or test/ directory found"**
   - Solution: Create `tests/` or `test/` directory with test files

2. **Import errors for mellea-integration-core**
   - Solution: Ensure `pyproject.toml` has correct path reference:
     ```toml
     [tool.uv.sources]
     mellea-integration-core = { path = "../mellea-integration-core" }
     ```

3. **Ollama timeout issues**
   - Solution: CI has 30-90 minute timeout, may need to skip heavy ollama tests locally

4. **Python version incompatibility**
   - Solution: Verify `requires-python` in `pyproject.toml` matches supported versions

5. **Missing dev dependencies**
   - Solution: Add pytest and other test dependencies to `[project.optional-dependencies] dev`

## Local Development

### Install All Subpackages Locally
```bash
cd mellea_contribs/mellea-integration-core
pip install -e ".[dev]"

cd ../crewai_backend
pip install -e ".[dev]"

# ... repeat for other subpackages
```

### Run Tests Locally
```bash
cd <subpackage>
uv sync --all-extras
uv run -- pytest -v tests/
```

### Code Quality Checks
```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/
```

## Summary

The `mellea_contribs` directory contains multiple framework integrations and supporting libraries for the Mellea generative programming framework. Each subpackage is independently testable but shares common dependencies through `mellea-integration-core`. The CI pipeline automatically discovers changes and runs appropriate tests with Python 3.11-3.13, Ollama support for backends, and configurable timeouts per package.


## Tools

- **[Robustness Testing](docs/ROBUSTNESS_TESTING.md)** — Test m-program consistency against semantic variations using BenchDrift
