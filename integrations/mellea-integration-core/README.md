# Mellea Integration Core

Core abstractions and utilities for building clean, maintainable integrations between Mellea and various AI frameworks (LangChain, CrewAI, DSPy, etc.).

## Overview

This package provides a **hybrid approach** to framework integration:
- **Core abstractions** for common patterns (message conversion, tool handling, session management)
- **Framework-specific extensions** for customization and framework-specific features
- **Reduced code duplication** (~60% reduction across integrations)
- **Consistent behavior** across all framework integrations

## Features

- ✅ **Base Integration Class**: Common patterns for all framework adapters
- ✅ **Message Conversion**: Utilities for converting between framework and Mellea formats
- ✅ **Tool Conversion**: Utilities for handling tool calling across frameworks
- ✅ **Requirements Support**: Built-in support for Mellea's validation features
- ✅ **Async Support**: Both sync and async generation patterns
- ✅ **Type Safety**: Protocol-based design with type hints

## Installation

```bash
pip install mellea-integration-core
```

## Quick Start

### Creating a Framework Integration

```python
from mellea_integration import MelleaIntegrationBase, BaseMessageConverter
from typing import Any

class MyFrameworkMessageConverter(BaseMessageConverter):
    """Convert MyFramework messages to/from Mellea format."""
    
    def to_mellea(self, messages: Any) -> list[Any]:
        # Convert framework messages to Mellea Message objects
        mellea_messages = []
        for msg in messages:
            mellea_messages.append(
                self.create_mellea_message(
                    role=msg.role,
                    content=msg.content
                )
            )
        return mellea_messages
    
    def from_mellea(self, response: Any) -> Any:
        # Convert Mellea response to framework format
        content = self.extract_content_from_response(response)
        return MyFrameworkResponse(content=content)

class MyFrameworkAdapter(MelleaIntegrationBase):
    """Adapter for MyFramework using Mellea."""
    
    def __init__(self, mellea_session: Any, **kwargs: Any):
        super().__init__(
            mellea_session=mellea_session,
            message_converter=MyFrameworkMessageConverter(),
            **kwargs
        )
    
    def generate(self, messages: Any, **kwargs: Any) -> Any:
        # Prepare inputs
        prompt, model_options, tool_calls_enabled = self._prepare_generation(
            messages, kwargs.get("tools"), **kwargs
        )
        
        # Generate with Mellea
        response = self._generate_with_mellea(
            prompt, model_options, tool_calls_enabled,
            kwargs.get("requirements"), kwargs.get("strategy")
        )
        
        # Convert response
        return self.message_converter.from_mellea(response)
    
    async def agenerate(self, messages: Any, **kwargs: Any) -> Any:
        # Async version
        prompt, model_options, tool_calls_enabled = self._prepare_generation(
            messages, kwargs.get("tools"), **kwargs
        )
        
        response = await self._agenerate_with_mellea(
            prompt, model_options, tool_calls_enabled,
            kwargs.get("requirements"), kwargs.get("strategy")
        )
        
        return self.message_converter.from_mellea(response)
```

## Architecture

```
┌─────────────────────────────────────────┐
│     Framework Application               │
│  (LangChain, CrewAI, DSPy, etc.)       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Framework-Specific Adapter            │
│   (Inherits MelleaIntegrationBase)      │
│   • Custom message conversion           │
│   • Framework-specific features         │
│   • Event/callback handling             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   MelleaIntegrationBase                 │
│   • Message preparation                 │
│   • Tool handling                       │
│   • Session management                  │
│   • Requirements/strategy support       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│        Mellea Session                   │
│   (chat, instruct, achat, ainstruct)    │
└─────────────────────────────────────────┘
```

## Core Components

### MelleaIntegrationBase

Base class providing common functionality:

- **`_prepare_generation()`**: Convert messages and tools, prepare model options
- **`_generate_with_mellea()`**: Choose between chat() and instruct() methods
- **`_agenerate_with_mellea()`**: Async version of generation
- **`_handle_sampling_results()`**: Extract results from validation

### BaseMessageConverter

Utilities for message conversion:

- **`extract_last_user_message()`**: Get the last user message content
- **`create_mellea_message()`**: Create Mellea Message with validation
- **`normalize_content()`**: Normalize various content formats to string
- **`extract_content_from_response()`**: Extract text from Mellea response

### BaseToolConverter

Utilities for tool conversion:

- **`extract_tool_schema()`**: Get JSON schema from tool object
- **`get_tool_callable()`**: Extract callable function from tool
- **`parse_tool_calls_from_string()`**: Parse tool calls from string representation
- **`extract_tool_calls_from_response()`**: Extract tool calls from Mellea response

## Usage Examples

### Basic Integration

```python
from mellea import start_session
from mellea_integration import MelleaIntegrationBase

# Create Mellea session
m = start_session()

# Create adapter (framework-specific)
adapter = MyFrameworkAdapter(mellea_session=m)

# Generate
response = adapter.generate(messages=[...])
```

### With Requirements

```python
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import RejectionSamplingStrategy

adapter = MyFrameworkAdapter(
    mellea_session=m,
    requirements=[
        req("Response must be professional"),
        req("Include specific examples")
    ],
    strategy=RejectionSamplingStrategy(loop_budget=5)
)

response = adapter.generate(messages=[...])
```

### With Tools

```python
# Tools are automatically converted
response = adapter.generate(
    messages=[...],
    tools=[tool1, tool2]
)
```

## Benefits

### Code Reduction
- **~60% reduction** in duplicated code across integrations
- Shared message/tool conversion logic
- Common Mellea session management

### Maintainability
- Single source of truth for core patterns
- Bug fixes propagate to all integrations
- Easier to add new framework integrations

### Consistency
- Uniform behavior across frameworks
- Consistent requirements/strategy handling
- Standardized error handling

### Extensibility
- Easy to add new frameworks
- Framework-specific customization preserved
- Protocol-based design allows flexibility

## Existing Integrations

This core package is used by:

- **[mellea-langchain](../langchain/)**: LangChain integration
- **[mellea-crewai](../crewai/)**: CrewAI integration
- **[mellea-dspy](../dspy/)**: DSPy integration

## Development

### Setup

```bash
# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mellea_integration --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/mellea_integration
```

## Contributing

Contributions welcome! To add features or improve the core:

1. Create a feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit a pull request

## License

Apache License 2.0 - see LICENSE file for details.

## Support

- **Issues**: Report bugs on GitHub Issues
- **Discord**: Join the Mellea Discord
- **Documentation**: Visit mellea.ai

---

**Version**: 0.1.0  
**Status**: Alpha  
**Built with ❤️ for the Mellea community**