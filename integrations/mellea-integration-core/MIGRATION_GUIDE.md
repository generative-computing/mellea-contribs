# Migration Guide: Using mellea-integration-core

This guide explains how to migrate existing Mellea framework integrations to use the new `mellea-integration-core` package.

## Overview

The `mellea-integration-core` package provides:
- **Base classes** for common integration patterns
- **Utility functions** for message and tool conversion
- **Reduced code duplication** (~60% reduction)
- **Consistent behavior** across all integrations

## Benefits of Migration

### Before (Without Core Package)
```python
# Each integration duplicates:
# - Message conversion logic
# - Tool conversion logic
# - Mellea session management
# - Requirements/strategy handling
# Total: ~500-600 lines per integration
```

### After (With Core Package)
```python
# Each integration only needs:
# - Framework-specific message converter (~50 lines)
# - Framework-specific adapter (~100 lines)
# - Framework-specific features (events, callbacks, etc.)
# Total: ~150-200 lines per integration
```

## Migration Steps

### Step 1: Install Core Package

```bash
pip install mellea-integration-core
```

### Step 2: Create Message Converter

Replace your existing message conversion code with a class that extends `BaseMessageConverter`:

#### Before (LangChain Example)
```python
# langchain/src/mellea_langchain/message_conversion.py
def langchain_to_mellea_messages(messages: list[BaseMessage]) -> list[Message]:
    mellea_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            mellea_messages.append(Message(role="system", content=str(msg.content)))
        elif isinstance(msg, HumanMessage):
            mellea_messages.append(Message(role="user", content=str(msg.content)))
        # ... more conversion logic
    return mellea_messages

def mellea_to_langchain_result(response: Any) -> ChatResult:
    content = response.content if hasattr(response, "content") else str(response)
    message = AIMessage(content=content)
    generation = ChatGeneration(message=message)
    return ChatResult(generations=[generation])
```

#### After
```python
# langchain/src/mellea_langchain/message_conversion.py
from mellea_integration import BaseMessageConverter
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class LangChainMessageConverter(BaseMessageConverter):
    """Convert between LangChain and Mellea message formats."""
    
    def to_mellea(self, messages: list[BaseMessage]) -> list[Any]:
        """Convert LangChain messages to Mellea format."""
        mellea_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                mellea_messages.append(
                    self.create_mellea_message("system", str(msg.content))
                )
            elif isinstance(msg, HumanMessage):
                mellea_messages.append(
                    self.create_mellea_message("user", str(msg.content))
                )
            elif isinstance(msg, AIMessage):
                mellea_messages.append(
                    self.create_mellea_message("assistant", str(msg.content))
                )
        return mellea_messages
    
    def from_mellea(self, response: Any) -> ChatResult:
        """Convert Mellea response to LangChain format."""
        content = self.extract_content_from_response(response)
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
```

**Key Changes:**
- Extend `BaseMessageConverter`
- Use `create_mellea_message()` utility
- Use `extract_content_from_response()` utility
- Implement `to_mellea()` and `from_mellea()` methods

### Step 3: Create Tool Converter (Optional)

If your integration supports tools:

#### Before
```python
def langchain_to_mellea_tools(tools: list[Any]) -> list[Any]:
    mellea_tools = []
    for tool in tools:
        # Complex conversion logic...
        mellea_tool = MelleaTool.from_langchain(tool)
        mellea_tools.append(mellea_tool)
    return mellea_tools
```

#### After
```python
from mellea_integration import BaseToolConverter

class LangChainToolConverter(BaseToolConverter):
    """Convert between LangChain and Mellea tool formats."""
    
    def to_mellea(self, tools: list[Any]) -> list[Any]:
        """Convert LangChain tools to Mellea format."""
        mellea_tools = []
        for tool in tools:
            # Use built-in utilities
            schema = self.extract_tool_schema(tool)
            callable_func = self.get_tool_callable(tool)
            
            mellea_tool = MelleaTool.from_langchain(tool)
            mellea_tools.append(mellea_tool)
        return mellea_tools
```

### Step 4: Refactor Main Integration Class

Replace your integration class to extend `MelleaIntegrationBase`:

#### Before
```python
class MelleaChatModel(BaseChatModel):
    def __init__(self, mellea_session: Any, **kwargs):
        super().__init__(**kwargs)
        self.mellea_session = mellea_session
    
    def _generate(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
        # Convert messages
        mellea_messages = langchain_to_mellea_messages(messages)
        
        # Extract prompt
        last_message_content = mellea_messages[-1].content
        
        # Prepare model options
        model_options = kwargs.get("model_options", {}).copy()
        
        # Handle requirements/strategy
        requirements = kwargs.get("requirements")
        strategy = kwargs.get("strategy")
        
        # Choose method
        if requirements or strategy:
            response = self.mellea_session.instruct(
                last_message_content,
                requirements=requirements,
                strategy=strategy,
                model_options=model_options
            )
        else:
            response = self.mellea_session.chat(
                last_message_content,
                model_options=model_options
            )
        
        # Convert response
        return mellea_to_langchain_result(response)
```

#### After
```python
from mellea_integration import MelleaIntegrationBase
from langchain_core.language_models import BaseChatModel

class MelleaChatModel(BaseChatModel, MelleaIntegrationBase):
    def __init__(self, mellea_session: Any, **kwargs):
        # Initialize base integration
        MelleaIntegrationBase.__init__(
            self,
            mellea_session=mellea_session,
            message_converter=LangChainMessageConverter(),
            tool_converter=LangChainToolConverter(),
            **kwargs
        )
        # Initialize LangChain base
        BaseChatModel.__init__(self, **kwargs)
    
    def _generate(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
        # Use base class preparation (handles everything!)
        prompt, model_options, tool_calls_enabled = self._prepare_generation(
            messages, kwargs.get("tools"), **kwargs
        )
        
        # Generate with Mellea (handles chat vs instruct automatically!)
        response = self._generate_with_mellea(
            prompt, model_options, tool_calls_enabled,
            kwargs.get("requirements"), kwargs.get("strategy")
        )
        
        # Convert response (uses your converter)
        return self.message_converter.from_mellea(response)
```

**Key Changes:**
- Inherit from both framework base class AND `MelleaIntegrationBase`
- Initialize both base classes in `__init__`
- Use `_prepare_generation()` instead of manual conversion
- Use `_generate_with_mellea()` instead of manual method selection
- Much simpler and cleaner!

### Step 5: Update Async Methods

#### Before
```python
async def _agenerate(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
    # Duplicate all the logic from _generate but with async...
    mellea_messages = langchain_to_mellea_messages(messages)
    # ... 30+ lines of duplicated code
```

#### After
```python
async def _agenerate(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
    # Same pattern as sync, just use async method
    prompt, model_options, tool_calls_enabled = self._prepare_generation(
        messages, kwargs.get("tools"), **kwargs
    )
    
    response = await self._agenerate_with_mellea(
        prompt, model_options, tool_calls_enabled,
        kwargs.get("requirements"), kwargs.get("strategy")
    )
    
    return self.message_converter.from_mellea(response)
```

## Complete Example: LangChain Integration

Here's a complete before/after comparison:

### Before: ~500 lines
- `chat_model.py`: 300 lines
- `message_conversion.py`: 150 lines
- `tool_conversion.py`: 50 lines

### After: ~200 lines
```python
# message_conversion.py (~80 lines)
from mellea_integration import BaseMessageConverter

class LangChainMessageConverter(BaseMessageConverter):
    def to_mellea(self, messages): ...
    def from_mellea(self, response): ...

# tool_conversion.py (~30 lines)
from mellea_integration import BaseToolConverter

class LangChainToolConverter(BaseToolConverter):
    def to_mellea(self, tools): ...

# chat_model.py (~90 lines)
from mellea_integration import MelleaIntegrationBase

class MelleaChatModel(BaseChatModel, MelleaIntegrationBase):
    def __init__(self, mellea_session, **kwargs):
        MelleaIntegrationBase.__init__(
            self, mellea_session,
            LangChainMessageConverter(),
            LangChainToolConverter(),
            **kwargs
        )
        BaseChatModel.__init__(self, **kwargs)
    
    def _generate(self, messages, **kwargs):
        prompt, opts, tools_enabled = self._prepare_generation(messages, **kwargs)
        response = self._generate_with_mellea(prompt, opts, tools_enabled)
        return self.message_converter.from_mellea(response)
    
    async def _agenerate(self, messages, **kwargs):
        prompt, opts, tools_enabled = self._prepare_generation(messages, **kwargs)
        response = await self._agenerate_with_mellea(prompt, opts, tools_enabled)
        return self.message_converter.from_mellea(response)
```

## Testing Migration

After migration, ensure:

1. **All existing tests pass**
2. **Message conversion works correctly**
3. **Tool calling works (if applicable)**
4. **Requirements/strategy work**
5. **Async methods work**

## Troubleshooting

### Issue: Import errors
**Solution**: Ensure `mellea-integration-core` is installed:
```bash
pip install mellea-integration-core
```

### Issue: Type errors with message converter
**Solution**: Ensure your converter implements both `to_mellea()` and `from_mellea()`:
```python
class MyConverter(BaseMessageConverter):
    def to_mellea(self, messages): ...  # Required
    def from_mellea(self, response): ...  # Required
```

### Issue: Tool conversion not working
**Solution**: Ensure you pass the tool converter to the base class:
```python
MelleaIntegrationBase.__init__(
    self,
    mellea_session=mellea_session,
    message_converter=MyMessageConverter(),
    tool_converter=MyToolConverter(),  # Don't forget this!
)
```

## Next Steps

After successful migration:

1. **Remove old utility files** that are now in the core package
2. **Update documentation** to reference the core package
3. **Update examples** to show the new pattern
4. **Consider contributing** improvements back to the core package

## Support

- **Issues**: Report problems on GitHub
- **Discord**: Join the Mellea Discord for help
- **Documentation**: See the core package README

---

**Migration completed?** Your integration should now be cleaner, more maintainable, and consistent with other Mellea integrations!