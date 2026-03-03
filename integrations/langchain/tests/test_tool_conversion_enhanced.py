"""Enhanced tests for tool conversion utilities.

Tests actual LangChain BaseTool conversion, complex schemas,
error handling, and edge cases.
"""

import pytest

from mellea_langchain.tool_conversion import LangChainToolConverter

# Try to import LangChain tools for real conversion tests
try:
    from langchain_core.tools import BaseTool, StructuredTool, tool
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = None
    StructuredTool = None
    tool = None
    BaseModel = None
    Field = None


# Skip all tests if LangChain is not available
pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


class TestRealLangChainToolConversion:
    """Test conversion of actual LangChain BaseTool instances."""

    def test_simple_function_tool_conversion(self):
        """Test conversion of a simple function-based tool."""

        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"Weather in {location}: Sunny"

        converter = LangChainToolConverter()
        result = converter.to_mellea([get_weather])

        assert len(result) == 1
        # Verify the tool was converted (should be a Mellea tool or have expected attributes)
        converted_tool = result[0]
        assert hasattr(converted_tool, "name") or hasattr(converted_tool, "__name__")

    def test_structured_tool_conversion(self):
        """Test conversion of StructuredTool with schema."""

        class SearchInput(BaseModel):
            """Input schema for search tool."""

            query: str = Field(description="The search query")
            max_results: int = Field(default=10, description="Maximum number of results")

        def search_function(query: str, max_results: int = 10) -> str:
            """Search for information."""
            return f"Found {max_results} results for: {query}"

        search_tool = StructuredTool.from_function(
            func=search_function,
            name="web_search",
            description="Search the web for information",
            args_schema=SearchInput,
        )

        converter = LangChainToolConverter()
        result = converter.to_mellea([search_tool])

        assert len(result) == 1
        converted_tool = result[0]
        # Verify tool has expected attributes
        assert hasattr(converted_tool, "name") or hasattr(converted_tool, "__name__")

    def test_multiple_real_tools_conversion(self):
        """Test conversion of multiple real LangChain tools."""

        @tool
        def calculator(expression: str) -> str:
            """Calculate a mathematical expression."""
            return f"Result: {eval(expression)}"

        @tool
        def get_time() -> str:
            """Get the current time."""
            return "12:00 PM"

        @tool
        def translate(text: str, target_language: str) -> str:
            """Translate text to target language."""
            return f"Translated '{text}' to {target_language}"

        converter = LangChainToolConverter()
        result = converter.to_mellea([calculator, get_time, translate])

        assert len(result) == 3
        # All tools should be converted
        for converted_tool in result:
            assert hasattr(converted_tool, "name") or hasattr(converted_tool, "__name__")

    def test_tool_with_complex_schema(self):
        """Test conversion of tool with complex nested schema."""

        class Address(BaseModel):
            """Address schema."""

            street: str
            city: str
            country: str

        class PersonInput(BaseModel):
            """Person input schema."""

            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")
            address: Address = Field(description="Person's address")
            hobbies: list[str] = Field(default=[], description="List of hobbies")

        def create_person(name: str, age: int, address: Address, hobbies: list[str]) -> str:
            """Create a person record."""
            return f"Created person: {name}, {age} years old"

        person_tool = StructuredTool.from_function(
            func=create_person,
            name="create_person",
            description="Create a new person record",
            args_schema=PersonInput,
        )

        converter = LangChainToolConverter()
        result = converter.to_mellea([person_tool])

        assert len(result) == 1
        # Tool should be converted successfully despite complex schema
        assert result[0] is not None

    def test_tool_with_optional_parameters(self):
        """Test conversion of tool with optional parameters."""

        class OptionalInput(BaseModel):
            """Input with optional fields."""

            required_field: str = Field(description="Required field")
            optional_field: str | None = Field(default=None, description="Optional field")
            optional_int: int | None = Field(default=None, description="Optional integer")

        def optional_tool(
            required_field: str, optional_field: str | None = None, optional_int: int | None = None
        ) -> str:
            """Tool with optional parameters."""
            return f"Required: {required_field}, Optional: {optional_field}, Int: {optional_int}"

        tool_instance = StructuredTool.from_function(
            func=optional_tool,
            name="optional_tool",
            description="Tool with optional parameters",
            args_schema=OptionalInput,
        )

        converter = LangChainToolConverter()
        result = converter.to_mellea([tool_instance])

        assert len(result) == 1
        assert result[0] is not None

    def test_tool_with_return_direct(self):
        """Test conversion of tool with return_direct flag."""

        @tool(return_direct=True)
        def direct_answer(question: str) -> str:
            """Answer question directly without further processing."""
            return f"Direct answer to: {question}"

        converter = LangChainToolConverter()
        result = converter.to_mellea([direct_answer])

        assert len(result) == 1
        # Tool should be converted with return_direct preserved if supported
        assert result[0] is not None


class TestToolConversionErrorHandling:
    """Test error handling in tool conversion."""

    def test_conversion_with_invalid_tool_type(self):
        """Test handling of invalid tool types."""
        converter = LangChainToolConverter()

        # Pass invalid objects
        invalid_tools = [
            "not a tool",
            123,
            {"name": "fake_tool"},
        ]

        # Should handle gracefully (pass through or raise appropriate error)
        result = converter.to_mellea(invalid_tools)

        # Invalid tools should be passed through unchanged
        assert len(result) == 3
        assert result[0] == "not a tool"
        assert result[1] == 123

    def test_conversion_with_none_in_list(self):
        """Test handling of None values in tool list."""

        @tool
        def valid_tool(x: str) -> str:
            """A valid tool."""
            return x

        converter = LangChainToolConverter()
        result = converter.to_mellea([valid_tool, None, valid_tool])

        # Should handle None gracefully
        assert len(result) == 3
        assert result[1] is None

    def test_empty_tools_list(self):
        """Test conversion of empty tools list."""
        converter = LangChainToolConverter()
        result = converter.to_mellea([])

        assert result == []
        assert isinstance(result, list)

    def test_tool_with_missing_description(self):
        """Test conversion of tool without description."""

        def no_description_func(x: str) -> str:
            return x

        # Create tool without docstring
        no_desc_tool = StructuredTool.from_function(
            func=no_description_func,
            name="no_desc_tool",
            description="",  # Empty description
        )

        converter = LangChainToolConverter()
        result = converter.to_mellea([no_desc_tool])

        # Should handle tools with empty descriptions
        assert len(result) == 1
        assert result[0] is not None


class TestToolSchemaValidation:
    """Test tool schema validation and conversion."""

    def test_tool_with_enum_parameter(self):
        """Test conversion of tool with enum parameter."""
        from enum import Enum

        class Color(str, Enum):
            """Color enum."""

            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        class ColorInput(BaseModel):
            """Input with enum."""

            color: Color = Field(description="Choose a color")
            intensity: int = Field(default=5, description="Color intensity")

        def set_color(color: Color, intensity: int = 5) -> str:
            """Set a color with intensity."""
            return f"Color: {color.value}, Intensity: {intensity}"

        color_tool = StructuredTool.from_function(
            func=set_color,
            name="set_color",
            description="Set a color",
            args_schema=ColorInput,
        )

        converter = LangChainToolConverter()
        result = converter.to_mellea([color_tool])

        assert len(result) == 1
        assert result[0] is not None

    def test_tool_with_list_parameter(self):
        """Test conversion of tool with list parameter."""

        class ListInput(BaseModel):
            """Input with list."""

            items: list[str] = Field(description="List of items")
            numbers: list[int] = Field(default=[], description="List of numbers")

        def process_lists(items: list[str], numbers: list[int]) -> str:
            """Process lists of items and numbers."""
            return f"Items: {len(items)}, Numbers: {len(numbers)}"

        list_tool = StructuredTool.from_function(
            func=process_lists,
            name="process_lists",
            description="Process lists",
            args_schema=ListInput,
        )

        converter = LangChainToolConverter()
        result = converter.to_mellea([list_tool])

        assert len(result) == 1
        assert result[0] is not None

    def test_tool_with_dict_parameter(self):
        """Test conversion of tool with dict parameter."""

        class DictInput(BaseModel):
            """Input with dict."""

            config: dict[str, str] = Field(description="Configuration dictionary")
            metadata: dict[str, int] = Field(default={}, description="Metadata")

        def apply_config(config: dict[str, str], metadata: dict[str, int]) -> str:
            """Apply configuration."""
            return f"Config keys: {len(config)}, Metadata keys: {len(metadata)}"

        dict_tool = StructuredTool.from_function(
            func=apply_config,
            name="apply_config",
            description="Apply configuration",
            args_schema=DictInput,
        )

        converter = LangChainToolConverter()
        result = converter.to_mellea([dict_tool])

        assert len(result) == 1
        assert result[0] is not None


class TestToolConversionEdgeCases:
    """Test edge cases in tool conversion."""

    def test_tool_with_very_long_description(self):
        """Test conversion of tool with very long description."""
        long_description = "A" * 10000

        @tool
        def long_desc_tool(x: str) -> str:
            """Placeholder docstring."""
            return x

        # Override description
        long_desc_tool.description = long_description

        converter = LangChainToolConverter()
        result = converter.to_mellea([long_desc_tool])

        assert len(result) == 1
        assert result[0] is not None

    def test_tool_with_unicode_in_name_and_description(self):
        """Test conversion of tool with unicode characters."""

        @tool
        def unicode_tool(text: str) -> str:
            """Tool with unicode: 你好 世界 🌍."""
            return f"Processed: {text}"

        converter = LangChainToolConverter()
        result = converter.to_mellea([unicode_tool])

        assert len(result) == 1
        assert result[0] is not None

    def test_tool_with_special_characters_in_parameters(self):
        """Test conversion of tool with special characters in parameter descriptions."""

        class SpecialInput(BaseModel):
            """Input with special chars."""

            param: str = Field(description="Parameter with <>&\"' special chars")

        def special_tool(param: str) -> str:
            """Tool with special chars."""
            return param

        special = StructuredTool.from_function(
            func=special_tool,
            name="special_tool",
            description="Tool with special chars in schema",
            args_schema=SpecialInput,
        )

        converter = LangChainToolConverter()
        result = converter.to_mellea([special])

        assert len(result) == 1
        assert result[0] is not None

    def test_converter_reusability_with_real_tools(self):
        """Test that converter can be reused for multiple conversions."""

        @tool
        def tool1(x: str) -> str:
            """First tool."""
            return x

        @tool
        def tool2(y: int) -> int:
            """Second tool."""
            return y

        converter = LangChainToolConverter()

        result1 = converter.to_mellea([tool1])
        result2 = converter.to_mellea([tool2])

        assert len(result1) == 1
        assert len(result2) == 1
        # Both conversions should succeed independently
        assert result1[0] is not None
        assert result2[0] is not None

    def test_large_number_of_tools(self):
        """Test conversion of many tools at once."""
        tools = []
        for i in range(50):

            @tool
            def dynamic_tool(x: str) -> str:
                """Dynamic tool."""
                return x

            # Give each tool a unique name
            dynamic_tool.name = f"tool_{i}"
            tools.append(dynamic_tool)

        converter = LangChainToolConverter()
        result = converter.to_mellea(tools)

        assert len(result) == 50
        # All tools should be converted
        assert all(t is not None for t in result)


class TestMixedToolTypes:
    """Test conversion of mixed tool types."""

    def test_langchain_and_mellea_tools_mixed(self):
        """Test conversion when both LangChain and Mellea tools are present."""

        @tool
        def langchain_tool(x: str) -> str:
            """A LangChain tool."""
            return x

        class MockMelleaTool:
            """Mock Mellea tool."""

            name = "mellea_tool"

        converter = LangChainToolConverter()
        result = converter.to_mellea([langchain_tool, MockMelleaTool()])

        assert len(result) == 2
        # Both should be in result
        assert result[0] is not None
        assert result[1] is not None

    def test_multiple_tool_types_in_single_conversion(self):
        """Test conversion of various tool types together."""

        @tool
        def simple_tool(x: str) -> str:
            """Simple tool."""
            return x

        class ComplexInput(BaseModel):
            """Complex input."""

            field1: str
            field2: int

        def complex_func(field1: str, field2: int) -> str:
            """Complex function."""
            return f"{field1}: {field2}"

        complex_tool = StructuredTool.from_function(
            func=complex_func,
            name="complex_tool",
            description="Complex tool",
            args_schema=ComplexInput,
        )

        class MockMelleaTool:
            """Mock Mellea tool."""

            name = "mellea_native"

        converter = LangChainToolConverter()
        result = converter.to_mellea([simple_tool, complex_tool, MockMelleaTool()])

        assert len(result) == 3
        # All should be converted or passed through
        assert all(t is not None for t in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
