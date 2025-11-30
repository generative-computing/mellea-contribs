"""Tests for import repair functionality."""

from unittest.mock import MagicMock

import pytest

from mellea_contribs.reqlib.common_aliases import COMMON_ALIASES, MODULE_RELOCATIONS
from mellea_contribs.reqlib.import_repair import PythonImportRepair, extract_python_code
from mellea_contribs.reqlib.import_resolution import (
    find_undefined_names,
    get_installed_packages,
    is_module_available,
    parse_execution_error,
    resolve_attribute_error,
    resolve_import_error,
    resolve_module_not_found,
    resolve_undefined_name,
)


class TestErrorParsing:
    """Test error message parsing."""

    def test_parse_module_not_found(self):
        """Test parsing ModuleNotFoundError."""
        error = "ModuleNotFoundError: No module named 'numppy'"
        errors = parse_execution_error(error)
        assert len(errors) == 1
        assert errors[0].error_type == "module_not_found"
        assert errors[0].name == "numppy"

    def test_parse_module_not_found_double_quotes(self):
        """Test parsing ModuleNotFoundError with double quotes."""
        error = 'ModuleNotFoundError: No module named "numppy"'
        errors = parse_execution_error(error)
        assert len(errors) == 1
        assert errors[0].error_type == "module_not_found"
        assert errors[0].name == "numppy"

    def test_parse_import_error(self):
        """Test parsing ImportError."""
        error = "ImportError: cannot import name 'LinearRegression' from 'sklearn'"
        errors = parse_execution_error(error)
        assert len(errors) == 1
        assert errors[0].error_type == "import_error"
        assert errors[0].name == "LinearRegression"
        assert errors[0].from_module == "sklearn"

    def test_parse_name_error(self):
        """Test parsing NameError."""
        error = "NameError: name 'np' is not defined"
        errors = parse_execution_error(error)
        assert len(errors) == 1
        assert errors[0].error_type == "name_error"
        assert errors[0].name == "np"

    def test_parse_attribute_error(self):
        """Test parsing AttributeError for module attributes."""
        error = "AttributeError: module 'sklearn' has no attribute 'LinearRegression'"
        errors = parse_execution_error(error)
        assert len(errors) == 1
        assert errors[0].error_type == "attribute_error"
        assert errors[0].name == "LinearRegression"
        assert errors[0].from_module == "sklearn"

    def test_parse_multiple_errors(self):
        """Test parsing multiple errors in one output."""
        error = """
        ModuleNotFoundError: No module named 'numppy'
        NameError: name 'pd' is not defined
        """
        errors = parse_execution_error(error)
        assert len(errors) == 2

    def test_parse_no_errors(self):
        """Test parsing text with no import errors."""
        error = "ValueError: invalid literal for int()"
        errors = parse_execution_error(error)
        assert len(errors) == 0


class TestUndefinedNameDetection:
    """Test AST-based undefined name detection."""

    def test_find_simple_undefined(self):
        """Test finding simple undefined names."""
        code = "x = np.array([1, 2, 3])"
        undefined = find_undefined_names(code)
        assert "np" in undefined

    def test_imported_names_not_undefined(self):
        """Test that imported names are not flagged."""
        code = """
import numpy as np
x = np.array([1, 2, 3])
"""
        undefined = find_undefined_names(code)
        assert "np" not in undefined

    def test_assigned_names_not_undefined(self):
        """Test that assigned names are not flagged."""
        code = """
x = 10
y = x + 5
"""
        undefined = find_undefined_names(code)
        assert "x" not in undefined
        assert "y" not in undefined

    def test_function_names_not_undefined(self):
        """Test that function definitions are not flagged."""
        code = """
def my_func():
    pass

my_func()
"""
        undefined = find_undefined_names(code)
        assert "my_func" not in undefined

    def test_function_params_not_undefined(self):
        """Test that function parameters are not flagged."""
        code = """
def add(a, b):
    return a + b
"""
        undefined = find_undefined_names(code)
        assert "a" not in undefined
        assert "b" not in undefined

    def test_for_loop_vars_not_undefined(self):
        """Test that for loop variables are not flagged."""
        code = """
for i in range(10):
    print(i)
"""
        undefined = find_undefined_names(code)
        assert "i" not in undefined

    def test_comprehension_vars_not_undefined(self):
        """Test that comprehension variables are not flagged."""
        code = """
squares = [x * x for x in range(10)]
"""
        undefined = find_undefined_names(code)
        assert "x" not in undefined

    def test_exception_handler_vars_not_undefined(self):
        """Test that exception handler variables are not flagged."""
        code = """
try:
    pass
except Exception as e:
    print(e)
"""
        undefined = find_undefined_names(code)
        assert "e" not in undefined

    def test_builtins_not_undefined(self):
        """Test that builtin names are not flagged."""
        code = """
x = len([1, 2, 3])
y = print("hello")
"""
        undefined = find_undefined_names(code)
        assert "len" not in undefined
        assert "print" not in undefined

    def test_multiple_undefined(self):
        """Test finding multiple undefined names."""
        code = """
x = np.array([1, 2, 3])
y = pd.DataFrame({"a": [1, 2]})
"""
        undefined = find_undefined_names(code)
        assert "np" in undefined
        assert "pd" in undefined


class TestResolution:
    """Test import suggestion resolution."""

    @pytest.fixture
    def packages(self):
        """Get installed packages."""
        return get_installed_packages()

    def test_resolve_common_alias_np(self, packages):
        """Test resolving 'np' to numpy."""
        suggestions = resolve_undefined_name("np", packages)
        assert len(suggestions) > 0
        assert any("numpy" in s.import_statement.lower() for s in suggestions)
        assert suggestions[0].confidence > 0.9

    def test_resolve_common_alias_pd(self, packages):
        """Test resolving 'pd' to pandas."""
        suggestions = resolve_undefined_name("pd", packages)
        assert len(suggestions) > 0
        assert any("pandas" in s.import_statement.lower() for s in suggestions)

    def test_resolve_common_alias_path(self, packages):
        """Test resolving 'Path' to pathlib."""
        suggestions = resolve_undefined_name("Path", packages)
        assert len(suggestions) > 0
        assert any("pathlib" in s.import_statement.lower() for s in suggestions)

    def test_resolve_sklearn_relocation(self, packages):
        """Test resolving sklearn import relocation."""
        suggestions = resolve_import_error("LinearRegression", "sklearn", packages)
        assert len(suggestions) > 0
        assert any("sklearn.linear_model" in s.import_statement for s in suggestions)

    def test_resolve_sklearn_attribute_error(self, packages):
        """Test resolving sklearn attribute error."""
        suggestions = resolve_attribute_error("LinearRegression", "sklearn", packages)
        assert len(suggestions) > 0
        assert any("sklearn.linear_model" in s.import_statement for s in suggestions)

    def test_resolve_misspelled_module(self, packages):
        """Test resolving misspelled module name."""
        # Only test if numpy is installed
        if "numpy" in packages:
            suggestions = resolve_module_not_found("numppy", packages)
            assert len(suggestions) > 0
            assert any("numpy" in s.import_statement for s in suggestions)


class TestCommonAliases:
    """Test the common aliases database."""

    def test_common_aliases_not_empty(self):
        """Test that common aliases database is populated."""
        assert len(COMMON_ALIASES) > 50

    def test_numpy_alias(self):
        """Test numpy alias exists."""
        assert "np" in COMMON_ALIASES
        assert "numpy" in COMMON_ALIASES["np"]

    def test_pandas_alias(self):
        """Test pandas alias exists."""
        assert "pd" in COMMON_ALIASES
        assert "pandas" in COMMON_ALIASES["pd"]

    def test_typing_aliases(self):
        """Test typing aliases exist."""
        assert "Optional" in COMMON_ALIASES
        assert "List" in COMMON_ALIASES
        assert "Dict" in COMMON_ALIASES

    def test_module_relocations_not_empty(self):
        """Test that module relocations database is populated."""
        assert len(MODULE_RELOCATIONS) > 0

    def test_sklearn_relocations(self):
        """Test sklearn relocations exist."""
        assert "sklearn" in MODULE_RELOCATIONS
        assert "LinearRegression" in MODULE_RELOCATIONS["sklearn"]


class TestModuleAvailability:
    """Test module availability checking."""

    def test_stdlib_available(self):
        """Test standard library modules are available."""
        assert is_module_available("os")
        assert is_module_available("sys")
        assert is_module_available("json")

    def test_nonexistent_unavailable(self):
        """Test nonexistent modules are unavailable."""
        assert not is_module_available("this_module_definitely_does_not_exist_xyz")


class TestGetInstalledPackages:
    """Test package discovery."""

    def test_returns_set(self):
        """Test that get_installed_packages returns a set."""
        packages = get_installed_packages()
        assert isinstance(packages, set)

    def test_contains_stdlib(self):
        """Test that stdlib modules are included."""
        packages = get_installed_packages()
        # Some common stdlib modules should be discoverable
        assert len(packages) > 0


class TestExtractPythonCode:
    """Test Python code extraction from markdown."""

    def test_extract_single_python_block(self):
        """Test extracting single python code block."""
        text = """Here is the code:
```python
import numpy as np
x = np.array([1, 2, 3])
```
"""
        code = extract_python_code(text)
        assert code is not None
        assert "numpy" in code
        assert "np.array" in code

    def test_extract_generic_block(self):
        """Test extracting generic code block with Python content."""
        text = """Here is the code:
```
def hello():
    print("Hello")
```
"""
        code = extract_python_code(text)
        assert code is not None
        assert "def hello" in code

    def test_extract_raw_python(self):
        """Test extracting raw Python (no code blocks)."""
        text = """import os
def main():
    pass
"""
        code = extract_python_code(text)
        assert code is not None
        assert "import os" in code

    def test_extract_multiple_blocks_picks_best(self):
        """Test that multiple blocks picks the best one."""
        text = """Here's a simple example:
```python
x = 1
```

Here's the correct solution:
```python
def process_data(data):
    for item in data:
        if item > 0:
            print(item)
    return data
```
"""
        code = extract_python_code(text)
        assert code is not None
        assert "def process_data" in code

    def test_extract_returns_none_for_no_code(self):
        """Test returning None when no code found."""
        text = "This is just regular text with no code."
        code = extract_python_code(text)
        assert code is None


class TestPythonImportRepairClass:
    """Test PythonImportRepair requirement class."""

    def test_instantiation_static_mode(self):
        """Test creating requirement in static analysis mode."""
        req = PythonImportRepair()
        assert "static analysis" in req.description
        assert req._allow_unsafe is False

    def test_instantiation_execution_mode(self):
        """Test creating requirement in execution mode."""
        req = PythonImportRepair(allow_unsafe_execution=True)
        assert "execution-based" in req.description
        assert req._allow_unsafe is True

    def test_validate_with_mock_context_valid_code(self):
        """Test validation with valid imports passes."""
        req = PythonImportRepair()

        # Mock the mellea Context
        mock_output = MagicMock()
        mock_output.value = """```python
import os
import sys

def main():
    print(os.getcwd())
    print(sys.version)
```"""
        mock_ctx = MagicMock()
        mock_ctx.last_output.return_value = mock_output

        result = req._validate_imports(mock_ctx)
        assert bool(result) is True

    def test_validate_with_mock_context_missing_import(self):
        """Test validation detects missing imports."""
        req = PythonImportRepair()

        mock_output = MagicMock()
        mock_output.value = """```python
x = np.array([1, 2, 3])
y = pd.DataFrame({"a": [1, 2]})
```"""
        mock_ctx = MagicMock()
        mock_ctx.last_output.return_value = mock_output

        result = req._validate_imports(mock_ctx)
        assert bool(result) is False
        assert "np" in result.reason or "pd" in result.reason
        assert "numpy" in result.reason or "pandas" in result.reason

    def test_validate_syntax_error(self):
        """Test validation catches syntax errors."""
        req = PythonImportRepair()

        mock_output = MagicMock()
        mock_output.value = """```python
def broken(
    print("missing paren"
```"""
        mock_ctx = MagicMock()
        mock_ctx.last_output.return_value = mock_output

        result = req._validate_imports(mock_ctx)
        assert bool(result) is False
        assert "Syntax error" in result.reason

    def test_validate_no_output(self):
        """Test validation handles missing output."""
        req = PythonImportRepair()

        mock_ctx = MagicMock()
        mock_ctx.last_output.return_value = None

        result = req._validate_imports(mock_ctx)
        assert bool(result) is False
        assert "No output" in result.reason

    def test_validate_unavailable_module(self):
        """Test validation detects unavailable modules."""
        req = PythonImportRepair()

        mock_output = MagicMock()
        mock_output.value = """```python
import nonexistent_module_xyz123
```"""
        mock_ctx = MagicMock()
        mock_ctx.last_output.return_value = mock_output

        result = req._validate_imports(mock_ctx)
        assert bool(result) is False
        assert "nonexistent_module_xyz123" in result.reason
