"""Tests for import repair functionality."""

import pytest

from mellea_contribs.reqlib.common_aliases import COMMON_ALIASES, MODULE_RELOCATIONS
from mellea_contribs.reqlib.import_resolution import (
    ImportIssue,
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
