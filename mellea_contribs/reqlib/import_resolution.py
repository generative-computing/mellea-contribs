"""Import error parsing and resolution utilities.

This module provides functions to:
1. Parse execution errors to extract import-related issues
2. Detect undefined names via AST analysis (static mode)
3. Resolve issues to suggested import statements
"""

import ast
import builtins
import importlib.util
import pkgutil
import re
from dataclasses import dataclass
from typing import Literal

from .common_aliases import COMMON_ALIASES, MODULE_RELOCATIONS


@dataclass
class ImportIssue:
    """Represents a detected import-related error."""

    error_type: Literal[
        "module_not_found", "import_error", "name_error", "attribute_error"
    ]
    name: str
    original_error: str
    from_module: str | None = None


@dataclass
class ImportSuggestion:
    """A suggested fix for an import error."""

    import_statement: str
    confidence: float  # 0.0 to 1.0
    reason: str


def parse_execution_error(error_text: str) -> list[ImportIssue]:
    """Parse stderr output to extract import-related errors.

    Handles:
    - ModuleNotFoundError: No module named 'xxx'
    - ImportError: cannot import name 'xxx' from 'yyy'
    - NameError: name 'xxx' is not defined
    - AttributeError: module 'xxx' has no attribute 'yyy'

    Args:
        error_text: The stderr output from code execution.

    Returns:
        List of ImportIssue objects representing detected errors.
    """
    errors: list[ImportIssue] = []

    # ModuleNotFoundError: No module named 'xxx'
    module_pattern = r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"
    for match in re.finditer(module_pattern, error_text):
        errors.append(
            ImportIssue(
                error_type="module_not_found",
                name=match.group(1),
                original_error=match.group(0),
            )
        )

    # ImportError: cannot import name 'xxx' from 'yyy'
    import_pattern = (
        r"ImportError: cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]"
    )
    for match in re.finditer(import_pattern, error_text):
        errors.append(
            ImportIssue(
                error_type="import_error",
                name=match.group(1),
                original_error=match.group(0),
                from_module=match.group(2),
            )
        )

    # NameError: name 'xxx' is not defined
    name_pattern = r"NameError: name ['\"]([^'\"]+)['\"] is not defined"
    for match in re.finditer(name_pattern, error_text):
        errors.append(
            ImportIssue(
                error_type="name_error",
                name=match.group(1),
                original_error=match.group(0),
            )
        )

    # AttributeError: module 'xxx' has no attribute 'yyy'
    attr_pattern = r"AttributeError: module ['\"]([^'\"]+)['\"] has no attribute ['\"]([^'\"]+)['\"]"
    for match in re.finditer(attr_pattern, error_text):
        errors.append(
            ImportIssue(
                error_type="attribute_error",
                name=match.group(2),
                original_error=match.group(0),
                from_module=match.group(1),
            )
        )

    return errors


def find_undefined_names(code: str) -> set[str]:
    """Find names used but never defined/imported using AST analysis.

    This function performs static analysis to detect undefined names without
    executing the code. It tracks:
    - Imports (import x, from x import y)
    - Assignments (x = ...)
    - Function/class definitions
    - For loop variables
    - Comprehension variables
    - Exception handlers (except E as e)
    - With statement variables (with x as y)
    - Function parameters

    Args:
        code: Python source code to analyze.

    Returns:
        Set of names that are used but never defined.

    Raises:
        SyntaxError: If the code cannot be parsed.
    """
    tree = ast.parse(code)
    defined: set[str] = set(dir(builtins))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    defined.add(alias.asname if alias.asname else alias.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            defined.add(node.name)
            _collect_args(node.args, defined)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Lambda):
            _collect_args(node.args, defined)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.For | ast.AsyncFor):
            _collect_target_names(node.target, defined)
        elif isinstance(node, ast.comprehension):
            _collect_target_names(node.target, defined)
        elif isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                if item.optional_vars:
                    _collect_target_names(item.optional_vars, defined)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined.add(node.name)
        elif isinstance(node, ast.NamedExpr):
            _collect_target_names(node.target, defined)
        elif isinstance(node, ast.match_case):
            _collect_pattern_names(node.pattern, defined)

    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)

    return used - defined


def _collect_args(args: ast.arguments, names: set[str]) -> None:
    """Collect parameter names from function/lambda arguments.

    Args:
        args: The ast.arguments node containing parameters.
        names: Set to add discovered names to.
    """
    for arg in args.args:
        names.add(arg.arg)
    for arg in args.posonlyargs:
        names.add(arg.arg)
    for arg in args.kwonlyargs:
        names.add(arg.arg)
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)


def _collect_pattern_names(pattern: ast.pattern, names: set[str]) -> None:
    """Collect names bound by match statement patterns (Python 3.10+).

    Args:
        pattern: The AST pattern node from a match case.
        names: Set to add discovered names to.
    """
    if isinstance(pattern, ast.MatchAs):
        if pattern.name:
            names.add(pattern.name)
        if pattern.pattern:
            _collect_pattern_names(pattern.pattern, names)
    elif isinstance(pattern, ast.MatchOr):
        for p in pattern.patterns:
            _collect_pattern_names(p, names)
    elif isinstance(pattern, ast.MatchSequence):
        for p in pattern.patterns:
            _collect_pattern_names(p, names)
    elif isinstance(pattern, ast.MatchMapping):
        for p in pattern.patterns:
            _collect_pattern_names(p, names)
        if pattern.rest:
            names.add(pattern.rest)
    elif isinstance(pattern, ast.MatchClass):
        for p in pattern.patterns:
            _collect_pattern_names(p, names)
        for p in pattern.kwd_patterns:
            _collect_pattern_names(p, names)
    elif isinstance(pattern, ast.MatchStar):
        if pattern.name:
            names.add(pattern.name)


def _collect_target_names(target: ast.expr, names: set[str]) -> None:
    """Recursively collect names from assignment targets.

    Handles simple names, tuples, and lists.

    Args:
        target: The AST node representing the assignment target.
        names: Set to add discovered names to.
    """
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, ast.Tuple | ast.List):
        for elt in target.elts:
            _collect_target_names(elt, names)
    elif isinstance(target, ast.Starred):
        _collect_target_names(target.value, names)


def get_installed_packages() -> set[str]:
    """Discover all installed packages in current environment.

    Uses pkgutil to find available modules without maintaining a static database.

    Returns:
        Set of installed package names.
    """
    return {mod.name for mod in pkgutil.iter_modules() if not mod.name.startswith("_")}


def resolve_undefined_name(
    name: str, installed_packages: set[str] | None = None
) -> list[ImportSuggestion]:
    """Resolve an undefined name to possible import statements.

    Strategy:
    1. Check common aliases (high confidence)
    2. Check if it's a direct package name (high confidence)
    3. Fuzzy match against installed packages (medium confidence)

    Args:
        name: The undefined name to resolve.
        installed_packages: Set of installed packages. If None, will be discovered.

    Returns:
        List of ImportSuggestion objects, sorted by confidence.
    """
    if installed_packages is None:
        installed_packages = get_installed_packages()

    suggestions: list[ImportSuggestion] = []

    # 1. Check common aliases first (highest confidence)
    if name in COMMON_ALIASES:
        suggestions.append(
            ImportSuggestion(
                import_statement=COMMON_ALIASES[name],
                confidence=0.95,
                reason=f"'{name}' is a common alias",
            )
        )

    # 2. Check if it's a direct package name
    name_lower = name.lower()
    if name_lower in installed_packages:
        suggestions.append(
            ImportSuggestion(
                import_statement=f"import {name_lower}",
                confidence=0.9,
                reason=f"'{name_lower}' is an installed package",
            )
        )

    # 3. Fuzzy match against installed packages
    try:
        from rapidfuzz import fuzz, process

        matches = process.extract(
            name_lower, list(installed_packages), scorer=fuzz.ratio, limit=3
        )
        for match_name, score, _ in matches:
            if score > 70 and match_name != name_lower:
                suggestions.append(
                    ImportSuggestion(
                        import_statement=f"import {match_name}",
                        confidence=score / 100.0 * 0.8,
                        reason=f"Similar to '{match_name}' (similarity: {score}%)",
                    )
                )
    except ImportError:
        # rapidfuzz not installed, skip fuzzy matching
        pass

    return sorted(suggestions, key=lambda x: x.confidence, reverse=True)


def resolve_module_not_found(
    module_name: str, installed_packages: set[str] | None = None
) -> list[ImportSuggestion]:
    """Suggest corrections for ModuleNotFoundError.

    Handles:
    - Misspelled module names (fuzzy matching)
    - Common module path errors

    Args:
        module_name: The module name that was not found.
        installed_packages: Set of installed packages. If None, will be discovered.

    Returns:
        List of ImportSuggestion objects, sorted by confidence.
    """
    if installed_packages is None:
        installed_packages = get_installed_packages()

    suggestions: list[ImportSuggestion] = []

    # Fuzzy match against installed packages
    try:
        from rapidfuzz import fuzz, process

        matches = process.extract(
            module_name, list(installed_packages), scorer=fuzz.ratio, limit=3
        )

        for match_name, score, _ in matches:
            if score > 60:
                suggestions.append(
                    ImportSuggestion(
                        import_statement=f"import {match_name}",
                        confidence=score / 100.0,
                        reason=f"Did you mean '{match_name}'? (similarity: {score}%)",
                    )
                )
    except ImportError:
        # rapidfuzz not installed, skip fuzzy matching
        pass

    # Check if base module exists for submodule imports
    base_module = module_name.split(".")[0]
    if base_module in installed_packages and "." in module_name:
        suggestions.append(
            ImportSuggestion(
                import_statement=f"import {base_module}",
                confidence=0.6,
                reason=f"Base module '{base_module}' exists - check submodule path",
            )
        )

    return sorted(suggestions, key=lambda x: x.confidence, reverse=True)


def resolve_import_error(
    name: str, from_module: str, installed_packages: set[str] | None = None
) -> list[ImportSuggestion]:
    """Suggest corrections for ImportError (cannot import name from module).

    Handles common causes:
    - Wrong submodule (e.g., LinearRegression from sklearn vs sklearn.linear_model)
    - Name doesn't exist in that module

    Args:
        name: The name that could not be imported.
        from_module: The module the import was attempted from.
        installed_packages: Set of installed packages. If None, will be discovered.

    Returns:
        List of ImportSuggestion objects, sorted by confidence.
    """
    suggestions: list[ImportSuggestion] = []

    # Check module relocations database
    base_module = from_module.split(".")[0]
    if base_module in MODULE_RELOCATIONS:
        relocations = MODULE_RELOCATIONS[base_module]
        if name in relocations:
            correct_module = relocations[name]
            suggestions.append(
                ImportSuggestion(
                    import_statement=f"from {correct_module} import {name}",
                    confidence=0.95,
                    reason=f"'{name}' is located in '{correct_module}'",
                )
            )

    # Generic fallback suggestion
    if not suggestions:
        suggestions.append(
            ImportSuggestion(
                import_statement=f"# Check if '{name}' exists in '{from_module}' or its submodules",
                confidence=0.3,
                reason=f"'{name}' not found in '{from_module}' - verify correct import path",
            )
        )

    return suggestions


def resolve_attribute_error(
    attribute_name: str, module_name: str, installed_packages: set[str] | None = None
) -> list[ImportSuggestion]:
    """Suggest corrections for AttributeError on module access.

    Handles cases like `sklearn.LinearRegression` where the attribute
    is in a submodule.

    Args:
        attribute_name: The attribute that was not found.
        module_name: The module the attribute was accessed on.
        installed_packages: Set of installed packages. If None, will be discovered.

    Returns:
        List of ImportSuggestion objects, sorted by confidence.
    """
    suggestions: list[ImportSuggestion] = []

    # Check module relocations database
    base_module = module_name.split(".")[0]
    if base_module in MODULE_RELOCATIONS:
        relocations = MODULE_RELOCATIONS[base_module]
        if attribute_name in relocations:
            correct_module = relocations[attribute_name]
            suggestions.append(
                ImportSuggestion(
                    import_statement=f"from {correct_module} import {attribute_name}",
                    confidence=0.95,
                    reason=f"'{attribute_name}' is in submodule '{correct_module}'",
                )
            )

    # Also check common aliases in case it's a known name
    if attribute_name in COMMON_ALIASES:
        suggestions.append(
            ImportSuggestion(
                import_statement=COMMON_ALIASES[attribute_name],
                confidence=0.8,
                reason=f"'{attribute_name}' has a common import pattern",
            )
        )

    return sorted(suggestions, key=lambda x: x.confidence, reverse=True)


def is_module_available(module_name: str) -> bool:
    """Check if a module is available in the current environment.

    Args:
        module_name: The module name to check.

    Returns:
        True if the module is available, False otherwise.
    """
    return importlib.util.find_spec(module_name) is not None
