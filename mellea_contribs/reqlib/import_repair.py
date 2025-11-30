"""Import repair requirement for LLM-generated Python code.

This module provides a Requirement class that validates Python imports
and provides actionable feedback for the repair loop when issues are found.
"""

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement, ValidationResult

from .import_resolution import (
    ImportSuggestion,
    find_undefined_names,
    get_installed_packages,
    is_module_available,
    parse_execution_error,
    resolve_attribute_error,
    resolve_import_error,
    resolve_module_not_found,
    resolve_undefined_name,
)


def _score_code_block(code: str) -> int:
    """Score a code block to determine if it's likely the main answer.

    Args:
        code: The code block to score.

    Returns:
        Score indicating likelihood this is the primary code block.
    """
    score = 0
    lines = code.split("\n")

    # Longer blocks generally better
    score += min(len(lines), 10)

    # Prefer complete functions/classes
    if "def " in code or "class " in code:
        score += 5

    # Prefer blocks with actual logic
    if any(keyword in code for keyword in ["if ", "for ", "while ", "try:", "with "]):
        score += 3

    # Penalize blocks that are mostly imports/comments
    non_trivial_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith(("#", "import ", "from "))
    ]
    if len(non_trivial_lines) < 2:
        score -= 5

    return score


def extract_python_code(text: str) -> str | None:
    """Extract Python code from markdown code blocks or plain text.

    Uses intelligent extraction strategy:
    1. Finds all ```python...``` blocks
    2. Scores each block (prefers longer, non-test code after positive cues)
    3. Returns highest-scoring block
    4. Falls back to generic blocks or raw text

    Args:
        text: The text to extract Python code from.

    Returns:
        The extracted Python code, or None if no code found.
    """
    # Try explicit python code blocks first (case-insensitive)
    python_block_pattern = r"```[Pp]ython\s*\n(.*?)```"
    matches = re.findall(python_block_pattern, text, re.DOTALL)

    if matches:
        if len(matches) == 1:
            return matches[0].strip()

        # Multiple blocks - need to be smart about which one
        best_block = None
        best_score = -999

        for match in matches:
            match_pos = text.find(f"```python\n{match}")
            context_before = text[max(0, match_pos - 200) : match_pos]

            score = _score_code_block(match) + (
                5 if "correct" in context_before.lower() else 0
            )

            if score > best_score:
                best_score = score
                best_block = match

        return best_block.strip() if best_block else matches[0].strip()

    # Try generic code blocks
    generic_block_pattern = r"```\s*\n(.*?)```"
    matches = re.findall(generic_block_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            candidate = match.strip()
            if any(
                keyword in candidate
                for keyword in [
                    "def ",
                    "class ",
                    "import ",
                    "from ",
                    "if ",
                    "for ",
                    "while ",
                ]
            ):
                return candidate

    # If no code blocks, check if entire text looks like Python
    stripped_text = text.strip()
    if any(
        keyword in stripped_text for keyword in ["def ", "class ", "import ", "from "]
    ):
        return stripped_text

    return None


class PythonImportRepair(Requirement):
    """Validates Python code imports and provides actionable repair feedback.

    This requirement detects missing or incorrect imports in LLM-generated
    Python code and provides detailed suggestions for fixing them. It follows
    the mellea pattern of providing feedback via ValidationResult.reason for
    use with RepairTemplateStrategy.

    Two detection modes are available:
    - Static analysis (default): Uses AST to detect undefined names and checks
      if imported modules exist. Safe, fast, no code execution.
    - Execution-based: Runs the code and parses runtime errors. More
      comprehensive but requires executing untrusted code.

    Example:
        ```python
        from mellea_contribs.reqlib import PythonImportRepair
        from mellea.stdlib.sampling import RepairTemplateStrategy

        result = session.instruct(
            "Write a function to analyze data using pandas and numpy",
            requirements=[PythonImportRepair()],
            strategy=RepairTemplateStrategy(loop_budget=3)
        )
        ```
    """

    def __init__(
        self,
        timeout: int = 5,
        allow_unsafe_execution: bool = False,
        use_sandbox: bool = False,
        max_suggestions: int = 3,
    ):
        """Initialize the import repair requirement.

        Args:
            timeout: Maximum seconds for code execution (if enabled).
            allow_unsafe_execution: If True, execute code to detect runtime errors.
                If False (default), only use static analysis.
            use_sandbox: If True with allow_unsafe_execution, use Docker sandbox
                via llm-sandbox (must be installed separately).
            max_suggestions: Maximum number of suggestions per error.
        """
        self._timeout = timeout
        self._allow_unsafe = allow_unsafe_execution
        self._use_sandbox = use_sandbox
        self._max_suggestions = max_suggestions
        self._installed_packages: set[str] | None = None

        mode = "execution-based" if allow_unsafe_execution else "static analysis"
        super().__init__(
            description=f"Python code should have all necessary imports ({mode}).",
            validation_fn=lambda ctx: self._validate_imports(ctx),
            check_only=True,  # Don't include in prompt to avoid purple elephant effects
        )

    def _get_installed_packages(self) -> set[str]:
        """Lazy-load installed packages list."""
        if self._installed_packages is None:
            self._installed_packages = get_installed_packages()
        return self._installed_packages

    def _validate_imports(self, ctx: Context) -> ValidationResult:
        """Validate imports and provide repair feedback.

        Args:
            ctx: The mellea Context containing the LLM output.

        Returns:
            ValidationResult with success/failure and actionable feedback.
        """
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        code = extract_python_code(last_output.value)
        if not code:
            return ValidationResult(
                result=False,
                reason="Could not extract Python code for import validation",
            )

        # Check syntax first
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                result=False,
                reason=f"Syntax error at line {e.lineno}: {e.msg} - fix syntax before checking imports",
            )

        if self._allow_unsafe:
            return self._validate_via_execution(code)
        else:
            return self._validate_via_static_analysis(code)

    def _validate_via_static_analysis(self, code: str) -> ValidationResult:
        """Validate imports using static AST analysis.

        Detects:
        - Undefined names (using AST analysis)
        - Unavailable imported modules (using importlib)

        Args:
            code: The Python code to validate.

        Returns:
            ValidationResult with suggestions for any issues found.
        """
        issues: list[tuple[str, list[ImportSuggestion]]] = []
        packages = self._get_installed_packages()

        # 1. Find undefined names via AST
        try:
            undefined_names = find_undefined_names(code)
            for name in undefined_names:
                suggestions = resolve_undefined_name(name, packages)
                if suggestions:
                    issues.append((f"Undefined name: '{name}'", suggestions))
        except SyntaxError:
            pass  # Already checked above

        # 2. Check if imported modules exist
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split(".")[0]
                        if not is_module_available(module_name):
                            suggestions = resolve_module_not_found(
                                module_name, packages
                            )
                            issues.append(
                                (f"Module not found: '{module_name}'", suggestions)
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split(".")[0]
                        if not is_module_available(module_name):
                            suggestions = resolve_module_not_found(
                                module_name, packages
                            )
                            issues.append(
                                (f"Module not found: '{module_name}'", suggestions)
                            )
        except SyntaxError:
            pass

        if not issues:
            return ValidationResult(
                result=True, reason="All imports appear valid (static analysis)"
            )

        return self._format_feedback(issues)

    def _validate_via_execution(self, code: str) -> ValidationResult:
        """Validate imports by executing the code.

        Runs the code in a subprocess and parses any import-related errors
        from stderr.

        Args:
            code: The Python code to validate.

        Returns:
            ValidationResult with suggestions for any issues found.
        """
        if self._use_sandbox:
            return self._execute_in_sandbox(code)
        else:
            return self._execute_in_subprocess(code)

    def _execute_in_subprocess(self, code: str) -> ValidationResult:
        """Execute code in a subprocess and parse errors.

        Args:
            code: The Python code to execute.

        Returns:
            ValidationResult with suggestions for any issues found.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )

            if result.returncode == 0:
                return ValidationResult(
                    result=True,
                    reason="Code executed successfully - all imports resolved",
                )

            # Parse errors from stderr
            error_text = result.stderr
            return self._parse_and_suggest(error_text)

        except subprocess.TimeoutExpired:
            return ValidationResult(
                result=False,
                reason=f"Execution timed out after {self._timeout} seconds",
            )
        except Exception as e:
            return ValidationResult(result=False, reason=f"Execution error: {e!s}")
        finally:
            try:
                Path(temp_file).unlink()
            except Exception:
                pass

    def _execute_in_sandbox(self, code: str) -> ValidationResult:
        """Execute code in a Docker sandbox via llm-sandbox.

        Args:
            code: The Python code to execute.

        Returns:
            ValidationResult with suggestions for any issues found.
        """
        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            return ValidationResult(
                result=False,
                reason="llm-sandbox not installed. Install with: uv add 'llm-sandbox[docker]'",
            )

        try:
            with SandboxSession(
                lang="python", verbose=False, keep_template=False
            ) as session:
                result = session.run(code, timeout=self._timeout)

                if result.exit_code == 0:
                    return ValidationResult(
                        result=True,
                        reason="Code executed successfully in sandbox - all imports resolved",
                    )

                error_text = result.stderr if result.stderr else ""
                return self._parse_and_suggest(error_text)

        except Exception as e:
            return ValidationResult(
                result=False, reason=f"Sandbox execution error: {e!s}"
            )

    def _parse_and_suggest(self, error_text: str) -> ValidationResult:
        """Parse error text and generate suggestions.

        Args:
            error_text: The stderr output from code execution.

        Returns:
            ValidationResult with suggestions for any import issues found.
        """
        import_errors = parse_execution_error(error_text)

        if not import_errors:
            # Execution failed but not due to import errors
            return ValidationResult(
                result=False,
                reason=f"Execution error (not import-related): {error_text[:300]}",
            )

        packages = self._get_installed_packages()
        issues: list[tuple[str, list[ImportSuggestion]]] = []

        for error in import_errors:
            suggestions: list[ImportSuggestion] = []

            if error.error_type == "module_not_found":
                suggestions = resolve_module_not_found(error.name, packages)
            elif error.error_type == "name_error":
                suggestions = resolve_undefined_name(error.name, packages)
            elif error.error_type == "import_error":
                suggestions = resolve_import_error(
                    error.name, error.from_module or "", packages
                )
            elif error.error_type == "attribute_error":
                suggestions = resolve_attribute_error(
                    error.name, error.from_module or "", packages
                )

            issues.append((error.original_error, suggestions))

        return self._format_feedback(issues)

    def _format_feedback(
        self, issues: list[tuple[str, list[ImportSuggestion]]]
    ) -> ValidationResult:
        """Format issues and suggestions into repair feedback.

        Args:
            issues: List of (error description, suggestions) tuples.

        Returns:
            ValidationResult with formatted feedback in the reason field.
        """
        feedback_lines = ["The following import issues were detected:"]

        for error_desc, suggestions in issues:
            feedback_lines.append(f"\n* {error_desc}")

            for suggestion in suggestions[: self._max_suggestions]:
                feedback_lines.append(
                    f"  - Suggestion: `{suggestion.import_statement}` ({suggestion.reason})"
                )

        return ValidationResult(result=False, reason="\n".join(feedback_lines))
