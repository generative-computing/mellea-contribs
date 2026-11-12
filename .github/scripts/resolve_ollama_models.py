"""Resolve and validate the ollama models a subpackage's tests need.

Most subpackages call bare ``start_session()`` in their tests, so the model
they exercise is whatever the installed mellea defaults to — not a tag CI gets
to hardcode. This module derives that default from the installed mellea by
introspecting ``start_session``'s ``model_id`` parameter, unions it with any
extra models a package declares under ``[tool.mellea-contribs.ci].ollama_models``
in its ``pyproject.toml``, and validates that CI pulled exactly that set.

CLI usage (run from a package directory so the right mellea env is active)::

    # Print the required model tags, one per line, for CI to pull.
    uv run python .github/scripts/resolve_ollama_models.py resolve .

    # `ollama pull` each required model.
    uv run python .github/scripts/resolve_ollama_models.py pull .

    # Fail if `ollama list` is not exactly the required set.
    uv run python .github/scripts/resolve_ollama_models.py validate .
"""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable


def default_ollama_model(start_session: Callable[..., Any] | None = None) -> str:
    """Return the ollama tag the installed mellea's ``start_session`` defaults to.

    Introspects the ``model_id`` parameter of ``start_session`` and reads the
    ``ollama_name`` off its default value (a ``ModelIdentifier``), or returns the
    default verbatim when it is already a plain string.

    Args:
        start_session: The ``start_session`` callable to introspect. Defaults to
            ``mellea.start_session`` (imported lazily). Injected in tests.

    Returns:
        The ollama model tag, e.g. ``"granite4.2:3b"``.

    Raises:
        RuntimeError: If ``start_session`` has no ``model_id`` parameter, or its
            default carries no usable ollama name.
    """
    if start_session is None:
        from mellea import start_session as start_session  # noqa: PLC0415

    params = inspect.signature(start_session).parameters
    if "model_id" not in params:
        raise RuntimeError(
            "start_session has no 'model_id' parameter; cannot derive the default "
            "ollama model. The mellea API may have changed."
        )

    default = params["model_id"].default
    if isinstance(default, str):
        return default

    ollama_name = getattr(default, "ollama_name", None)
    if not ollama_name:
        raise RuntimeError(
            f"start_session's default model_id ({default!r}) has no ollama name; "
            "cannot derive the model to pull."
        )
    return str(ollama_name)


def read_declared_models(pyproject_path: str | Path) -> list[str]:
    """Return the extra ollama models a package declares for its tests.

    Reads ``[tool.mellea-contribs.ci].ollama_models`` from ``pyproject_path``.
    An absent key or table means the package relies solely on the default model.

    Args:
        pyproject_path: Path to the package's ``pyproject.toml``.

    Returns:
        The declared model tags, or an empty list if none are declared.

    Raises:
        RuntimeError: If ``ollama_models`` is present but is not a list of strings.
    """
    data = tomllib.loads(Path(pyproject_path).read_text())
    ci = data.get("tool", {}).get("mellea-contribs", {}).get("ci", {})
    declared = ci.get("ollama_models", [])
    if not isinstance(declared, list) or not all(isinstance(m, str) for m in declared):
        raise RuntimeError(
            "[tool.mellea-contribs.ci].ollama_models must be a list of strings, "
            f"got {declared!r}"
        )
    return declared


def required_models(
    pyproject_path: str | Path,
    start_session: Callable[..., Any] | None = None,
) -> list[str]:
    """Return every ollama model a package's tests need, default first.

    The union of the installed mellea's default model and the package's declared
    extras, de-duplicated while preserving order (default always first).

    Args:
        pyproject_path: Path to the package's ``pyproject.toml``.
        start_session: The ``start_session`` callable to introspect. Defaults to
            ``mellea.start_session``. Injected in tests.

    Returns:
        The ordered, de-duplicated list of required model tags.

    Raises:
        RuntimeError: If the default model or declared list cannot be resolved.
    """
    ordered = [default_ollama_model(start_session=start_session)]
    ordered += read_declared_models(pyproject_path)
    seen: set[str] = set()
    deduped: list[str] = []
    for model in ordered:
        if model not in seen:
            seen.add(model)
            deduped.append(model)
    return deduped


def _normalize(tag: str) -> str:
    """Strip an implicit ``:latest`` so bare and tagged names compare equal."""
    return tag[: -len(":latest")] if tag.endswith(":latest") else tag


def validate_pulled(required: list[str], installed: list[str]) -> None:
    """Assert the installed ollama models are exactly the required set.

    Args:
        required: The model tags the package's tests need.
        installed: The model tags currently present (e.g. from ``ollama list``).

    Raises:
        RuntimeError: If any required model is missing, or any extra model was
            pulled that no package requires.
    """
    required_set = {_normalize(m) for m in required}
    installed_set = {_normalize(m) for m in installed}

    missing = required_set - installed_set
    extra = installed_set - required_set
    problems = []
    if missing:
        problems.append(f"missing required models: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected models pulled: {sorted(extra)}")
    if problems:
        raise RuntimeError(
            "ollama model validation failed — " + "; ".join(problems) + ". "
            "A test pulling a model mid-run, or a stale declaration, is the usual "
            "cause. Declare extra models in [tool.mellea-contribs.ci].ollama_models."
        )


def _installed_models() -> list[str]:
    """Return the model tags reported by ``ollama list``."""
    out = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=True
    ).stdout
    models = []
    for line in out.splitlines()[1:]:  # skip the header row
        line = line.strip()
        if line:
            models.append(line.split()[0])
    return models


def _pull_models(required: list[str]) -> None:
    """``ollama pull`` each required model, in order."""
    for model in required:
        subprocess.run(["ollama", "pull", model], check=True)


def main(argv: list[str] | None = None) -> int:
    """Run the ``resolve``, ``pull`` or ``validate`` subcommand.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 1 on a resolution/validation failure).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("resolve", "pull", "validate"):
        p = sub.add_parser(name)
        p.add_argument("package_dir", help="Path to the package directory")
    args = parser.parse_args(argv)

    pyproject = Path(args.package_dir) / "pyproject.toml"
    try:
        required = required_models(pyproject)
        if args.command == "resolve":
            print("\n".join(required))
        elif args.command == "pull":
            _pull_models(required)
        else:
            validate_pulled(required, _installed_models())
            print(f"OK — pulled exactly: {required}")
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
