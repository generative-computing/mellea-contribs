"""Smoke test for {{ cookiecutter.name }}."""

import importlib


def test_import() -> None:
    """The subpackage's main module imports cleanly."""
    module = importlib.import_module(
        "mellea_contribs.{{ cookiecutter.name }}.{{ cookiecutter.core_path }}.{{ cookiecutter.name }}"
    )
    assert module.hello() == "Hello from {{ cookiecutter.name }}!"
