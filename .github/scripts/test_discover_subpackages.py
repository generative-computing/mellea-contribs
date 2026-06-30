"""Tests for discover_subpackages.py."""

from __future__ import annotations

from pathlib import Path

from discover_subpackages import DiscoverInputs, discover


def test_docs_only_pr_runs_nothing() -> None:
    inputs = DiscoverInputs(
        changed_files=["README.md", "RFCs/foo.md", "docs/bar.md"],
        base_ref="main",
        repo_root=Path("/dummy"),
    )
    result = discover(inputs)
    assert result.matrix == []
    assert result.run_template_smoke is False
    assert result.reason == "docs-only"


def test_cookiecutter_only_runs_template_smoke() -> None:
    inputs = DiscoverInputs(
        changed_files=["cookiecutter/cookiecutter.json"],
        base_ref="main",
        repo_root=Path("/dummy"),
    )
    result = discover(inputs)
    assert result.matrix == []
    assert result.run_template_smoke is True


def test_root_pyproject_runs_all_packages() -> None:
    inputs = DiscoverInputs(
        changed_files=["pyproject.toml"],
        base_ref="main",
        repo_root=Path("/dummy"),
    )
    result = discover(inputs, all_subpackages=["dspy", "tools"])
    assert sorted(result.matrix) == ["dspy", "tools"]


def test_workflow_change_runs_all_packages() -> None:
    inputs = DiscoverInputs(
        changed_files=[".github/workflows/ci.yml"],
        base_ref="main",
        repo_root=Path("/dummy"),
    )
    result = discover(inputs, all_subpackages=["dspy", "tools"])
    assert sorted(result.matrix) == ["dspy", "tools"]


def test_subpackage_change_runs_only_that_subpackage() -> None:
    inputs = DiscoverInputs(
        changed_files=["dspy/stdlib/frameworks/dspy/integration.py"],
        base_ref="main",
        repo_root=Path("/dummy"),
    )
    result = discover(inputs, all_subpackages=["dspy", "tools"])
    assert result.matrix == ["dspy"]


def test_union_of_categories(tmp_path: Path) -> None:
    inputs = DiscoverInputs(
        changed_files=["cookiecutter/foo.py", "dspy/stdlib/frameworks/dspy/x.py"],
        base_ref="main",
        repo_root=Path("/dummy"),
    )
    result = discover(inputs, all_subpackages=["dspy", "tools"])
    assert result.matrix == ["dspy"]
    assert result.run_template_smoke is True


def test_stacked_pr_promotes_to_all_packages() -> None:
    """When base_ref != 'main', promote to all-packages."""
    inputs = DiscoverInputs(
        changed_files=["dspy/stdlib/frameworks/dspy/x.py"],
        base_ref="feat/restructure-validate-structure",  # stacked PR
        repo_root=Path("/dummy"),
    )
    result = discover(inputs, all_subpackages=["dspy", "tools"])
    assert sorted(result.matrix) == ["dspy", "tools"]
    assert "stacked" in result.reason.lower()


def test_legacy_subpackage_change_does_not_trigger_new_ci() -> None:
    """Changes under mellea_contribs/<old>/ are routed to legacy-ci.yml."""
    inputs = DiscoverInputs(
        changed_files=["mellea_contribs/dspy_backend/src/foo.py"],
        base_ref="main",
        repo_root=Path("/dummy"),
    )
    result = discover(inputs, all_subpackages=["dspy", "tools"])
    assert result.matrix == []
    assert result.reason == "legacy-only-no-new-ci"
