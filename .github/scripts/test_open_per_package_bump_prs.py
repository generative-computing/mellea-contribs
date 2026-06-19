"""Tests for open_per_package_bump_prs._select_pass and _discover_subpackages.

Side-effecting helpers (`_open_pr`, `main`) are not tested here — they
shell out to git/gh and are exercised end-to-end by the workflow. This
suite covers the pass-selection logic, which is the only piece with
non-trivial branching.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from open_per_package_bump_prs import _discover_subpackages, _select_pass


def write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip())


def _make_subpackage(repo: Path, name: str, version: str) -> None:
    write(
        repo / name / "pyproject.toml",
        f"""
        [project]
        name = "mellea-contribs-{name}"
        version = "{version}"
        dependencies = ["mellea>=0.5.0"]
        """,
    )


def test_pass1_selected_when_integration_core_lags(tmp_path: Path) -> None:
    """Pass 1 fires alone when ``_integration_core`` is behind, even if other subpackages also lag."""
    _make_subpackage(tmp_path, "_integration_core", "0.5.0")
    _make_subpackage(tmp_path, "dspy", "0.5.0")
    _make_subpackage(tmp_path, "reqlib", "0.5.0")

    selected = _select_pass(tmp_path, "0.6.0")
    assert selected is not None
    pass_name, pyprojects = selected
    assert pass_name == "integration-core"
    assert [p.parent.name for p in pyprojects] == ["_integration_core"]


def test_pass2_selected_when_integration_core_already_on_target(tmp_path: Path) -> None:
    """Once pass 1 is on target, every other subpackage opens together."""
    _make_subpackage(tmp_path, "_integration_core", "0.6.0")
    _make_subpackage(tmp_path, "dspy", "0.5.0")
    _make_subpackage(tmp_path, "langchain", "0.5.0")
    _make_subpackage(tmp_path, "crewai", "0.5.0")
    _make_subpackage(tmp_path, "agent-utilities", "0.5.0")
    _make_subpackage(tmp_path, "reqlib", "0.5.0")

    selected = _select_pass(tmp_path, "0.6.0")
    assert selected is not None
    pass_name, pyprojects = selected
    assert pass_name == "subpackages"
    assert sorted(p.parent.name for p in pyprojects) == [
        "agent-utilities",
        "crewai",
        "dspy",
        "langchain",
        "reqlib",
    ]


def test_returns_none_when_all_subpackages_on_target(tmp_path: Path) -> None:
    """Steady state: every subpackage is on target, no PRs to open."""
    _make_subpackage(tmp_path, "_integration_core", "0.6.0")
    _make_subpackage(tmp_path, "dspy", "0.6.0")
    _make_subpackage(tmp_path, "reqlib", "0.6.0")

    selected = _select_pass(tmp_path, "0.6.0")
    assert selected is None


def test_pass2_when_integration_core_absent(tmp_path: Path) -> None:
    """If ``_integration_core`` is not present, pass 2 fires for the rest."""
    _make_subpackage(tmp_path, "dspy", "0.5.0")

    selected = _select_pass(tmp_path, "0.6.0")
    assert selected is not None
    pass_name, pyprojects = selected
    assert pass_name == "subpackages"
    assert [p.parent.name for p in pyprojects] == ["dspy"]


def test_discover_subpackages_skips_root_and_dotdirs(tmp_path: Path) -> None:
    """Discovery skips the repo-root pyproject.toml, dotdirs, and ``cookiecutter/``."""
    # Repo-root pyproject (signpost, not a subpackage).
    write(tmp_path / "pyproject.toml", "[tool.mellea-contribs]\n")
    # Dotdir with a pyproject inside (e.g. .venv) — must be skipped.
    write(tmp_path / ".venv" / "pyproject.toml", "[project]\n")
    # cookiecutter/ has its own pyproject template but is not a subpackage.
    write(tmp_path / "cookiecutter" / "pyproject.toml", "[project]\n")
    _make_subpackage(tmp_path, "_integration_core", "0.6.0")
    _make_subpackage(tmp_path, "dspy", "0.6.0")

    discovered = [p.name for p in _discover_subpackages(tmp_path)]
    assert sorted(discovered) == ["_integration_core", "dspy"]


def test_every_discovered_subpackage_has_a_pyproject(tmp_path: Path) -> None:
    """Sanity: every result of discovery resolves to an on-disk pyproject.toml.

    Defends against the silent-skip class of bug Paul caught on the
    original hand-coded TIERS list, where renames left some subpackages
    in no tier.
    """
    _make_subpackage(tmp_path, "_integration_core", "0.6.0")
    _make_subpackage(tmp_path, "dspy", "0.6.0")
    _make_subpackage(tmp_path, "reqlib", "0.6.0")

    for sub in _discover_subpackages(tmp_path):
        assert (sub / "pyproject.toml").is_file(), sub
