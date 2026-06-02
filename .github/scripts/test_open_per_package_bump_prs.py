"""Tests for open_per_package_bump_prs._detect_tier.

Side-effecting helpers (`_open_pr`, `main`) are not tested here — they
shell out to git/gh and are exercised end-to-end by the workflow. This
test suite covers the tier-detection logic, which is the only piece
with non-trivial branching.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from open_per_package_bump_prs import _detect_tier


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


def test_detects_tier1_when_only_integration_core_lags(tmp_path: Path) -> None:
    """Tier 1 fires first if `_integration_core` is behind, even if other tiers also lag."""
    _make_subpackage(tmp_path, "_integration_core", "0.5.0")
    _make_subpackage(tmp_path, "dspy", "0.5.0")
    _make_subpackage(tmp_path, "legal-reqs", "0.5.0")

    detected = _detect_tier(tmp_path, "0.6.0")
    assert detected is not None
    tier_name, pyprojects = detected
    assert tier_name == "integration-core"
    assert [p.parent.name for p in pyprojects] == ["_integration_core"]


def test_detects_tier2_when_tier1_already_on_target(tmp_path: Path) -> None:
    """Once tier 1 is on target, tier 2 (frameworks) becomes the active tier."""
    _make_subpackage(tmp_path, "_integration_core", "0.6.0")
    _make_subpackage(tmp_path, "dspy", "0.5.0")
    _make_subpackage(tmp_path, "langchain", "0.5.0")
    _make_subpackage(tmp_path, "legal-reqs", "0.5.0")

    detected = _detect_tier(tmp_path, "0.6.0")
    assert detected is not None
    tier_name, pyprojects = detected
    assert tier_name == "frameworks"
    assert sorted(p.parent.name for p in pyprojects) == ["dspy", "langchain"]


def test_detects_tier3_when_tier1_and_tier2_on_target(tmp_path: Path) -> None:
    """Leaves are last."""
    _make_subpackage(tmp_path, "_integration_core", "0.6.0")
    _make_subpackage(tmp_path, "dspy", "0.6.0")
    _make_subpackage(tmp_path, "legal-reqs", "0.5.0")
    _make_subpackage(tmp_path, "python-imports", "0.5.0")

    detected = _detect_tier(tmp_path, "0.6.0")
    assert detected is not None
    tier_name, pyprojects = detected
    assert tier_name == "leaves"
    assert sorted(p.parent.name for p in pyprojects) == ["legal-reqs", "python-imports"]


def test_returns_none_when_all_subpackages_on_target(tmp_path: Path) -> None:
    """Steady state: every subpackage is on target, no PRs to open."""
    _make_subpackage(tmp_path, "_integration_core", "0.6.0")
    _make_subpackage(tmp_path, "dspy", "0.6.0")
    _make_subpackage(tmp_path, "legal-reqs", "0.6.0")

    detected = _detect_tier(tmp_path, "0.6.0")
    assert detected is None


def test_skips_subpackages_that_dont_exist(tmp_path: Path) -> None:
    """Tier candidates that aren't present on disk are silently skipped."""
    # Only `dspy` exists; other tier-2 candidates are absent.
    _make_subpackage(tmp_path, "dspy", "0.5.0")

    detected = _detect_tier(tmp_path, "0.6.0")
    assert detected is not None
    tier_name, pyprojects = detected
    assert tier_name == "frameworks"
    assert [p.parent.name for p in pyprojects] == ["dspy"]


def test_legacy_mellea_integration_core_name_recognized(tmp_path: Path) -> None:
    """During the migration window the directory is `mellea-integration-core`,
    not `_integration_core`. Tier 1 must accept either."""
    _make_subpackage(tmp_path, "mellea-integration-core", "0.5.0")

    detected = _detect_tier(tmp_path, "0.6.0")
    assert detected is not None
    tier_name, pyprojects = detected
    assert tier_name == "integration-core"
    assert [p.parent.name for p in pyprojects] == ["mellea-integration-core"]
