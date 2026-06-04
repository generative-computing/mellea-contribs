"""Tests for validate_package_contract.py."""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path

# Ensure the script under test is importable when pytest is invoked from the
# repo root (the script lives next to this test file, not on sys.path).
sys.path.insert(0, str(Path(__file__).parent))

from validate_package_contract import (  # noqa: E402
    Violation,
    validate_repo,
)


def write(p: Path, content: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip())


def setup_minimal_subpackage(
    repo: Path, name: str, core_path: str = "stdlib.sampling_algos"
) -> None:
    """Create a minimally-valid subpackage with the nested mellea_contribs layout."""
    sub = repo / name
    sub.mkdir(parents=True)
    write(
        sub / "pyproject.toml",
        f"""
        [project]
        name = "mellea-contribs-{name}"
        version = "0.6.0"
        dependencies = ["mellea>=0.6.0"]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["mellea_contribs"]
    """,
    )
    write(sub / "OWNERS", "@someone\n")
    write(sub / "README.md", f"# {name}\n")
    (sub / "tests").mkdir()
    write(sub / "tests" / "test_smoke.py", "")
    (sub / "examples").mkdir()
    # Namespace package: mellea_contribs/<name>/ is a regular package.
    inner_name = name.replace("-", "_")
    inner = sub / "mellea_contribs" / inner_name
    inner.mkdir(parents=True)
    write(inner / "__init__.py", "")
    # Core mirror dirs.
    for d in ["stdlib", "backends", "formatters", "helpers", "core"]:
        (inner / d).mkdir()
        write(inner / d / "__init__.py", "")
    # Build out the core_path chain under the namespace dir.
    parts = core_path.split(".")
    p = inner
    for part in parts:
        p = p / part
        if not p.exists():
            p.mkdir()
            write(p / "__init__.py", "")


def setup_core_paths(repo: Path) -> None:
    """Create a minimal cookiecutter/core_paths.json the validator can read."""
    cc = repo / "cookiecutter"
    cc.mkdir()
    (cc / "core_paths.json").write_text(
        json.dumps(
            {
                "snapshot_date": "2026-05-29",
                "mellea_version": "0.6.0",
                "paths": [
                    "stdlib",
                    "stdlib.sampling_algos",
                    "stdlib.frameworks",
                    "stdlib.frameworks.dspy",
                    "stdlib.requirements",
                    "stdlib.tools",
                    "stdlib.components",
                    "stdlib.reqlib",
                    "stdlib.reqlib.legal",
                    "backends",
                    "formatters",
                    "helpers",
                    "core",
                ],
            }
        )
    )


def setup_grandfather(repo: Path, paths: list[str] | None = None) -> None:
    if paths is None:
        paths = []
    gh = repo / ".github" / "scripts"
    gh.mkdir(parents=True)
    (gh / "grandfather_legacy.json").write_text(json.dumps({"legacy_paths": paths}))


def test_valid_subpackage_passes(tmp_path: Path) -> None:
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    violations = validate_repo(tmp_path)
    assert violations == []


def test_missing_owners_fails(tmp_path: Path) -> None:
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    (tmp_path / "demo" / "OWNERS").unlink()
    violations = validate_repo(tmp_path)
    assert any("OWNERS" in v.message for v in violations)


def test_missing_tests_fails(tmp_path: Path) -> None:
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    shutil.rmtree(tmp_path / "demo" / "tests")
    violations = validate_repo(tmp_path)
    assert any("tests" in v.message for v in violations)


def test_invalid_core_dir_fails(tmp_path: Path) -> None:
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    # Add a bogus dir under the namespace package; it doesn't match any core_path.
    (tmp_path / "demo" / "mellea_contribs" / "demo" / "stdlib" / "totally_bogus").mkdir()
    violations = validate_repo(tmp_path)
    assert any("totally_bogus" in v.message for v in violations)


def test_unexpected_top_level_dir_fails(tmp_path: Path) -> None:
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    # Add a bogus dir at the subpackage root (not a meta dir, not a known dir).
    (tmp_path / "demo" / "totally_bogus").mkdir()
    violations = validate_repo(tmp_path)
    assert any("totally_bogus" in v.message for v in violations)


def test_missing_packages_field_fails(tmp_path: Path) -> None:
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    # Replace pyproject.toml with one missing the wheel packages field.
    write(
        tmp_path / "demo" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-demo"
        version = "0.6.0"
        dependencies = []
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"
    """,
    )
    violations = validate_repo(tmp_path)
    assert any("packages" in v.message for v in violations)


def test_missing_namespace_init_fails(tmp_path: Path) -> None:
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    # Remove the namespace package's __init__.py.
    (tmp_path / "demo" / "mellea_contribs" / "demo" / "__init__.py").unlink()
    violations = validate_repo(tmp_path)
    assert any("__init__.py" in v.message for v in violations)


def test_bare_mellea_dependency_fails(tmp_path: Path) -> None:
    """Bare `mellea` (no version constraint) is rejected — receiver needs an explicit floor."""
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    pp = tmp_path / "demo" / "pyproject.toml"
    pp.write_text(pp.read_text().replace('"mellea>=0.6.0"', '"mellea"'))
    violations = validate_repo(tmp_path)
    assert any("must declare a version constraint" in v.message for v in violations)


def test_mellea_git_ref_fails(tmp_path: Path) -> None:
    """`mellea @ git+...` is rejected — replace with explicit version before merging."""
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    pp = tmp_path / "demo" / "pyproject.toml"
    pp.write_text(
        pp.read_text().replace(
            '"mellea>=0.6.0"', '"mellea @ git+https://github.com/example/mellea.git"'
        )
    )
    violations = validate_repo(tmp_path)
    assert any("must not be a git/url ref" in v.message for v in violations)


def test_mellea_extras_with_explicit_constraint_passes(tmp_path: Path) -> None:
    """`mellea[litellm]==0.5.0` (extras + explicit constraint) is accepted."""
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    pp = tmp_path / "demo" / "pyproject.toml"
    pp.write_text(pp.read_text().replace('"mellea>=0.6.0"', '"mellea[litellm]==0.5.0"'))
    violations = validate_repo(tmp_path)
    assert violations == []


def test_sibling_distribution_names_not_flagged(tmp_path: Path) -> None:
    """`mellea-contribs-X` and `mellea-tools` are sibling dists, not the upstream package.

    They start with `mellea` but should NOT trigger the explicit-constraint
    rule — they're contribs subpackages depending on each other, not the
    upstream mellea package.
    """
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    pp = tmp_path / "demo" / "pyproject.toml"
    # Add a sibling-dependency dep alongside the explicit `mellea>=` line.
    pp.write_text(
        pp.read_text().replace(
            '"mellea>=0.6.0"',
            '"mellea>=0.6.0",\n    "mellea-contribs-integration-core",\n    "mellea-tools"',
        )
    )
    violations = validate_repo(tmp_path)
    assert violations == [], (
        "sibling distribution names should not trigger the mellea-constraint rule; "
        f"got: {[str(v) for v in violations]}"
    )


def test_grandfathered_legacy_dir_skipped(tmp_path: Path) -> None:
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path, paths=["mellea_contribs/legacy_thing"])
    # Set up a legacy dir that would otherwise fail the gate.
    legacy = tmp_path / "mellea_contribs" / "legacy_thing"
    legacy.mkdir(parents=True)
    (legacy / "totally_bogus").mkdir()
    # Also set up a valid new-flat subpackage so the test exercises both branches.
    setup_minimal_subpackage(tmp_path, "demo")
    violations = validate_repo(tmp_path)
    assert violations == []


def test_duplicate_distribution_name_fails(tmp_path: Path) -> None:
    setup_core_paths(tmp_path)
    setup_grandfather(tmp_path)
    setup_minimal_subpackage(tmp_path, "demo")
    setup_minimal_subpackage(tmp_path, "demo2")
    # Force demo2's pyproject to declare the same distribution name as demo.
    pp = tmp_path / "demo2" / "pyproject.toml"
    pp.write_text(pp.read_text().replace('"mellea-contribs-demo2"', '"mellea-contribs-demo"'))
    violations = validate_repo(tmp_path)
    assert any("duplicate" in v.message.lower() for v in violations)


def test_violation_str_includes_name_and_message() -> None:
    v = Violation("demo", "missing required file: OWNERS")
    assert "demo" in str(v)
    assert "missing required file: OWNERS" in str(v)


def test_real_repo_passes() -> None:
    """The current contribs repo must pass the gate (legacy paths grandfathered)."""
    repo = Path(__file__).resolve().parents[2]
    # Sanity: we are pointing at the contribs repo root.
    assert (repo / "cookiecutter" / "core_paths.json").exists()
    violations = validate_repo(repo)
    assert violations == [], "unexpected violations:\n" + "\n".join(str(v) for v in violations)
