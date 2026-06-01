"""Tests for update_contribs_versions.py.

The script walks every pyproject.toml in a contribs-repo checkout and
bumps two things in-place: project.version and the mellea>= dependency
constraint. It is invoked by sync-contribs-version.yml after every
published mellea release.

Design intent (informs these tests):
- Idempotent. Running twice with the same target produces no edits.
- Touches `>=` and bare-name forms only. Leaves `==` exact pins alone
  (subpackages that pin exact versions own their own bumps).
- Skips .venv/, dist/, build/, .git/ when walking.
- Adds a `version = "X.Y.Z"` line to a `[project]` table that has a
  name but no version (the root meta-package case after Phase 1 F2).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from update_contribs_versions import update_repo


def write(p: Path, content: str) -> None:
    """Write `content` to `p`, dedented and with leading blank line stripped."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip())


def test_bumps_version_and_mellea_constraint(tmp_path: Path) -> None:
    """Happy path: a typical contribs subpackage gets version + mellea>= bumped."""
    write(
        tmp_path / "dspy" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-dspy"
        version = "0.5.0"
        dependencies = [
            "mellea>=0.5.0",
            "mellea-contribs-integration-core",
            "dspy>=3.1",
        ]
        """,
    )

    changed = update_repo(tmp_path, "0.6.0")

    assert (tmp_path / "dspy" / "pyproject.toml") in changed
    out = (tmp_path / "dspy" / "pyproject.toml").read_text()
    assert 'version = "0.6.0"' in out
    assert '"mellea>=0.6.0"' in out
    # Cross-deps must NOT be touched.
    assert '"mellea-contribs-integration-core"' in out
    assert '"dspy>=3.1"' in out


def test_preserves_mellea_extras(tmp_path: Path) -> None:
    """`mellea[hf]>=X` keeps its extras when bumped."""
    write(
        tmp_path / "x" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-x"
        version = "0.5.0"
        dependencies = ["mellea[hf]>=0.5.0"]
        """,
    )

    update_repo(tmp_path, "0.6.0")
    out = (tmp_path / "x" / "pyproject.toml").read_text()
    assert '"mellea[hf]>=0.6.0"' in out


def test_leaves_exact_pins_alone(tmp_path: Path) -> None:
    """`==` pins (used by reqlib_package and tools_package) are NOT touched.

    Subpackages pinning to a specific mellea version own their own bumps.
    The cookiecutter template (F3) generates `>=` so this only matters
    during the legacy migration window.
    """
    write(
        tmp_path / "reqlib" / "pyproject.toml",
        """
        [project]
        name = "mellea-reqlib"
        version = "0.5.0"
        dependencies = [
          "mellea[litellm]==0.3.2",
        ]
        """,
    )

    changed = update_repo(tmp_path, "0.6.0")
    out = (tmp_path / "reqlib" / "pyproject.toml").read_text()

    # The version line gets bumped (it's the project's own version, not a constraint).
    assert 'version = "0.6.0"' in out
    # The == constraint stays.
    assert '"mellea[litellm]==0.3.2"' in out
    # File is still in the changed list because the project.version bumped.
    assert (tmp_path / "reqlib" / "pyproject.toml") in changed


def test_idempotent(tmp_path: Path) -> None:
    """Running twice with the same target is a no-op the second time.

    GHA workflow uses this to skip opening empty PRs when contribs is
    already on the target version.
    """
    write(
        tmp_path / "x" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-x"
        version = "0.6.0"
        dependencies = ["mellea>=0.6.0"]
        """,
    )

    first = update_repo(tmp_path, "0.6.0")
    second = update_repo(tmp_path, "0.6.0")
    assert first == []
    assert second == []


def test_skips_hidden_and_build_dirs(tmp_path: Path) -> None:
    """`.venv/`, `dist/`, `build/`, `.git/` are not walked."""
    for sub in [".venv", "dist", "build", ".git"]:
        write(
            tmp_path / sub / "pyproject.toml",
            """
            [project]
            name = "ignored"
            version = "0.0.0"
            """,
        )
    write(
        tmp_path / "real" / "pyproject.toml",
        """
        [project]
        name = "real"
        version = "0.5.0"
        """,
    )

    changed = update_repo(tmp_path, "0.6.0")
    assert changed == [tmp_path / "real" / "pyproject.toml"]


def test_handles_root_pyproject_with_no_version(tmp_path: Path) -> None:
    """Root meta-package after Phase 1 F2 has [project] with name but no version.

    The script must add a `version = "X.Y.Z"` line below `name`,
    not skip the file.
    """
    write(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs"
        dependencies = []
        """,
    )

    changed = update_repo(tmp_path, "0.6.0")
    out = (tmp_path / "pyproject.toml").read_text()
    assert (tmp_path / "pyproject.toml") in changed
    assert 'version = "0.6.0"' in out
