"""Tests for update_contribs_versions.py.

The script walks every pyproject.toml in a contribs-repo checkout and
bumps `[project] version` in-place. It is invoked by
receive-mellea-release.yml after every published mellea release.

Design intent (informs these tests):
- Bumps `[project] version` only. Does NOT rewrite `mellea>=` constraints.
  The `mellea>=` floor is owned by each subpackage (sliding-window model);
  it only raises when CI proves something below it breaks.
- Idempotent. Running twice with the same target produces no edits.
- Errors loudly when a `mellea` dep line is bare (`"mellea"`,
  `"mellea[extras]"`) or a git ref (`"mellea @ git+..."`). Acceptable
  forms: `mellea>=X.Y.Z`, `mellea==X.Y.Z`, with or without extras.
- Skips .venv/, dist/, build/, .git/ when walking.
- Adds a `version = "X.Y.Z"` line to a `[project]` table that has a
  name but no version (the root meta-package case).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from update_contribs_versions import UnacceptableMelleaLine, update_repo


def write(p: Path, content: str) -> None:
    """Write `content` to `p`, dedented and with leading blank line stripped."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip())


def test_bumps_version_only_not_mellea_constraint(tmp_path: Path) -> None:
    """A typical contribs subpackage gets version bumped; mellea>= stays put."""
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
    # mellea>= floor is owned by the subpackage; receiver does NOT raise it.
    assert '"mellea>=0.5.0"' in out
    # Cross-deps must NOT be touched.
    assert '"mellea-contribs-integration-core"' in out
    assert '"dspy>=3.1"' in out


def test_mellea_keyword_is_not_mistaken_for_a_dependency(tmp_path: Path) -> None:
    """`keywords = ["mellea", ...]` must not trip the bare-mellea validator.

    Regression: the receiver aborted on _integration_core because the mellea
    check scanned the whole file and matched the keyword, not just deps.
    """
    write(
        tmp_path / "_integration_core" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-integration-core"
        version = "0.1.0"
        keywords = ["mellea", "contribs", "integration-core"]
        dependencies = [
            "mellea>=0.3.2",
        ]
        """,
    )

    changed = update_repo(tmp_path, "0.6.0")

    out = (tmp_path / "_integration_core" / "pyproject.toml").read_text()
    assert (tmp_path / "_integration_core" / "pyproject.toml") in changed
    assert 'version = "0.6.0"' in out
    assert '"mellea>=0.3.2"' in out  # dep floor untouched
    assert '"mellea"' in out  # keyword untouched


def test_mellea_keyword_on_its_own_line_is_not_a_dependency(tmp_path: Path) -> None:
    """A multi-line `keywords` array with `"mellea"` alone on a line is fine too."""
    write(
        tmp_path / "x" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-x"
        version = "0.1.0"
        keywords = [
            "mellea",
            "contribs",
        ]
        dependencies = ["mellea>=0.3.2"]
        """,
    )

    changed = update_repo(tmp_path, "0.6.0")
    assert (tmp_path / "x" / "pyproject.toml") in changed


def test_bare_mellea_after_an_extras_dep_is_still_caught(tmp_path: Path) -> None:
    """An extras `]` earlier in the array must not truncate the scanned body.

    Regression: the array walker stopped at the first `]`, so a dep with
    extras (whose `[extra]` contains `]`) dropped every dep below it — a bare
    `mellea` after it escaped the safety check.
    """
    write(
        tmp_path / "x" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-x"
        version = "0.1.0"
        dependencies = [
            "some-pkg[extra]>=1.0",
            "mellea",
        ]
        """,
    )

    with pytest.raises(UnacceptableMelleaLine, match="bare `mellea`"):
        update_repo(tmp_path, "0.6.0")


def test_preserves_mellea_extras_constraint(tmp_path: Path) -> None:
    """`mellea[hf]>=X` is preserved verbatim across a version bump."""
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
    assert 'version = "0.6.0"' in out
    assert '"mellea[hf]>=0.5.0"' in out


def test_leaves_exact_pins_alone(tmp_path: Path) -> None:
    """`==` pins are preserved; only project.version bumps."""
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

    # Project's own version bumps.
    assert 'version = "0.6.0"' in out
    # The == constraint stays.
    assert '"mellea[litellm]==0.3.2"' in out
    # File appears in changed list because project.version bumped.
    assert (tmp_path / "reqlib" / "pyproject.toml") in changed


def test_rejects_bare_mellea_dependency(tmp_path: Path) -> None:
    """Bare `mellea` (no version constraint) is rejected — receiver can't reason."""
    write(
        tmp_path / "x" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-x"
        version = "0.5.0"
        dependencies = ["mellea"]
        """,
    )

    with pytest.raises(UnacceptableMelleaLine, match="bare `mellea`"):
        update_repo(tmp_path, "0.6.0")


def test_rejects_bare_mellea_with_extras(tmp_path: Path) -> None:
    """`mellea[extras]` without an operator is rejected."""
    write(
        tmp_path / "x" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-x"
        version = "0.5.0"
        dependencies = ["mellea[hf]"]
        """,
    )

    with pytest.raises(UnacceptableMelleaLine, match="bare `mellea`"):
        update_repo(tmp_path, "0.6.0")


def test_rejects_git_ref(tmp_path: Path) -> None:
    """`mellea @ git+...` is rejected — replace before merging."""
    write(
        tmp_path / "x" / "pyproject.toml",
        """
        [project]
        name = "mellea-contribs-x"
        version = "0.5.0"
        dependencies = ["mellea @ git+https://github.com/example/mellea.git"]
        """,
    )

    with pytest.raises(UnacceptableMelleaLine, match="git/url ref"):
        update_repo(tmp_path, "0.6.0")


def test_idempotent(tmp_path: Path) -> None:
    """Running twice with the same target is a no-op the second time."""
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
    """Root meta-package has [project] with name but no version.

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
