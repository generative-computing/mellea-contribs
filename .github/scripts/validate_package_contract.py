"""Validate the structural contract for mellea-contribs subpackages.

Used by ``.github/workflows/validate-structure.yml`` on every PR. Walks every
subpackage at the repo root and asserts:

- Required files: ``pyproject.toml``, ``OWNERS``, ``README.md``
- Required dirs: ``tests/``, ``examples/``, ``mellea_contribs/<name>/``
- ``OWNERS`` is non-empty
- ``pyproject.toml`` has ``[tool.hatch.build.targets.wheel] packages =
  ["mellea_contribs"]`` so the wheel ships the namespace package
- The subpackage's namespace dir ``mellea_contribs/<name>/`` exists and
  contains an ``__init__.py``
- Every top-level directory under ``mellea_contribs/<name>/`` is one of the
  known core mirror dirs (stdlib, backends, formatters, helpers, core)
- Every dir under a mirror dir corresponds to a known dotted path in
  ``cookiecutter/core_paths.json``
- No two subpackages declare the same distribution name

Legacy old-flat subpackages listed in ``.github/scripts/grandfather_legacy.json``
are skipped. The list shrinks as each legacy subpackage is migrated.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FILES = ["pyproject.toml", "OWNERS", "README.md"]
REQUIRED_DIRS = ["tests", "examples"]
KNOWN_MIRROR_DIRS = {"stdlib", "backends", "formatters", "helpers", "core"}
NAMESPACE_PKG = "mellea_contribs"
REQUIRED_HATCH_PACKAGES = [NAMESPACE_PKG]

META_DIRS = {
    "tests",
    "examples",
    "dist",
    "build",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "docs",
    "config",
    NAMESPACE_PKG,
}


@dataclass(frozen=True)
class Violation:
    """A single contract violation discovered during validation."""

    subpackage: str
    message: str

    def __str__(self) -> str:
        return f"  - [{self.subpackage}] {self.message}"


def _load_core_paths(repo: Path) -> set[str]:
    snapshot = repo / "cookiecutter" / "core_paths.json"
    if not snapshot.exists():
        raise FileNotFoundError(f"core_paths.json not found at {snapshot}")
    with snapshot.open() as f:
        return set(json.load(f)["paths"])


def _load_grandfather(repo: Path) -> set[Path]:
    gh = repo / ".github" / "scripts" / "grandfather_legacy.json"
    if not gh.exists():
        return set()
    with gh.open() as f:
        return {(repo / p).resolve() for p in json.load(f).get("legacy_paths", [])}


def _is_subpackage_dir(d: Path) -> bool:
    """A directory at repo root is a subpackage if it has a pyproject.toml."""
    return d.is_dir() and (d / "pyproject.toml").exists()


def _bad_mellea_constraint(dep: str) -> str | None:
    """Return a human-readable error if dep is a `mellea` line lacking a constraint.

    Acceptable: ``mellea>=X.Y.Z``, ``mellea==X.Y.Z``, ``mellea[extras]>=...``,
    ``mellea[extras]==...``. Any other form involving the bare ``mellea``
    package name (no operator, git/url ref, or unsupported operator) is
    rejected so the receiver workflow can reason about the line.

    Sibling distribution names like ``mellea-contribs-integration-core`` and
    ``mellea-tools`` start with ``mellea`` but are NOT the upstream mellea
    package — they're skipped here.

    Returns None if dep is not a mellea line or is acceptable.
    """
    s = dep.strip()
    # Strip optional extras: "mellea[hf]>=0.5.0" -> "mellea>=0.5.0".
    if s.startswith("mellea["):
        close = s.find("]")
        if close == -1:
            return None  # Malformed; let toml/pep508 parser complain elsewhere.
        spec = s[close + 1 :].strip()
        head = "mellea[...]"
    elif s.startswith("mellea"):
        # Disambiguate from sibling distribution names (mellea-X, mellea_X).
        # The bare `mellea` package is followed by an operator, whitespace,
        # `@` (git ref), or end-of-string — never by a hyphen, underscore,
        # letter, or digit.
        rest = s[len("mellea") :]
        if rest and rest[0] in "-_" or (rest and rest[0].isalnum()):
            return None  # Sibling distribution name, not the upstream mellea package.
        spec = rest.strip()
        head = "mellea"
    else:
        return None  # Not a mellea line.

    # Reject bare names ("mellea" or "mellea[extras]") and git/url refs.
    if not spec:
        return f"`{head}` dependency must declare a version constraint (e.g. `>=X.Y.Z` or `==X.Y.Z`)"
    if spec.startswith("@"):
        return f"`{head}` dependency must not be a git/url ref; use `>=X.Y.Z` or `==X.Y.Z`"
    if not (spec.startswith(">=") or spec.startswith("==")):
        return f"`{head}` dependency must use `>=X.Y.Z` or `==X.Y.Z`"
    return None


def _check_mirror_paths(
    name: str,
    namespace_dir: Path,
    mirror_dir: Path,
    valid_core_paths: set[str],
    violations: list[Violation],
) -> None:
    """Walk mirror_dir; every dir must match a known dotted core_path.

    Paths are computed relative to the namespace dir
    (``mellea_contribs/<name>/``) so the resulting dotted form lines up with
    entries in ``cookiecutter/core_paths.json``.
    """
    for path in mirror_dir.rglob("*"):
        if not path.is_dir() or path.name == "__pycache__":
            continue
        rel = path.relative_to(namespace_dir)
        dotted = ".".join(rel.parts)
        if dotted not in valid_core_paths:
            violations.append(
                Violation(name, f"directory {rel} does not match any valid core_path")
            )


def _validate_subpackage(
    name: str, sub: Path, valid_core_paths: set[str]
) -> list[Violation]:
    violations: list[Violation] = []

    # 1. Required files and dirs at the subpackage root.
    for f in REQUIRED_FILES:
        if not (sub / f).exists():
            violations.append(Violation(name, f"missing required file: {f}"))
    for d in REQUIRED_DIRS:
        if not (sub / d).is_dir():
            violations.append(Violation(name, f"missing required directory: {d}/"))

    # 2. OWNERS not empty.
    owners = sub / "OWNERS"
    if owners.exists() and not owners.read_text().strip():
        violations.append(Violation(name, "OWNERS file is empty"))

    # 3. pyproject.toml [tool.hatch.build.targets.wheel] packages = ["mellea_contribs"].
    # 3b. Every `mellea` dep line must use an explicit `>=` or `==` constraint.
    pp = sub / "pyproject.toml"
    if pp.exists():
        try:
            data = tomllib.loads(pp.read_text())
            wheel = (
                data.get("tool", {})
                .get("hatch", {})
                .get("build", {})
                .get("targets", {})
                .get("wheel", {})
            )
            packages = wheel.get("packages")
            if packages != REQUIRED_HATCH_PACKAGES:
                violations.append(
                    Violation(
                        name,
                        f"[tool.hatch.build.targets.wheel].packages must be {REQUIRED_HATCH_PACKAGES} (got {packages!r})",
                    )
                )
            for dep in data.get("project", {}).get("dependencies", []):
                msg = _bad_mellea_constraint(dep)
                if msg:
                    violations.append(Violation(name, f"{msg} (got {dep!r})"))
        except tomllib.TOMLDecodeError as exc:
            violations.append(Violation(name, f"pyproject.toml is invalid TOML: {exc}"))

    # 4. Subpackage root must contain mellea_contribs/<name>/ with __init__.py.
    namespace_dir = sub / NAMESPACE_PKG
    inner_dir = namespace_dir / name.replace("-", "_")
    if not namespace_dir.is_dir():
        violations.append(
            Violation(name, f"missing required directory: {NAMESPACE_PKG}/")
        )
    elif not inner_dir.is_dir():
        violations.append(
            Violation(
                name, f"missing required namespace directory: {NAMESPACE_PKG}/{inner_dir.name}/"
            )
        )
    elif not (inner_dir / "__init__.py").exists():
        violations.append(
            Violation(
                name,
                f"missing __init__.py at {NAMESPACE_PKG}/{inner_dir.name}/ (regular-package boundary)",
            )
        )
    else:
        # Walk the namespace dir's children: each must be a known mirror dir.
        for child in inner_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name == "__pycache__":
                continue
            if child.name not in KNOWN_MIRROR_DIRS:
                violations.append(
                    Violation(
                        name,
                        f"unexpected directory under {NAMESPACE_PKG}/{inner_dir.name}/: {child.name}/ (not a core mirror)",
                    )
                )
            else:
                _check_mirror_paths(
                    name, inner_dir, child, valid_core_paths, violations
                )

    # 5. No unexpected top-level dirs at the subpackage root.
    for child in sub.iterdir():
        if not child.is_dir():
            continue
        if child.name in META_DIRS or child.name.startswith("."):
            continue
        if child.name in REQUIRED_DIRS:
            continue
        violations.append(
            Violation(
                name,
                f"unexpected top-level directory: {child.name}/",
            )
        )

    return violations


def validate_repo(repo: Path) -> list[Violation]:
    """Validate every new-flat subpackage at the repo root.

    Args:
        repo: Path to the contribs repo root.

    Returns:
        List of violations; empty if the repo conforms.
    """
    valid_core_paths = _load_core_paths(repo)
    grandfathered = _load_grandfather(repo)
    violations: list[Violation] = []

    distribution_names: dict[str, str] = {}

    for child in sorted(repo.iterdir()):
        if not _is_subpackage_dir(child):
            continue
        # Skip cookiecutter (it has a templated subdir but is not a subpackage).
        if child.name == "cookiecutter":
            continue
        # Skip grandfathered legacy paths (or any subpackage living under one).
        resolved = child.resolve()
        if resolved in grandfathered or any(p in grandfathered for p in resolved.parents):
            continue

        name = child.name
        violations.extend(_validate_subpackage(name, child, valid_core_paths))

        # Distribution-name uniqueness.
        try:
            data = tomllib.loads((child / "pyproject.toml").read_text())
            dist = data.get("project", {}).get("name")
            if dist:
                if dist in distribution_names:
                    violations.append(
                        Violation(
                            name,
                            f"duplicate distribution name '{dist}' (also in {distribution_names[dist]})",
                        )
                    )
                else:
                    distribution_names[dist] = name
        except (tomllib.TOMLDecodeError, FileNotFoundError):
            pass  # Already reported above.

    # Also walk legacy subpackages nested under grandfathered roots so we still
    # check distribution-name uniqueness across the migration window? Not yet —
    # legacy packages are explicitly excluded until they migrate.

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the structural contract for mellea-contribs subpackages."
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Path to the contribs repo root (defaults to CWD).",
    )
    args = parser.parse_args()

    violations = validate_repo(args.repo_root)
    if not violations:
        print("validate-structure: PASS - all subpackages conform.")
        return 0

    print(f"validate-structure: FAIL ({len(violations)} violation(s)):")
    for v in violations:
        print(v)
    return 1


if __name__ == "__main__":
    sys.exit(main())
