"""Update version + mellea>= constraints across every pyproject.toml in a contribs checkout.

Used by .github/workflows/sync-contribs-version.yml after every published mellea release.

Idempotent: running twice with the same version produces no edits. Leaves `==`
exact pins alone (subpackages that pin exact mellea versions own their own bumps).

Edits are line-based regex substitutions, not full TOML round-trip, so formatting
and comments are preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EXCLUDED = {".venv", "dist", "build", ".git", "__pycache__", "node_modules"}

# Matches a "mellea" or "mellea[extras]" dep line that uses `>=` or `>`. Leaves
# `==`, `<`, `<=`, `~=`, `!=`, and bare name (no operator) alone — `==` because
# subpackages opting into exact pins own their own bumps; the others because
# they're unusual enough that we'd rather surface them via a failing PR than
# silently rewrite.
_DEP_MELLEA_RE = re.compile(r'(\s*)"mellea(\[[^\]]+\])?>=[^"]*"')


def update_pyproject(path: Path, target_version: str) -> bool:
    """Edit a single pyproject.toml in-place. Return True if the file changed."""
    text = path.read_text()
    original = text

    text = _set_project_version(text, target_version)
    text = _rewrite_mellea_deps(text, target_version)

    if text == original:
        return False
    path.write_text(text)
    return True


def _set_project_version(text: str, target: str) -> str:
    """Set `version = "X.Y.Z"` under [project].

    If [project] has a name but no version, insert one after the name line.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_project = False
    project_has_version = False
    project_name_idx = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
        elif in_project:
            if stripped.startswith("name") and "=" in stripped:
                project_name_idx = i
            if stripped.startswith("version") and "=" in stripped:
                project_has_version = True
                line = re.sub(r'(\s*version\s*=\s*)"[^"]*"', rf'\1"{target}"', line)
        out.append(line)

    text = "".join(out)

    # If [project] has a name but no version, insert one after the name line.
    if project_name_idx >= 0 and not project_has_version:
        lines = text.splitlines(keepends=True)
        name_line = lines[project_name_idx]
        indent = name_line[: len(name_line) - len(name_line.lstrip())]
        insertion = f'{indent}version = "{target}"\n'
        lines.insert(project_name_idx + 1, insertion)
        text = "".join(lines)

    return text


def _rewrite_mellea_deps(text: str, target: str) -> str:
    """Rewrite `"mellea>=X"` and `"mellea[extras]>=X"` to use the target version.

    Leaves `==` pins, other operators, and non-mellea deps untouched.
    """

    def _sub(m: re.Match[str]) -> str:
        leading_ws = m.group(1)
        extras = m.group(2) or ""
        return f'{leading_ws}"mellea{extras}>={target}"'

    return _DEP_MELLEA_RE.sub(_sub, text)


def update_repo(root: Path, target_version: str) -> list[Path]:
    """Walk the repo and update every relevant pyproject.toml.

    Returns a list of changed paths in deterministic (sorted) order.
    """
    changed: list[Path] = []
    for path in sorted(root.rglob("pyproject.toml")):
        rel_parts = path.relative_to(root).parts[:-1]
        if any(part in EXCLUDED or part.startswith(".") for part in rel_parts):
            continue
        if update_pyproject(path, target_version):
            changed.append(path)
    return changed


def main() -> int:
    """CLI entry point: walk a repo, bump versions, report changed files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("version")
    args = parser.parse_args()

    if not args.repo_root.is_dir():
        print(f"error: {args.repo_root} is not a directory", file=sys.stderr)
        return 1

    changed = update_repo(args.repo_root, args.version)
    if not changed:
        print("No pyproject.toml files needed updating.")
        return 0

    print(f"Updated {len(changed)} file(s):")
    for path in changed:
        print(f"  - {path.relative_to(args.repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
