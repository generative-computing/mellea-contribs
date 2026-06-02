"""Update [project] version across every pyproject.toml in a contribs checkout.

Used by .github/workflows/receive-mellea-release.yml after every published
mellea release. Bumps only ``[project] version`` so contribs's release
version tracks mellea exactly. Does NOT rewrite the ``mellea>=`` constraint
line — that floor is owned by each subpackage and only raised when CI
proves something below it breaks (sliding-window model).

Idempotent: running twice with the same version produces no edits. Edits
are line-based regex substitutions, not full TOML round-trip, so
formatting and comments are preserved.

Errors loudly when a subpackage declares ``mellea`` without an explicit
version constraint (bare ``mellea``, ``mellea[extras]``, or ``mellea @
git+...``). The receiver cannot reason about those forms; the right fix
is to declare an explicit ``>=`` or ``==`` constraint in the source.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EXCLUDED = {".venv", "dist", "build", ".git", "__pycache__", "node_modules"}

# Matches any "mellea..." dep line. Used to inspect each line and decide
# whether it's an acceptable form (>=, ==) or a rejected form (bare, git+,
# or unknown operator).
_MELLEA_LINE_RE = re.compile(r'"mellea(\[[^\]]+\])?(?P<spec>[^"]*)"')


class UnacceptableMelleaLine(ValueError):
    """Raised when a pyproject.toml has a mellea dep line we can't reason about."""


def _validate_mellea_lines(text: str, path: Path) -> None:
    """Raise UnacceptableMelleaLine if any `mellea` dep is bare or git-ref'd.

    Acceptable forms: `mellea>=X.Y.Z`, `mellea==X.Y.Z`,
    `mellea[extras]>=X.Y.Z`, `mellea[extras]==X.Y.Z`.

    Rejected: bare `mellea`, `mellea[extras]` without operator, `mellea @ git+...`.
    """
    for match in _MELLEA_LINE_RE.finditer(text):
        spec = match.group("spec").strip()
        if not spec:
            raise UnacceptableMelleaLine(
                f"{path}: bare `mellea` dependency without version constraint. "
                "Use `mellea>=X.Y.Z` or `mellea==X.Y.Z`."
            )
        if spec.startswith("@"):
            raise UnacceptableMelleaLine(
                f"{path}: mellea declared as a git/url ref ({spec!r}). "
                "Replace with `mellea>=X.Y.Z` or `mellea==X.Y.Z` before merging."
            )
        if not (spec.startswith(">=") or spec.startswith("==")):
            raise UnacceptableMelleaLine(
                f"{path}: mellea constraint uses an operator we won't auto-bump ({spec!r}). "
                "Use `mellea>=X.Y.Z` or `mellea==X.Y.Z`."
            )


def update_pyproject(path: Path, target_version: str) -> bool:
    """Edit a single pyproject.toml in-place. Return True if the file changed.

    Validates mellea dep lines first; raises UnacceptableMelleaLine if any
    line uses a form the receiver cannot reason about.
    """
    text = path.read_text()
    _validate_mellea_lines(text, path)
    original = text

    text = _set_project_version(text, target_version)

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


def update_repo(root: Path, target_version: str) -> list[Path]:
    """Walk the repo and update every relevant pyproject.toml.

    Returns a list of changed paths in deterministic (sorted) order.
    Raises UnacceptableMelleaLine on the first malformed mellea dep line.
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

    try:
        changed = update_repo(args.repo_root, args.version)
    except UnacceptableMelleaLine as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not changed:
        print("No pyproject.toml files needed updating.")
        return 0

    print(f"Updated {len(changed)} file(s):")
    for path in changed:
        print(f"  - {path.relative_to(args.repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
