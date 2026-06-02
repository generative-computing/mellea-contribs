"""Open one bump PR per subpackage in dependency-tier order.

Used by ``.github/workflows/receive-mellea-release.yml`` after every
mellea release. Replaces the "one big PR" model with one PR per
subpackage so owners can merge their bump independently — slow movers
don't block fast movers, and per-package CI signals compatibility.

Tiers (subpackages within a tier are opened concurrently; tiers serialize):

- Tier 1: ``_integration_core`` (consumed by frameworks).
- Tier 2: framework subpackages — ``dspy``, ``langchain``, ``crewai``,
  ``tools``. Opened once Tier 1's PR has merged.
- Tier 3: leaf subpackages with no contribs-internal dependencies —
  ``legal-reqs``, ``python-imports``, ``grounding-context``. Opened once
  Tier 2 has merged.

The script runs once per workflow invocation and opens only the PRs for
the *current* tier — i.e., the lowest tier with subpackages that still
need a bump and whose dependencies are already on the target version.
The workflow re-runs (via the bot itself observing tier-N PRs being
merged) to advance to the next tier.

Subpackages whose ``pyproject.toml`` doesn't exist or doesn't need a
bump (already at target version) are skipped silently.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from update_contribs_versions import UnacceptableMelleaLine, update_pyproject

# Tier definitions — see module docstring. The tier name is used as a
# stable label only; the receiver looks up subpackages by directory name.
TIERS: list[tuple[str, list[str]]] = [
    ("integration-core", ["_integration_core", "mellea-integration-core"]),
    ("frameworks", ["dspy", "langchain", "crewai", "tools"]),
    ("leaves", ["legal-reqs", "python-imports", "grounding-context"]),
]


def _subpackage_needs_bump(pyproject: Path, target: str) -> bool:
    """True iff editing pyproject would actually change [project] version."""
    if not pyproject.exists():
        return False
    text = pyproject.read_text()
    # Cheap signal: target version string already present in a [project]
    # block? Not perfect but avoids reparsing.
    needle = f'version = "{target}"'
    return needle not in text


def _detect_tier(repo: Path, target: str) -> tuple[str, list[Path]] | None:
    """Return (tier_name, list_of_subpackage_pyprojects_that_need_bump).

    Returns the lowest tier that has at least one subpackage needing a bump.
    If every tier is already on target, returns None.
    """
    for tier_name, candidates in TIERS:
        needs_bump: list[Path] = []
        for sub in candidates:
            pyproject = repo / sub / "pyproject.toml"
            if _subpackage_needs_bump(pyproject, target):
                needs_bump.append(pyproject)
        if needs_bump:
            return tier_name, needs_bump
    return None


def _open_pr(repo: Path, subpackage: Path, target: str) -> None:
    """Branch, edit, lock, commit, push, and gh-pr-create for one subpackage.

    Idempotent at the branch level: if the branch already exists upstream,
    skip cleanly (the previous run already opened this PR).
    """
    sub_dir = subpackage.parent
    name = sub_dir.name
    branch = f"sync-mellea-{target}-{name}"

    # Skip if branch already exists upstream (PR already opened).
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", "origin", f"refs/heads/{branch}"],
        cwd=repo,
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"  [{name}] branch {branch} already exists upstream — skipping.")
        return

    # Bump pyproject.toml.
    try:
        update_pyproject(subpackage, target)
    except UnacceptableMelleaLine as exc:
        print(f"  [{name}] error: {exc}", file=sys.stderr)
        sys.exit(2)

    # Refresh uv.lock.
    subprocess.run(
        ["uv", "lock", "--upgrade-package", "mellea"],
        cwd=sub_dir,
        check=True,
    )

    # Branch, commit, push.
    subprocess.run(["git", "checkout", "-b", branch], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"chore({name}): bump version to v{target}"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "push", "origin", branch], cwd=repo, check=True)

    # Open PR.
    body = (
        f"Automated bump of `{name}`'s `[project] version` to `{target}` "
        f"after the matching mellea release.\n\n"
        f"`mellea>=` constraint untouched — owners control that floor.\n\n"
        f"Merging this PR is independent of any other tier-{name} subpackage's "
        f"bump PR; please verify CI passes."
    )
    subprocess.run(
        [
            "gh", "pr", "create",
            "--title", f"chore({name}): bump version to v{target}",
            "--body", body,
            "--head", branch,
            "--base", "main",
        ],
        cwd=repo,
        check=True,
    )

    # Reset to main for the next iteration.
    subprocess.run(["git", "checkout", "main"], cwd=repo, check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=repo, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path, help="contribs repo root")
    parser.add_argument("version", help="target version (matches mellea release)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report which tier and subpackages would be bumped, but don't edit",
    )
    args = parser.parse_args()

    if not args.repo_root.is_dir():
        print(f"error: {args.repo_root} is not a directory", file=sys.stderr)
        return 1

    detected = _detect_tier(args.repo_root, args.version)
    if detected is None:
        print(f"All subpackages already at v{args.version}; nothing to bump.")
        return 0

    tier_name, pyprojects = detected
    print(f"Active tier: {tier_name} ({len(pyprojects)} subpackage(s) need bumping)")
    for pp in pyprojects:
        print(f"  - {pp.parent.name}")

    if args.dry_run:
        # Emit JSON for the workflow to consume.
        print(json.dumps({"tier": tier_name, "subpackages": [pp.parent.name for pp in pyprojects]}))
        return 0

    for pp in pyprojects:
        _open_pr(args.repo_root, pp, args.version)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
