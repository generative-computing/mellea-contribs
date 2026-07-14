"""Open one bump PR per subpackage in dependency order.

Used by ``.github/workflows/receive-mellea-release.yml`` after every
mellea release. Replaces the "one big PR" model with one PR per
subpackage so owners can merge their bump independently — slow movers
don't block fast movers, and per-package CI signals compatibility.

Two passes (a pass's PRs open concurrently; passes serialize):

- Pass 1: ``_integration_core``. Frameworks depend on it, so it must
  land before they bump.
- Pass 2: every other repo-root subpackage with a ``pyproject.toml``.
  No subpackage outside ``_integration_core`` is depended on by
  another subpackage today, so they all open concurrently.

The script runs once per workflow invocation and opens only the PRs
for the *current* pass — i.e., pass 1 if ``_integration_core`` still
needs a bump, otherwise pass 2. The workflow re-runs (via the bot
itself observing pass-1's PR being merged) to advance to pass 2.

Subpackages are discovered by walking the repo root for any directory
that contains a ``pyproject.toml``. The repo-root ``pyproject.toml``
itself is not a subpackage and is skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from update_contribs_versions import (
    UnacceptableMelleaLine,
    _set_project_version,
    _validate_mellea_lines,
    update_pyproject,
)

INTEGRATION_CORE = "_integration_core"

# Repo-root entries that have a pyproject.toml but are not subpackages.
_NOT_A_SUBPACKAGE = {"cookiecutter"}


def _discover_subpackages(repo: Path) -> list[Path]:
    """Return repo-root subpackages (directories with a pyproject.toml)."""
    out: list[Path] = []
    for child in sorted(repo.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name in _NOT_A_SUBPACKAGE:
            continue
        if (child / "pyproject.toml").exists():
            out.append(child)
    return out


def _needs_bump(pyproject: Path, target: str) -> bool:
    """True iff ``update_pyproject`` would rewrite the [project] version line."""
    text = pyproject.read_text()
    return _set_project_version(text, target) != text


def _select_pass(repo: Path, target: str) -> tuple[str, list[Path]] | None:
    """Return (pass_name, list_of_subpackage_pyprojects_that_need_bump).

    Pass 1 fires alone whenever ``_integration_core`` lags. Pass 2 fires
    once pass 1 is on target. Returns None when every subpackage is on
    target.
    """
    subs = _discover_subpackages(repo)

    core = next((s for s in subs if s.name == INTEGRATION_CORE), None)
    others = [s for s in subs if s.name != INTEGRATION_CORE]

    if core is not None and _needs_bump(core / "pyproject.toml", target):
        return "integration-core", [core / "pyproject.toml"]

    pass2 = [
        s / "pyproject.toml"
        for s in others
        if _needs_bump(s / "pyproject.toml", target)
    ]
    if pass2:
        return "subpackages", pass2

    return None


def _validate_all(pyprojects: list[Path]) -> None:
    """Raise UnacceptableMelleaLine if any pyproject has a mellea dep we can't bump.

    Validates every candidate up front so a malformed dep aborts the run
    *before* any branches are pushed or PRs opened — no partial passes.
    """
    for pp in pyprojects:
        _validate_mellea_lines(pp.read_text(), pp)


def _load_owner_handles(sub_dir: Path) -> list[str]:
    """Return the subpackage's OWNERS as bare handles (no leading ``@``).

    OWNERS lists one ``@handle`` per line; a missing or empty file yields an
    empty list. Used to request the subpackage's owners as PR reviewers.
    """
    owners_file = sub_dir / "OWNERS"
    if not owners_file.exists():
        return []
    return [
        line.strip().lstrip("@")
        for line in owners_file.read_text().splitlines()
        if line.strip().startswith("@")
    ]


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

    # Everything from here edits the tree (pyproject bump, uv.lock, branch).
    # Wrap it all so that if any step raises — `uv lock` resolution/network
    # failure, push rejected, `gh pr create` failure — the `finally` still
    # resets the checkout to a clean `main`. A raised exception ends the whole
    # run (main()'s loop has no handler), so the reset's job is to leave a
    # clean checkout behind rather than a stranded, half-bumped tree.
    try:
        # Bump pyproject.toml. Pre-flight validation in main() guarantees this
        # won't raise UnacceptableMelleaLine; let any other error propagate.
        update_pyproject(subpackage, target)

        # Refresh uv.lock.
        subprocess.run(
            ["uv", "lock", "--upgrade-package", "mellea"], cwd=sub_dir, check=True
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

        # Ensure the version-carrying label exists before `gh pr create --label`
        # uses it — `gh pr create` fails (leaving an orphaned branch) if the
        # label is absent, and it will not auto-create it. `--force` is
        # idempotent: creates the label or no-ops if it already exists.
        subprocess.run(
            [
                "gh", "label", "create",
                f"sync-mellea-version:{target}",
                "--color", "BFD4F2",
                "--description", "Target mellea version for the receiver's pass-2 re-trigger",
                "--force",
            ],
            cwd=repo,
            check=True,
        )

        # Open PR with a label that carries the target version. The receiver
        # workflow's pull_request: closed re-trigger reads the version from
        # this label rather than reparsing the branch name, which would
        # truncate pre-release suffixes (e.g. ``0.6.0rc1`` → ``0.6.0``).
        body = (
            f"Automated bump of `{name}`'s `[project] version` to `{target}` "
            f"after the matching mellea release.\n\n"
            f"`mellea>=` constraint untouched — owners control that floor.\n\n"
            f"Merging this PR is independent of any other bump PR in the same "
            f"pass; please verify CI passes."
        )
        subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--title",
                f"chore({name}): bump version to v{target}",
                "--body",
                body,
                "--head",
                branch,
                "--base",
                "main",
                "--label",
                f"sync-mellea-version:{target}",
            ],
            cwd=repo,
            check=True,
        )

        # Request the subpackage's OWNERS as reviewers, as a separate,
        # non-fatal step. `gh pr create --reviewer` aborts the whole create
        # if any handle is invalid/not a collaborator (which would strand the
        # branch), so we add reviewers after the PR exists and don't fail the
        # run if the request is rejected — a bad OWNERS entry just means no
        # reviewer got tagged, not a broken release.
        reviewers = _load_owner_handles(sub_dir)
        if reviewers:
            r = subprocess.run(
                ["gh", "pr", "edit", branch, "--add-reviewer", ",".join(reviewers)],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(
                    f"  [{name}] could not request reviewers {reviewers}: "
                    f"{r.stderr.strip()}"
                )
    finally:
        # Always leave a clean checkout on main, even if a step above raised.
        subprocess.run(["git", "checkout", "main"], cwd=repo, check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=repo, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path, help="contribs repo root")
    parser.add_argument("version", help="target version (matches mellea release)")
    args = parser.parse_args()

    if not args.repo_root.is_dir():
        print(f"error: {args.repo_root} is not a directory", file=sys.stderr)
        return 1

    selected = _select_pass(args.repo_root, args.version)
    if selected is None:
        print(f"All subpackages already at v{args.version}; nothing to bump.")
        return 0

    pass_name, pyprojects = selected
    print(f"Active pass: {pass_name} ({len(pyprojects)} subpackage(s) need bumping)")
    for pp in pyprojects:
        print(f"  - {pp.parent.name}")

    # Validate every candidate before opening any PRs so a malformed
    # mellea dep aborts cleanly without leaving a partial pass behind.
    try:
        _validate_all(pyprojects)
    except UnacceptableMelleaLine as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for pp in pyprojects:
        _open_pr(args.repo_root, pp, args.version)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
