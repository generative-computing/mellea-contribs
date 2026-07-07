"""Discover which subpackages a PR touches; output a CI matrix.

Implements the PR-scoping rules and the stacked-PR override (when
``base_ref`` is not ``main``, the matrix is promoted to all packages so
the stacked branch sees the full effect of its parent's changes).

Used by ``.github/workflows/ci.yml``'s ``discover`` job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

DOCS_PREFIXES = ("docs/", "RFCs/")
DOCS_FILE_SUFFIX = (".md",)
COOKIECUTTER_PREFIX = "cookiecutter/"
ROOT_TRIGGER_FILES = {"pyproject.toml", "uv.lock"}
WORKFLOWS_PREFIX = ".github/workflows/"
SHARED_SCRIPTS_PREFIX = ".github/scripts/"
LEGACY_PREFIX = "mellea_contribs/"


@dataclass
class DiscoverInputs:
    """Inputs to :func:`discover`."""

    changed_files: list[str]
    base_ref: str
    repo_root: Path


@dataclass
class DiscoverResult:
    """Result of :func:`discover`."""

    matrix: list[str] = field(default_factory=list)
    run_template_smoke: bool = False
    reason: str = ""


def _is_top_level_doc(f: str) -> bool:
    # A repo-root markdown file (e.g. README.md), not one nested in a subpackage.
    return "/" not in f and f.endswith(DOCS_FILE_SUFFIX)


def _is_docs_only(files: list[str]) -> bool:
    # Docs prefixes and repo-root `.md` are docs-only. A `.md` *inside* a
    # subpackage (e.g. dspy/README.md) is not — it must run that package's CI,
    # so it deliberately fails both checks here.
    return len(files) > 0 and all(
        f.startswith(DOCS_PREFIXES) or _is_top_level_doc(f) for f in files
    )


def _has_cookiecutter_change(files: list[str]) -> bool:
    return any(f.startswith(COOKIECUTTER_PREFIX) for f in files)


def _has_root_trigger(files: list[str]) -> bool:
    return any(
        f in ROOT_TRIGGER_FILES
        or f.startswith(WORKFLOWS_PREFIX)
        or f.startswith(SHARED_SCRIPTS_PREFIX)
        for f in files
    )


def _changed_subpackages(files: list[str], all_subpackages: list[str]) -> list[str]:
    """Among the new-flat subpackages, which ones did this PR touch?"""
    touched: set[str] = set()
    sub_set = set(all_subpackages)
    for f in files:
        first_segment = f.split("/", 1)[0] if "/" in f else f
        if first_segment in sub_set:
            touched.add(first_segment)
    return sorted(touched)


def discover(
    inputs: DiscoverInputs, all_subpackages: list[str] | None = None
) -> DiscoverResult:
    """Compute which subpackages need CI for the given changed-files set."""
    if all_subpackages is None:
        all_subpackages = []
    files = inputs.changed_files

    # Stacked-PR override: when base_ref isn't main, run all packages so the
    # stacked branch sees the full effect of its parent's changes.
    if inputs.base_ref and inputs.base_ref != "main":
        return DiscoverResult(
            matrix=list(all_subpackages),
            run_template_smoke=False,
            reason="stacked-PR (base_ref != main -> all-packages)",
        )

    if not files:
        return DiscoverResult(matrix=[], run_template_smoke=False, reason="no-changes")

    # Docs-only short-circuit.
    if _is_docs_only(files):
        return DiscoverResult(matrix=[], run_template_smoke=False, reason="docs-only")

    template_smoke = _has_cookiecutter_change(files)

    # Root-level / workflow / shared-script trigger: run everything.
    if _has_root_trigger(files):
        return DiscoverResult(
            matrix=list(all_subpackages),
            run_template_smoke=template_smoke,
            reason="root-or-workflow-change (all-packages)",
        )

    # Subpackage-scoped changes.
    touched = _changed_subpackages(files, all_subpackages)

    # If only legacy changes hit, the new ci.yml runs nothing (legacy-ci.yml
    # picks up these PRs).
    legacy_only = (
        all(
            f.startswith(LEGACY_PREFIX) or f.startswith(COOKIECUTTER_PREFIX)
            for f in files
        )
        and not touched
    )
    if legacy_only and not template_smoke:
        return DiscoverResult(
            matrix=[], run_template_smoke=False, reason="legacy-only-no-new-ci"
        )

    if touched:
        reason = "subpackage-scoped"
    elif template_smoke:
        reason = "cookiecutter-only"
    else:
        reason = "no-trigger"

    return DiscoverResult(
        matrix=touched, run_template_smoke=template_smoke, reason=reason
    )


def _load_subpackages_from_repo(repo_root: Path) -> list[str]:
    """Find new-flat subpackages: top-level dirs containing a pyproject.toml."""
    subs: list[str] = []
    exclude = {"cookiecutter", "mellea_contribs", ".github", ".venv", "RFCs", "docs"}
    for child in sorted(repo_root.iterdir()):
        if not child.is_dir() or child.name in exclude or child.name.startswith("."):
            continue
        if (child / "pyproject.toml").exists():
            subs.append(child.name)
    return subs


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--changed-files", required=True, help="Newline-separated list of changed files"
    )
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    files = [f.strip() for f in args.changed_files.splitlines() if f.strip()]
    inputs = DiscoverInputs(
        changed_files=files, base_ref=args.base_ref, repo_root=args.repo_root
    )
    all_subs = _load_subpackages_from_repo(args.repo_root)
    result = discover(inputs, all_subpackages=all_subs)

    print(
        f"Discover: matrix={result.matrix}, "
        f"template_smoke={result.run_template_smoke}, reason={result.reason}",
        file=sys.stderr,
    )

    output = {
        "matrix": result.matrix,
        "run_template_smoke": result.run_template_smoke,
        "reason": result.reason,
    }
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"matrix={json.dumps(result.matrix)}\n")
            f.write(
                "run_template_smoke="
                f"{'true' if result.run_template_smoke else 'false'}\n"
            )
            f.write(f"reason={result.reason}\n")
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
