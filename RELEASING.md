# Releasing mellea-contribs

This document describes the coordinated-release model used by `mellea-contribs`. A single root tag `v<X.Y.Z>` builds and publishes every subpackage plus the meta-package as assets of one GitHub Release. There is no PyPI publishing and no per-subpackage tag pipeline.

> **Note:** the `release.yml` workflow that drives this lands in Phase 3 of the restructure. This document is the contract that pipeline implements; some references below are forward-looking until P3 merges.

## Repository Prerequisites

The receiver (`.github/workflows/receive-mellea-release.yml`) opens bump PRs using a token minted from the **"mellea auto release"** GitHub App, not the default `GITHUB_TOKEN`. This is because the `generative-computing` org locks `GITHUB_TOKEN` to read-only and forbids Actions from creating pull requests org-wide (and the per-repo override isn't available), so `gh pr create` under the default token fails and leaves orphaned `sync-mellea-*` branches.

For the receiver to work, all of the following must hold (they already do today):

- The **"mellea auto release"** App is installed on `generative-computing/mellea-contribs` with **Contents: write** and **Pull requests: write**.
- Its credentials are available to the repo's Actions as `CI_APP_ID` (variable) and `CI_PRIVATE_KEY` (secret) — the same pair `cd.yml` uses.

No repo or org Actions *setting* needs changing; the App token supplies the write access directly.

## Overview

`mellea-contribs` releases track upstream `mellea` exactly. Every published `mellea` `vX.Y.Z` triggers a contribs release at the same `vX.Y.Z`. Off-cycle subpackage patches are not released independently — they ride the next coordinated `vX.Y.(Z+1)`.

The release ships:

- One wheel + sdist per subpackage (`_integration_core`, `dspy`, `crewai`, `langchain`, `agent-utilities`, `reqlib`).
- One meta-package wheel + sdist whose `[project.optional-dependencies]` extras pin each subpackage by direct-URL to assets of the same release.
- A version-free copy of the meta-package wheel named `mellea_contribs-py3-none-any.whl` so the GitHub `releases/latest/download/` redirector resolves it without callers naming a version.

## Versioning

Versions are not chosen by contribs maintainers. Each contribs release adopts the upstream `mellea` version that triggered it; pre-releases on the `mellea` side propagate the same suffix here. The `mellea>=` floor inside each subpackage's own `dependencies` is owned by that subpackage and changes only when the owner bumps it deliberately.

## Trigger paths

**Automatic (the normal path).** A release event on the upstream `mellea` repo is dispatched into this repo. The receiver (`receive-mellea-release.yml`) opens bump PRs in two passes:

1. A single PR for `_integration_core` (every framework subpackage depends on it, so it must land first).
2. Once that PR is merged, one PR per remaining subpackage opens together: `dspy`, `crewai`, `langchain`, `agent-utilities`, `reqlib`, plus the meta-package.

Each bump PR carries a `sync-mellea-version:<X.Y.Z>` label so the receiver can recover the target version on the second pass without re-parsing the branch name. Merging the final PRs runs `release.yml`, which builds and publishes the coordinated release.

**Manual escape hatch.** A maintainer pushes a `v<X.Y.Z>` tag from `main`. Use this only when the dispatch path is unavailable (e.g. recovering from a broken release).

## Operational Notes

### Pass-2 is triggered automatically (via the App token)

The pass-2 PRs are triggered by the merge event of the pass-1 PR. GitHub has a safety rule that prevents events caused by the default `GITHUB_TOKEN` from triggering further workflow runs — so if the receiver ran under `GITHUB_TOKEN`, merging the pass-1 PR (by anyone) would not open pass 2, and the release would silently stop.

The receiver avoids this by minting a token from the **"mellea auto release"** GitHub App (`.github/workflows/receive-mellea-release.yml`, via `CI_APP_ID` / `CI_PRIVATE_KEY`). An App token is exempt from the safety rule, so merging the pass-1 `_integration_core` PR — by a human, a bot, or auto-merge — correctly re-triggers the receiver and opens pass 2. No special merge handling is required.

> **If you ever revert the receiver to the default `GITHUB_TOKEN`**, this breaks: pass 2 stops firing on merge and a human would have to click merge on the `_integration_core` PR to advance it. Keep the App token.

## What the pipeline does

1. **Validate structure.** Runs the same gate as PR CI; fails fast if any subpackage's `pyproject.toml` is malformed.
2. **Build `_integration_core` first.** Framework subpackages depend on it, so it must be built and assembled before its dependents.
3. **Lint, mypy, pytest per subpackage.** Each subpackage runs its full CI suite at the tagged version. A failing subpackage is recorded as a straggler (see below) and excluded from the release.
4. **Build wheels with rewritten dependencies.** Source `pyproject.toml` files declare `[tool.uv.sources] mellea-contribs-integration-core = { path = "../_integration_core" }` for local-dev convenience. The pipeline rewrites this dependency to the GitHub Release URL of `mellea-contribs-integration-core` at the same `vX.Y.Z` before invoking `uv build`. The `[tool.uv.sources]` table does not appear in the published wheel's metadata — that is a property of how hatch builds wheels (uv sources are uv-only). Contributors do not need to think about this.
5. **Build the meta-package wheel.** Each `[project.optional-dependencies]` extra is rewritten at build time to point at the just-built subpackage's release-asset URL, pinned to `vX.Y.Z`.
6. **Upload assets.** Wheels and sdists are attached to the GitHub Release; the meta-package wheel is also uploaded under the version-free name `mellea_contribs-py3-none-any.whl`.
7. **Generate release notes.** Notes call out any stragglers explicitly (see below).

## Verifying a release

On the [Releases page](https://github.com/generative-computing/mellea-contribs/releases), confirm the `vX.Y.Z` release contains:

- A wheel + sdist per non-straggler subpackage (`mellea_contribs_<name>-X.Y.Z-py3-none-any.whl`, `.tar.gz`).
- The versioned meta-package wheel + sdist.
- The version-free `mellea_contribs-py3-none-any.whl` copy.
- Release notes listing the included subpackages and any stragglers excluded from this release.

## Installing from a release

The recommended install uses the `latest` redirector:

```bash
pip install "mellea-contribs[<extra>] @ https://github.com/generative-computing/mellea-contribs/releases/latest/download/mellea_contribs-py3-none-any.whl"
```

The meta-package wheel is downloaded "latest", but its extras pin concrete subpackage versions internally — the install is reproducible.

For a fully explicit pin, use the versioned URL:

```bash
pip install "mellea-contribs[<extra>] @ https://github.com/generative-computing/mellea-contribs/releases/download/vX.Y.Z/mellea_contribs-X.Y.Z-py3-none-any.whl"
```

Available extras: `dspy`, `crewai`, `langchain`, `agent-utilities`, `reqlib`, plus `all`. `_integration_core` is always installed transitively.

## Stragglers and skips

If a subpackage fails to build, lint, or test in the release pipeline, it is **excluded from `[all]` extras and from the GitHub Release asset set** for that version. The CHANGELOG entry for the release names the affected subpackages explicitly.

The straggler ships in the next coordinated release once its owner lands the fix. Distro-style: the train leaves on schedule; missed packages catch the next one. Users who need the missing subpackage can either pin to the prior release or install the subpackage from source until the next cut.

## CHANGELOG conventions

Each release entry in `CHANGELOG.md` records:

- The `mellea` version this release tracks.
- A short summary of changes per subpackage that shipped.
- An explicit "Skipped" subsection naming any straggler subpackages excluded from this release and linking to their tracking issue.
- Any breaking changes called out under a `### Breaking` heading per subpackage.

The CHANGELOG is updated in the sync PR before merge — not after the tag is pushed — so the diff is reviewable.

## Smoke + auto-issue bot

A daily smoke job (`.github/workflows/smoke-against-mellea-main.yml`) runs each opted-in subpackage's tests against `mellea @ main` at 07:17 UTC. The matrix is read from `.github/smoke-matrix.json`; while that list is empty the smoke job is skipped. Each subpackage opts in by appending its path to the `subpackages` array as part of its migration PR.

When a subpackage's smoke job goes red, the auto-issue bot (`.github/scripts/auto_issue_bot.py`) tracks consecutive failures and opens a tracking issue once the second consecutive red lands. Issues are labelled `contribs-broken` and assigned to the package's OWNERS.

A second daily workflow (`.github/workflows/auto-issue-archival.yml`) runs at 08:00 UTC and applies the archival timeline based on how long the `contribs-broken` label has continuously been present on each tracking issue:

- **Day 7** — a reminder comment is posted to escalate the failure.
- **Day 14** — the bot posts a notice that the subpackage will be called out as broken in the next contribs release notes. When you cut a release while a tracking issue is at or past day 14, mention the affected subpackages explicitly in the release notes.
- **Day 21** — the bot applies the `archived` label and posts a final comment. Contribs maintainers move the subpackage to the archived layout in a follow-up PR.

Recovery (a green smoke run) clears the `contribs-broken` label, posts a "smoke green again on `<date>`, fixed in `<sha>`" comment, resets the archival clock, and leaves the issue open for a human to close. Removing the `contribs-broken` label by hand has the same effect — the timeline is driven by the label, not the issue's open state.

The bot's persistent state lives at `.github/bot-state/<package>.json` (one JSON file per subpackage) on a bot-managed branch and is updated by each invocation. Per-file state eliminates the write race that a single shared file would create when smoke legs run concurrently.

To run the bot locally against the in-memory fake (no GitHub API calls):

```bash
uv run python .github/scripts/auto_issue_bot.py \
    --action record-failure \
    --package dspy \
    --run-url https://example/run/1 \
    --fake
```

## Local dry-run

Maintainers can build the full asset set locally without publishing:

```bash
# Build _integration_core first
cd _integration_core && uv build && cd ..

# Then each framework subpackage
for pkg in dspy crewai langchain agent-utilities reqlib; do
    (cd "$pkg" && uv build)
done

# Build the meta-package last
uv build
```

The wheels land in each subpackage's `dist/`. To verify that the published wheel's metadata does not leak `[tool.uv.sources]`, inspect a built wheel:

```bash
unzip -p dspy/dist/mellea_contribs_dspy-*.whl '*.dist-info/METADATA' | grep -i 'uv\.sources\|integration-core'
```

Only the rewritten direct-URL dependency on `mellea-contribs-integration-core` should appear.
