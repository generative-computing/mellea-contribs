# Adding a new `mellea-contribs` subpackage

This walks through landing a working idea (code in your fork or a local
script) as a real contribs subpackage. The friction is intentional but small:
cookiecutter scaffolds the layout, CI enforces the contract, and the daily smoke
keeps you honest against `mellea@main`.

## 1. Should this be a contribs subpackage?

Belongs **here** if it's:

- Opinionated or framework-specific (a particular sampling algorithm, a
  third-party framework integration, a niche reqlib)
- Not on the mellea-core roadmap (you've checked, or it's clearly out of scope)
- Useful to a known audience but doesn't have to satisfy every Mellea user
- Something you're willing to maintain across mellea releases — but accept that
  if you stop, the package can be evicted

Belongs in **mellea core** instead if it's stable, broadly applicable, and would
materially benefit from core review.

Belongs in a **standalone repo** if it's experimental, you don't want CI churn
tied to mellea releases, or you can't commit to bumping it when mellea moves.

## 2. Scaffold with cookiecutter

From the contribs repo root:

```bash
uv run --with cookiecutter cookiecutter ./cookiecutter \
  --no-input name=my_thing core_path=stdlib.frameworks
```

- `name`: the Python package name (snake_case). Used to derive the on-disk
  directory (`my-thing/`) and the distribution name (`mellea-contribs-my-thing`).
- `core_path`: a dotted path that mirrors a real location in mellea core
  (e.g. `stdlib.frameworks`, `backends.aloras`, `helpers`). Cookiecutter
  validates this against `cookiecutter/core_paths.json` — invalid values are
  rejected. Pick the path closest to where your code conceptually lives.

Other prompts (`version`, `mellea_version`, `owner_name`, `owner_email`,
`short_description`) can be passed on the command line, or filled in
interactively without `--no-input`.

## 3. What got generated

```
my-thing/
├── pyproject.toml          # distribution name, deps, ci flags, hatch config
├── OWNERS                  # one @github-username per line
├── README.md               # describe what the subpackage does
├── examples/               # runnable examples
├── tests/                  # pytest tests; mirrors the core_path layout
└── mellea_contribs/
    └── my_thing/
        ├── __init__.py     # public API surface (re-exports go here)
        ├── stdlib/
        │   └── frameworks/ # ← matches your `core_path`
        ├── backends/__init__.py
        ├── core/__init__.py
        ├── formatters/__init__.py
        └── helpers/__init__.py
```

The full mirror (`stdlib/`, `backends/`, etc.) is scaffolded unconditionally so
you don't hit "I added a `helpers/` dir and imports broke" later. Empty mirror
directories are fine; `validate-structure` doesn't require them to be populated.

## 4. Add your code

Put source files under `mellea_contribs/my_thing/<core_path>/`. The on-disk
layout matches the wheel layout 1:1 — no remap, no `src/`, no `setup.py`
trickery. Imports look like:

```python
from mellea_contribs.my_thing.stdlib.frameworks.my_module import MyClass
```

Re-export your public API from `mellea_contribs/my_thing/__init__.py` so
callers can do `from mellea_contribs.my_thing import MyClass` at the top
level. See [`_integration_core/mellea_contribs/_integration_core/__init__.py`](../../_integration_core/mellea_contribs/_integration_core/__init__.py)
for the pattern.

## 5. Declare dependencies

Edit `my-thing/pyproject.toml`. Two rules `validate-structure` enforces:

- Every `mellea` dependency line MUST declare an explicit `>=X.Y.Z` or
  `==X.Y.Z` constraint. Bare `mellea`, bare `mellea[extras]`, and
  `mellea @ git+...` refs are rejected.
- `[tool.hatch.build.targets.wheel].packages` must be `["mellea_contribs"]`
  (cookiecutter sets this; don't change it).

If your subpackage depends on another contribs subpackage (e.g. on
`_integration_core`), declare it in BOTH `dependencies` and
`[tool.uv.sources]` so local checkouts resolve from the path while published
wheels resolve from the GitHub Release URL. See
[`dspy/pyproject.toml`](../../dspy/pyproject.toml):

```toml
dependencies = [
    "dspy>=3.1.3",
    "mellea>=0.3.2",
    "mellea-contribs-integration-core",
]

[tool.uv.sources]
mellea-contribs-integration-core = { path = "../_integration_core" }
```

## 6. OWNERS

One `@github-username` per line. Plain text — no YAML, no role distinction.
Multiple owners are encouraged so a single owner going dark doesn't strand the
package. Auto-issues both @-mention and assign every owner.

```text
@your-handle
@your-co-owner
```

## 7. Run tests locally

```bash
cd my-thing
uv lock
uv sync
uv run pytest
```

Each subpackage has its own `uv.lock` and its own `.venv` after `uv sync`.
The cookiecutter ships a smoke test that imports the package; add real tests
under `tests/`. Place runnable examples under `examples/`.

## 8. What CI runs on your PR

When you open a PR:

- **Discovery** detects which subpackage(s) you touched (path-diff against the
  base ref).
- **`validate-structure`** asserts the directory contract: required files,
  namespace package shape, valid `core_path` mirror dirs, explicit `mellea`
  constraint, distribution-name uniqueness across the repo.
- **`package-ci`** runs lint, mypy, and `pytest` for each touched subpackage.
  CI flags (`skip_ollama`, `timeout_minutes`, `python_versions`) come from
  your `[tool.mellea-contribs.ci]` table — no need to edit the workflow.
- **Stacked PRs** (when `base_ref` isn't `main`) automatically promote to the
  all-packages matrix so cumulative drift across the stack gets caught.

Docs-only PRs run nothing. Cookiecutter-template-only PRs run a template smoke
plus `validate-structure`.

## 9. What gets shipped at release time

When the contribs repo cuts its next coordinated release, your subpackage's
wheel is built and attached as an asset of one GitHub Release at a single
version (see [`RELEASING.md`](../../RELEASING.md) for the full pipeline). The
meta-package wheel declares your subpackage as an extra, so users install it
via the GitHub Releases `latest` redirector:

```bash
pip install "mellea-contribs[my-thing] @ https://github.com/generative-computing/mellea-contribs/releases/latest/download/mellea_contribs-py3-none-any.whl"
```

In the same PR that adds your subpackage, append its directory name to the
`subpackages` array in [`.github/smoke-matrix.json`](../../.github/smoke-matrix.json)
so the daily smoke job picks it up.

## 10. Eviction signals

Two labels can land on your subpackage's tracking issue. Both share the same
day-7 / day-14 / day-21 archival timeline; only the trigger and remediation
differ.

- **`[contribs-broken]`** — the daily smoke against `mellea@main` has been red
  for two-plus consecutive days. Mellea changed something and your code
  doesn't run on `main` anymore. Fix: patch the subpackage. The bot tracks
  recovery automatically and posts a "smoke green again" comment when smoke
  passes; the issue stays open until OWNERS close it manually.

- **`[contribs-stale]`** — your `mellea>=` floor in `pyproject.toml` has fallen
  below the contribs-wide minimum (`[tool.mellea-contribs] minimum_mellea_version`
  in the root `pyproject.toml`). This is a maintenance signal: nobody has
  bumped the floor in a while. Fix: raise the floor and run tests against the
  newer mellea. The bot auto-closes the issue once the floor is at or above
  the minimum.

Shared timeline:

| Day | What happens |
|---|---|
| 7 | Reminder comment posted — escalate |
| 14 | Bot notes the subpackage will be called out as at-risk in the next contribs release notes |
| 21 | `archived` label applied; subpackage moved to the archived layout in a follow-up PR; dropped from `[all]` extras at the next release |

A green smoke run (or a floor bump) clears the relevant label and resets the
clock. The timeline is driven by how long the label has been continuously
present, not by issue open-state — keep the issue open as a historical record
without re-arming the eviction clock.
