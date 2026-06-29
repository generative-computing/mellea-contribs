"""Pre-generation hook: validate cookiecutter inputs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

name = "{{ cookiecutter.name }}"
core_path = "{{ cookiecutter.core_path }}"
template_dir = "{{ cookiecutter._template }}"

if not NAME_RE.match(name):
    print(
        f"ERROR: name '{name}' must be snake_case: lowercase, start with a letter, only [a-z0-9_].",
        file=sys.stderr,
    )
    sys.exit(1)

# Cookiecutter sets `cookiecutter._template` to the absolute path of the
# template directory it's rendering from. The core_paths.json snapshot lives
# at the root of that directory.
snapshot_path = Path(template_dir) / "core_paths.json"

if not snapshot_path.exists():
    print(
        f"WARNING: core_paths.json not found at {snapshot_path}; skipping core_path validation.",
        file=sys.stderr,
    )
else:
    with snapshot_path.open() as f:
        snapshot = json.load(f)
    valid_paths = set(snapshot["paths"])

    if core_path not in valid_paths:
        print(
            f"ERROR: core_path '{core_path}' is not a valid mellea core path.",
            file=sys.stderr,
        )
        print(f"Valid examples: {sorted(valid_paths)[:10]}", file=sys.stderr)
        sys.exit(1)

print(f"Validated: name={name}, core_path={core_path}")
