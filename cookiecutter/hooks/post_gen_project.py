"""Post-generation hook: create the inner mirror chain + run uv lock."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

name = "{{ cookiecutter.name }}"
core_path = "{{ cookiecutter.core_path }}"

cwd = Path.cwd()  # cookiecutter runs hooks in the generated dir

# Create the chain of directories from core_path under
# mellea_contribs/<name>/, e.g. core_path="stdlib.sampling_algos" ->
# mellea_contribs/<name>/stdlib/sampling_algos/.
parts = core_path.split(".")
chain = cwd / "mellea_contribs" / name
for part in parts:
    chain = chain / part
    chain.mkdir(parents=True, exist_ok=True)
    init_py = chain / "__init__.py"
    if not init_py.exists():
        init_py.write_text("")

# Create the stub module at the chain's leaf, named after the subpackage.
stub_path = chain / f"{name}.py"
stub_path.write_text(
    f'"""Stub module for {name}.\n\n'
    f"Replace this with your subpackage's implementation."
    f'"""\n\n\n'
    f"def hello() -> str:\n"
    f'    return "Hello from {name}!"\n'
)

print(f"Generated stub at {stub_path.relative_to(cwd)}")

# Run uv lock (best effort — don't fail the hook if uv isn't available).
try:
    result = subprocess.run(
        ["uv", "lock"], cwd=cwd, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"warning: uv lock failed: {result.stderr}", file=sys.stderr)
    else:
        print("Generated uv.lock")
except FileNotFoundError:
    print(
        "warning: uv not found; skip uv.lock generation. Run `uv lock` manually before committing.",
        file=sys.stderr,
    )
except subprocess.TimeoutExpired:
    print("warning: uv lock timed out", file=sys.stderr)
