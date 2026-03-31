# Quick Start: Releasing a Package

## TL;DR

```bash
# 1. Update version in pyproject.toml
vim mellea_contribs/dspy_backend/pyproject.toml

# 2. Commit and push
git add mellea_contribs/dspy_backend/pyproject.toml
git commit -m "chore: bump mellea-dspy to v0.2.0"
git push origin main

# 3. Create and push tag
git tag mellea-dspy/v0.2.0
git push origin mellea-dspy/v0.2.0

# 4. Done! Check GitHub Actions and Releases
```

## Tag Format

```
<package-name>/v<version>
```

## Package Names

| Package Name | Directory |
|--------------|-----------|
| `mellea-crewai` | `mellea_contribs/crewai_backend` |
| `mellea-dspy` | `mellea_contribs/dspy_backend` |
| `mellea-langchain` | `mellea_contribs/langchain_backend` |
| `mellea-reqlib` | `mellea_contribs/reqlib_package` |
| `mellea-tools` | `mellea_contribs/tools_package` |

## Examples

```bash
# Release mellea-dspy v0.2.0
git tag mellea-dspy/v0.2.0
git push origin mellea-dspy/v0.2.0

# Release mellea-crewai v0.1.1
git tag mellea-crewai/v0.1.1
git push origin mellea-crewai/v0.1.1
```

## What Happens Automatically

1. ✅ Package is built
2. ✅ Tests are run
3. ✅ GitHub release is created
4. ✅ Wheel (.whl) and source (.tar.gz) are uploaded
5. ✅ Release notes are generated
6. ✅ SHA256 checksums are included

## Installation from Release

```bash
pip install https://github.com/generative-computing/mellea-contribs/releases/download/mellea-dspy/v0.2.0/mellea-dspy-0.2.0-py3-none-any.whl
```

## Full Documentation

See [RELEASING.md](RELEASING.md) for complete details.