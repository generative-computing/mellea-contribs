# Release Process for Mellea Contribs Packages

This document describes how to release individual packages from the `mellea_contribs` directory to GitHub releases.

## Overview

Each package in the `mellea_contribs` directory can be released independently by creating a Git tag with a specific format. The CI system will automatically:

1. Build the package
2. Run tests
3. Create a GitHub release
4. Upload wheel and source distribution files as release assets
5. Generate release notes with installation instructions

## Package List

The following packages can be released:

| Package Name | Directory | Description |
|--------------|-----------|-------------|
| `mellea-crewai` | `mellea_contribs/crewai_backend` | CrewAI integration for Mellea |
| `mellea-dspy` | `mellea_contribs/dspy_backend` | DSPy integration for Mellea |
| `mellea-langchain` | `mellea_contribs/langchain_backend` | LangChain integration for Mellea |
| `mellea-reqlib` | `mellea_contribs/reqlib_package` | Requirements library utilities |
| `mellea-tools` | `mellea_contribs/tools_package` | Additional tools and utilities |

## Release Steps

### 1. Prepare the Release

Before creating a release, ensure:

- [ ] All changes are committed and pushed to the main branch
- [ ] Tests pass locally
- [ ] Version number is updated in `pyproject.toml`
- [ ] CHANGELOG or release notes are prepared (optional)

### 2. Update Version

Edit the `pyproject.toml` file in the package directory and update the version:

```toml
[project]
name = "mellea-dspy"
version = "0.2.0"  # Update this version
```

Commit the version change:

```bash
git add mellea_contribs/dspy_backend/pyproject.toml
git commit -m "chore: bump mellea-dspy to v0.2.0"
git push origin main
```

### 3. Create and Push the Release Tag

The tag format is: `<package-name>/v<version>`

**Examples:**
```bash
# Release mellea-dspy version 0.2.0
git tag mellea-dspy/v0.2.0
git push origin mellea-dspy/v0.2.0

# Release mellea-crewai version 0.1.1
git tag mellea-crewai/v0.1.1
git push origin mellea-crewai/v0.1.1

# Release mellea-langchain version 1.0.0
git tag mellea-langchain/v1.0.0
git push origin mellea-langchain/v1.0.0
```

### 4. Monitor the Release

After pushing the tag:

1. Go to the [Actions tab](https://github.com/generative-computing/mellea-contribs/actions)
2. Find the "Release Package" workflow run
3. Monitor the progress through these stages:
   - **Parse Tag**: Validates tag format and extracts package info
   - **Build**: Builds the package and runs tests
   - **Create Release**: Creates GitHub release with artifacts

### 5. Verify the Release

Once the workflow completes:

1. Go to the [Releases page](https://github.com/generative-computing/mellea-contribs/releases)
2. Find your release (e.g., `mellea-dspy v0.2.0`)
3. Verify the release includes:
   - Release notes with installation instructions
   - Wheel file (`.whl`)
   - Source distribution (`.tar.gz`)
   - SHA256 checksums (`SHA256SUMS`)

## Installation from GitHub Release

Users can install packages directly from GitHub releases:

### Option 1: Download and Install Wheel

```bash
# Download the wheel file from the release page, then:
pip install mellea-dspy-0.2.0-py3-none-any.whl
```

### Option 2: Install Directly from URL

```bash
pip install https://github.com/generative-computing/mellea-contribs/releases/download/mellea-dspy/v0.2.0/mellea-dspy-0.2.0-py3-none-any.whl
```

### Option 3: Install from Git Tag

```bash
git clone https://github.com/generative-computing/mellea-contribs.git
cd mellea-contribs
git checkout mellea-dspy/v0.2.0
cd mellea_contribs/dspy_backend
pip install .
```

## Tag Format Specification

Tags must follow this exact format:

```
<package-name>/v<major>.<minor>.<patch>
```

**Valid Examples:**
- `mellea-dspy/v0.1.0`
- `mellea-crewai/v1.2.3`
- `mellea-langchain/v2.0.0-beta.1`

**Invalid Examples:**
- `dspy/v0.1.0` (wrong package name)
- `mellea-dspy-v0.1.0` (missing slash)
- `mellea-dspy/0.1.0` (missing 'v' prefix)
- `v0.1.0` (missing package name)

## Version Validation

The CI system validates that:

1. The tag version matches the version in `pyproject.toml`
2. The package directory exists
3. The `pyproject.toml` file is valid

If validation fails, the workflow will stop and provide an error message.

## Troubleshooting

### Tag Already Exists

If you need to re-release with the same version:

```bash
# Delete the tag locally and remotely
git tag -d mellea-dspy/v0.2.0
git push origin :refs/tags/mellea-dspy/v0.2.0

# Create and push the tag again
git tag mellea-dspy/v0.2.0
git push origin mellea-dspy/v0.2.0
```

### Version Mismatch Error

If you see "Version mismatch" error:

1. Check the version in `pyproject.toml`
2. Ensure it matches the tag version (without the 'v' prefix)
3. Update the version and create a new tag

### Build Failures

If the build fails:

1. Check the workflow logs in the Actions tab
2. Run tests locally: `cd mellea_contribs/<package_dir> && pytest`
3. Fix any issues and create a new tag with a patch version bump

### Release Not Created

If the workflow succeeds but no release appears:

1. Check that you have the correct permissions
2. Verify the tag was pushed to the remote repository
3. Check the workflow logs for any errors in the "Create Release" step

## Advanced: Manual Release

If you need to create a release manually:

```bash
# Navigate to package directory
cd mellea_contribs/dspy_backend

# Build the package
uv build

# The built files will be in the dist/ directory:
# - mellea-dspy-0.2.0-py3-none-any.whl
# - mellea-dspy-0.2.0.tar.gz
```

Then create a GitHub release manually and upload these files.

## CI Workflow Details

The release process uses two GitHub Actions workflows:

1. **`release-package.yml`**: Main workflow triggered by tags
   - Parses the tag to extract package name and version
   - Maps package name to directory
   - Calls the build workflow
   - Creates GitHub release with artifacts

2. **`build-package.yml`**: Reusable workflow for building packages
   - Validates version matches `pyproject.toml`
   - Detects build system (hatchling or pdm-backend)
   - Installs dependencies with UV
   - Runs tests
   - Builds wheel and source distribution
   - Uploads artifacts

## Best Practices

1. **Semantic Versioning**: Follow [semver](https://semver.org/) for version numbers
   - MAJOR: Breaking changes
   - MINOR: New features (backward compatible)
   - PATCH: Bug fixes

2. **Test Before Release**: Always run tests locally before creating a tag

3. **Update Documentation**: Update README and documentation when releasing new features

4. **Changelog**: Consider maintaining a CHANGELOG.md in each package directory

5. **Release Notes**: The CI generates basic release notes, but you can edit them after release

6. **Coordinate Releases**: If multiple packages depend on each other, release them in dependency order

## Support

For issues with the release process:

1. Check the [GitHub Actions logs](https://github.com/generative-computing/mellea-contribs/actions)
2. Review this documentation
3. Open an issue in the repository
4. Contact the maintainers

## Examples

### Example 1: Patch Release

```bash
# Fix a bug in mellea-dspy
cd mellea_contribs/dspy_backend
# Make changes...
# Update version from 0.1.0 to 0.1.1 in pyproject.toml

git add .
git commit -m "fix: resolve issue with async operations"
git push origin main

git tag mellea-dspy/v0.1.1
git push origin mellea-dspy/v0.1.1
```

### Example 2: Minor Release with New Features

```bash
# Add new features to mellea-langchain
cd mellea_contribs/langchain_backend
# Implement new features...
# Update version from 0.1.0 to 0.2.0 in pyproject.toml

git add .
git commit -m "feat: add support for streaming responses"
git push origin main

git tag mellea-langchain/v0.2.0
git push origin mellea-langchain/v0.2.0
```

### Example 3: Major Release with Breaking Changes

```bash
# Make breaking changes to mellea-crewai
cd mellea_contribs/crewai_backend
# Implement breaking changes...
# Update version from 0.9.0 to 1.0.0 in pyproject.toml

git add .
git commit -m "feat!: redesign API for better usability"
git push origin main

git tag mellea-crewai/v1.0.0
git push origin mellea-crewai/v1.0.0
```

## Release Checklist

Use this checklist when releasing a package:

- [ ] All changes committed and pushed to main
- [ ] Tests pass locally
- [ ] Version updated in `pyproject.toml`
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG updated (if maintained)
- [ ] Tag created with correct format
- [ ] Tag pushed to remote
- [ ] CI workflow completed successfully
- [ ] GitHub release created with artifacts
- [ ] Release notes reviewed and edited (if needed)
- [ ] Installation tested from release artifacts
- [ ] Team notified of new release