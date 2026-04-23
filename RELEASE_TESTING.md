# Release Workflow Testing Guide

This document provides test scenarios to validate the release workflow for mellea_contribs packages.

## Test Scenarios

### Scenario 1: Successful Release (Happy Path)

**Objective**: Verify a complete successful release flow

**Steps**:
1. Choose a test package (e.g., `mellea-dspy`)
2. Update version in `mellea_contribs/dspy_backend/pyproject.toml` to `0.1.1`
3. Commit and push changes
4. Create tag: `git tag mellea-dspy/v0.1.1`
5. Push tag: `git push origin mellea-dspy/v0.1.1`

**Expected Results**:
- ✅ Workflow triggers automatically
- ✅ Tag is parsed correctly
- ✅ Package directory is identified
- ✅ Version validation passes
- ✅ Package builds successfully
- ✅ Tests run and pass
- ✅ GitHub release is created
- ✅ Release includes wheel file (.whl)
- ✅ Release includes source distribution (.tar.gz)
- ✅ Release includes SHA256SUMS
- ✅ Release notes are generated
- ✅ Installation instructions are included

**Validation**:
```bash
# Check workflow status
gh run list --workflow=release-package.yml

# Check release exists
gh release view mellea-dspy/v0.1.1

# Download and test installation
wget https://github.com/generative-computing/mellea-contribs/releases/download/mellea-dspy/v0.1.1/mellea-dspy-0.1.1-py3-none-any.whl
pip install mellea-dspy-0.1.1-py3-none-any.whl
python -c "import mellea_dspy; print(mellea_dspy.__version__)"
```

---

### Scenario 2: Version Mismatch Detection

**Objective**: Verify version validation catches mismatches

**Steps**:
1. Update version in `pyproject.toml` to `0.1.2`
2. Commit and push
3. Create tag with different version: `git tag mellea-dspy/v0.1.3`
4. Push tag: `git push origin mellea-dspy/v0.1.3`

**Expected Results**:
- ✅ Workflow triggers
- ✅ Tag parsing succeeds
- ❌ Version validation fails with clear error message
- ❌ Build does not proceed
- ❌ No release is created

**Error Message Should Include**:
```
Error: Version mismatch!
  pyproject.toml version: 0.1.2
  Tag version: 0.1.3
```

---

### Scenario 3: Invalid Tag Format

**Objective**: Verify tag format validation

**Test Cases**:

**3a. Missing package name**:
```bash
git tag v0.1.0
git push origin v0.1.0
```
Expected: ❌ Workflow fails with "Tag must be in format 'package-name/vX.Y.Z'"

**3b. Wrong separator**:
```bash
git tag mellea-dspy-v0.1.0
git push origin mellea-dspy-v0.1.0
```
Expected: ❌ Workflow fails with format error

**3c. Missing 'v' prefix**:
```bash
git tag mellea-dspy/0.1.0
git push origin mellea-dspy/0.1.0
```
Expected: ❌ Workflow fails with format error

**3d. Invalid package name**:
```bash
git tag unknown-package/v0.1.0
git push origin unknown-package/v0.1.0
```
Expected: ❌ Workflow fails with "Unknown package name 'unknown-package'"

---

### Scenario 4: Multiple Package Releases

**Objective**: Verify independent package releases don't interfere

**Steps**:
1. Release `mellea-dspy/v0.1.1`
2. Wait for completion
3. Release `mellea-crewai/v0.1.1`
4. Wait for completion
5. Release `mellea-langchain/v0.1.1`

**Expected Results**:
- ✅ Each release completes independently
- ✅ Each release has its own GitHub release
- ✅ Artifacts don't mix between releases
- ✅ Release notes are package-specific

---

### Scenario 5: Build System Detection

**Objective**: Verify both build systems work correctly

**Test Cases**:

**5a. Hatchling (mellea-dspy, mellea-crewai, mellea-langchain)**:
```bash
git tag mellea-dspy/v0.1.1
git push origin mellea-dspy/v0.1.1
```
Expected: ✅ Detects hatchling, builds successfully

**5b. PDM Backend (mellea-reqlib, mellea-tools)**:
```bash
git tag mellea-reqlib/v0.0.2
git push origin mellea-reqlib/v0.0.2
```
Expected: ✅ Detects pdm-backend, builds successfully

---

### Scenario 6: Test Failures

**Objective**: Verify workflow handles test failures gracefully

**Steps**:
1. Introduce a failing test in the package
2. Commit and push
3. Update version and create tag
4. Push tag

**Expected Results**:
- ✅ Workflow triggers
- ✅ Build succeeds
- ⚠️ Tests run but fail
- ⚠️ Warning is logged but build continues
- ✅ Release is still created (tests are non-blocking)

**Note**: Tests are currently non-blocking to allow releases even if some tests fail. This can be changed by modifying the test step in `build-package.yml`.

---

### Scenario 7: Changelog Generation

**Objective**: Verify release notes include commit history

**Steps**:
1. Make several commits to a package
2. Create a release tag
3. Check release notes

**Expected Results**:
- ✅ Release notes include "Changes since X.Y.Z"
- ✅ Commits are listed with hashes
- ✅ Installation instructions are included
- ✅ Package contents are listed

---

### Scenario 8: Re-releasing Same Version

**Objective**: Verify behavior when re-releasing

**Steps**:
1. Release `mellea-dspy/v0.1.1`
2. Delete the tag and release
3. Make changes
4. Re-release `mellea-dspy/v0.1.1`

**Expected Results**:
- ✅ Old release is replaced
- ✅ New artifacts are uploaded
- ✅ Release notes are regenerated

---

### Scenario 9: Artifact Verification

**Objective**: Verify all artifacts are present and valid

**Steps**:
1. Create a release
2. Download all artifacts
3. Verify contents

**Validation**:
```bash
# Download artifacts
gh release download mellea-dspy/v0.1.1

# Verify wheel
unzip -l mellea-dspy-0.1.1-py3-none-any.whl

# Verify source distribution
tar -tzf mellea-dspy-0.1.1.tar.gz

# Verify checksums
sha256sum -c SHA256SUMS
```

**Expected Results**:
- ✅ Wheel contains correct package structure
- ✅ Source distribution contains all source files
- ✅ Checksums match
- ✅ All files are present

---

### Scenario 10: Installation Testing

**Objective**: Verify packages can be installed from releases

**Test Cases**:

**10a. Install from wheel**:
```bash
pip install https://github.com/generative-computing/mellea-contribs/releases/download/mellea-dspy/v0.1.1/mellea-dspy-0.1.1-py3-none-any.whl
```

**10b. Install from source distribution**:
```bash
pip install https://github.com/generative-computing/mellea-contribs/releases/download/mellea-dspy/v0.1.1/mellea-dspy-0.1.1.tar.gz
```

**10c. Install from git tag**:
```bash
git clone https://github.com/generative-computing/mellea-contribs.git
cd mellea-contribs
git checkout mellea-dspy/v0.1.1
cd mellea_contribs/dspy_backend
pip install .
```

**Expected Results**:
- ✅ All installation methods work
- ✅ Package imports successfully
- ✅ Dependencies are installed
- ✅ Version is correct

---

## Automated Testing Checklist

Before merging workflow changes, verify:

- [ ] All 5 packages can be released successfully
- [ ] Version validation works
- [ ] Tag format validation works
- [ ] Both build systems (hatchling and pdm) work
- [ ] Release notes are generated correctly
- [ ] Artifacts are uploaded correctly
- [ ] Checksums are valid
- [ ] Packages can be installed from releases
- [ ] Multiple concurrent releases don't interfere
- [ ] Re-releasing works correctly

## Manual Testing Commands

```bash
# Test tag parsing locally
TAG="mellea-dspy/v0.1.1"
PACKAGE_NAME="${TAG%/v*}"
VERSION="${TAG#*/v}"
echo "Package: $PACKAGE_NAME, Version: $VERSION"

# Test version extraction from pyproject.toml
cd mellea_contribs/dspy_backend
grep -E '^version = ' pyproject.toml | sed -E 's/version = "(.*)"/\1/'

# Test build locally
cd mellea_contribs/dspy_backend
uv sync --all-extras
uv build
ls -lh dist/

# Test installation locally
pip install dist/*.whl
python -c "import mellea_dspy; print('Success!')"
```

## Troubleshooting Test Failures

### Workflow doesn't trigger
- Check tag format matches `*/v*` pattern
- Verify tag was pushed to remote
- Check GitHub Actions is enabled

### Version validation fails
- Verify version in pyproject.toml matches tag
- Check for extra whitespace or quotes
- Ensure version follows semver format

### Build fails
- Run build locally first
- Check for missing dependencies
- Verify pyproject.toml is valid

### Tests fail
- Run tests locally: `pytest tests/`
- Check test dependencies are installed
- Review test logs in workflow

### Release not created
- Check workflow permissions
- Verify GITHUB_TOKEN has write access
- Check for errors in "Create Release" step

### Artifacts missing
- Verify dist/ directory contains files
- Check upload-artifact step succeeded
- Verify download-artifact step succeeded

## Performance Testing

Monitor workflow performance:

```bash
# Check workflow duration
gh run list --workflow=release-package.yml --json durationMs

# Average duration should be:
# - Parse tag: < 30 seconds
# - Build: 2-5 minutes
# - Create release: < 1 minute
# Total: < 10 minutes
```

## Security Testing

Verify security aspects:

- [ ] Workflow uses pinned action versions
- [ ] No secrets are exposed in logs
- [ ] Artifacts are properly scoped
- [ ] Permissions follow least privilege
- [ ] Checksums are generated correctly

## Regression Testing

After workflow changes, test:

1. All 5 packages release successfully
2. Previous releases are not affected
3. Tag history is preserved
4. Release notes remain accessible
5. Old artifacts remain downloadable

## Success Criteria

A release is considered successful when:

1. ✅ Workflow completes without errors
2. ✅ GitHub release is created
3. ✅ All artifacts are present (wheel, sdist, checksums)
4. ✅ Release notes are generated
5. ✅ Package can be installed from release
6. ✅ Package imports and works correctly
7. ✅ Version matches tag
8. ✅ Checksums are valid
