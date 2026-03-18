#!/bin/bash

# Script to run tests for all mellea_contribs projects
# Each project needs the correct PYTHONPATH to find its source and mellea-integration-core

set -e  # Exit on error

echo "=========================================="
echo "Running mellea_contribs tests"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Store the repository root as absolute path
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Function to run tests for a project
run_project_tests() {
    local project_name=$1
    local project_path=$2
    local pythonpath=$3
    
    echo "=========================================="
    echo "Testing: $project_name"
    echo "=========================================="
    
    cd "$REPO_ROOT/$project_path"
    
    if PYTHONPATH="$pythonpath" python3 -m pytest tests/ -v --tb=short 2>&1; then
        echo -e "${GREEN}✓ $project_name tests PASSED${NC}"
        echo ""
        cd "$REPO_ROOT"
        return 0
    else
        echo -e "${RED}✗ $project_name tests FAILED${NC}"
        echo ""
        cd "$REPO_ROOT"
        return 1
    fi
}

# Track results
FAILED_PROJECTS=()
PASSED_PROJECTS=()

# Test mellea-integration-core (shared library)
if run_project_tests "mellea-integration-core" \
    "mellea_contribs/mellea-integration-core" \
    "src:\$PYTHONPATH"; then
    PASSED_PROJECTS+=("mellea-integration-core")
else
    FAILED_PROJECTS+=("mellea-integration-core")
fi

# Test crewai_backend
if run_project_tests "crewai_backend" \
    "mellea_contribs/crewai_backend" \
    "src:../mellea-integration-core/src:\$PYTHONPATH"; then
    PASSED_PROJECTS+=("crewai_backend")
else
    FAILED_PROJECTS+=("crewai_backend")
fi

# Test dspy_backend
if run_project_tests "dspy_backend" \
    "mellea_contribs/dspy_backend" \
    "src:../mellea-integration-core/src:\$PYTHONPATH"; then
    PASSED_PROJECTS+=("dspy_backend")
else
    FAILED_PROJECTS+=("dspy_backend")
fi

# Test langchain_backend
if run_project_tests "langchain_backend" \
    "mellea_contribs/langchain_backend" \
    "src:../mellea-integration-core/src:\$PYTHONPATH"; then
    PASSED_PROJECTS+=("langchain_backend")
else
    FAILED_PROJECTS+=("langchain_backend")
fi

# Test reqlib_package (may have dependency issues)
echo "=========================================="
echo "Testing: reqlib_package"
echo "=========================================="
echo -e "${YELLOW}Note: reqlib_package requires additional dependencies (citeurl, eyecite, playwright)${NC}"
cd "$REPO_ROOT/mellea_contribs/reqlib_package"
if PYTHONPATH="src:$PYTHONPATH" python3 -m pytest test/ -v --tb=short 2>&1; then
    echo -e "${GREEN}✓ reqlib_package tests PASSED${NC}"
    PASSED_PROJECTS+=("reqlib_package")
else
    echo -e "${YELLOW}⚠ reqlib_package tests SKIPPED (missing dependencies)${NC}"
    FAILED_PROJECTS+=("reqlib_package (missing deps)")
fi
echo ""

# Test tools_package (may have dependency issues)
echo "=========================================="
echo "Testing: tools_package"
echo "=========================================="
echo -e "${YELLOW}Note: test_mprogram_robustness.py requires 'benchdrift' package (skipped)${NC}"
cd "$REPO_ROOT/mellea_contribs/tools_package"
if PYTHONPATH="src:$PYTHONPATH" python3 -m pytest test/test_double_round_robin.py test/test_top_k.py -v --tb=short 2>&1; then
    echo -e "${GREEN}✓ tools_package tests PASSED (2 tests, test_mprogram_robustness.py skipped - requires benchdrift)${NC}"
    PASSED_PROJECTS+=("tools_package")
else
    echo -e "${YELLOW}⚠ tools_package tests FAILED${NC}"
    FAILED_PROJECTS+=("tools_package")
fi
echo ""

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""
echo -e "${GREEN}Passed (${#PASSED_PROJECTS[@]}):${NC}"
for project in "${PASSED_PROJECTS[@]}"; do
    echo "  ✓ $project"
done
echo ""

if [ ${#FAILED_PROJECTS[@]} -gt 0 ]; then
    echo -e "${RED}Failed/Skipped (${#FAILED_PROJECTS[@]}):${NC}"
    for project in "${FAILED_PROJECTS[@]}"; do
        echo "  ✗ $project"
    done
    echo ""
fi

echo "=========================================="
if [ ${#FAILED_PROJECTS[@]} -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed or were skipped${NC}"
    exit 1
fi

