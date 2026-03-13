#!/usr/bin/env python3
"""Integration Test Runner for Mellea Framework Integrations.

This script provides a unified interface to run integration tests across
all Mellea framework integrations (dspy, crewai, langchain).

Usage:
    # Run all integration tests
    python run_integration_tests.py

    # Run tests for specific framework(s)
    python run_integration_tests.py --frameworks dspy crewai

    # Run with specific pytest options
    python run_integration_tests.py --pytest-args "-v -s"

    # List available frameworks
    python run_integration_tests.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path


class IntegrationTestRunner:
    """Manages running integration tests across multiple framework integrations."""

    FRAMEWORKS = {
        "dspy": {
            "path": "dspy",
            "test_path": "dspy/tests/integration",
            "markers": ["integration"],
            "description": "DSPy integration tests",
        },
        "crewai": {
            "path": "crewai",
            "test_path": "crewai/tests/integration",
            "markers": ["integration", "llm"],
            "description": "CrewAI integration tests",
        },
        "langchain": {
            "path": "langchain",
            "test_path": "langchain/tests/integration",
            "markers": ["integration"],
            "description": "LangChain integration tests",
        },
    }

    def __init__(self, workspace_dir: Path):
        """Initialize the test runner.

        Args:
            workspace_dir: Path to the workspace directory containing integrations.
        """
        self.workspace_dir = workspace_dir

    def list_frameworks(self) -> None:
        """List all available frameworks and their test information."""
        print("\nAvailable Framework Integrations:")
        print("=" * 70)
        for name, info in self.FRAMEWORKS.items():
            print(f"\n{name.upper()}")
            print(f"  Description: {info['description']}")
            print(f"  Test Path: {info['test_path']}")
            print(f"  Markers: {', '.join(info['markers'])}")

    def check_framework_exists(self, framework: str) -> bool:
        """Check if a framework directory exists."""
        framework_path = self.workspace_dir / self.FRAMEWORKS[framework]["path"]
        return framework_path.exists()

    def install_dependencies(self, framework: str) -> bool:
        """Install dependencies for a specific framework."""
        framework_path = self.workspace_dir / self.FRAMEWORKS[framework]["path"]
        pyproject_path = framework_path / "pyproject.toml"

        if not pyproject_path.exists():
            print(f"⚠️  No pyproject.toml found for {framework}")
            return False

        print(f"\n📦 Installing dependencies for {framework}...")
        try:
            # Try uv first (faster), fall back to pip
            result = subprocess.run(
                ["uv", "pip", "install", "-e", f"{framework_path}[dev]"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                # Fall back to pip
                result = subprocess.run(
                    ["pip", "install", "-e", f"{framework_path}[dev]"],
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True,
                )

            if result.returncode == 0:
                print(f"✅ Dependencies installed for {framework}")
                return True
            else:
                print(f"❌ Failed to install dependencies for {framework}")
                print(result.stderr)
                return False
        except FileNotFoundError:
            print("⚠️  Neither uv nor pip found. Please install dependencies manually.")
            return False

    def run_tests(
        self,
        frameworks: list[str],
        pytest_args: str | None = None,
        install_deps: bool = False,
        verbose: bool = False,
    ) -> dict:
        """Run integration tests for specified frameworks.

        Args:
            frameworks: List of framework names to test
            pytest_args: Additional pytest arguments
            install_deps: Whether to install dependencies first
            verbose: Whether to show verbose output

        Returns:
            Dictionary with test results for each framework
        """
        results = {}

        for framework in frameworks:
            if framework not in self.FRAMEWORKS:
                print(f"❌ Unknown framework: {framework}")
                results[framework] = {"status": "unknown", "returncode": -1}
                continue

            if not self.check_framework_exists(framework):
                print(f"❌ Framework directory not found: {framework}")
                results[framework] = {"status": "not_found", "returncode": -1}
                continue

            # Install dependencies if requested
            if install_deps:
                if not self.install_dependencies(framework):
                    results[framework] = {"status": "deps_failed", "returncode": -1}
                    continue

            # Build pytest command
            test_path = self.workspace_dir / self.FRAMEWORKS[framework]["test_path"]
            cmd = ["pytest", str(test_path)]

            # Add markers
            markers = self.FRAMEWORKS[framework]["markers"]
            if markers:
                cmd.extend(["-m", " and ".join(markers)])

            # Add verbose flag
            if verbose:
                cmd.append("-v")

            # Add custom pytest args
            if pytest_args:
                cmd.extend(pytest_args.split())

            # Run tests
            print(f"\n{'=' * 70}")
            print(f"🧪 Running {framework.upper()} integration tests")
            print(f"{'=' * 70}")
            print(f"Command: {' '.join(cmd)}\n")

            result = subprocess.run(
                cmd, cwd=self.workspace_dir / self.FRAMEWORKS[framework]["path"]
            )

            results[framework] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
            }

        return results

    def print_summary(self, results: dict) -> None:
        """Print a summary of test results."""
        print(f"\n{'=' * 70}")
        print("📊 TEST SUMMARY")
        print(f"{'=' * 70}\n")

        passed = sum(1 for r in results.values() if r["status"] == "passed")
        failed = sum(1 for r in results.values() if r["status"] == "failed")
        other = len(results) - passed - failed

        for framework, result in results.items():
            status_icon = {
                "passed": "✅",
                "failed": "❌",
                "unknown": "❓",
                "not_found": "🔍",
                "deps_failed": "📦",
            }.get(result["status"], "❓")

            print(f"{status_icon} {framework.upper()}: {result['status']}")

        print(f"\n{'=' * 70}")
        print(
            f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Other: {other}"
        )
        print(f"{'=' * 70}\n")


def main() -> int:
    """Run the integration test runner CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Run integration tests for Mellea framework integrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all integration tests
  python run_integration_tests.py

  # Run tests for specific frameworks
  python run_integration_tests.py --frameworks dspy langchain

  # Install dependencies and run tests
  python run_integration_tests.py --install-deps

  # Run with verbose output
  python run_integration_tests.py -v

  # Pass custom pytest arguments
  python run_integration_tests.py --pytest-args "-k test_basic -x"

  # List available frameworks
  python run_integration_tests.py --list
        """,
    )

    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["dspy", "crewai", "langchain"],
        help="Specific frameworks to test (default: all)",
    )

    parser.add_argument(
        "--list", action="store_true", help="List available frameworks and exit"
    )

    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies before running tests",
    )

    parser.add_argument(
        "--pytest-args",
        type=str,
        help="Additional arguments to pass to pytest (in quotes)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Get workspace directory (script location)
    workspace_dir = Path(__file__).parent.resolve()
    runner = IntegrationTestRunner(workspace_dir)

    # Handle --list
    if args.list:
        runner.list_frameworks()
        return 0

    # Determine which frameworks to test
    frameworks = args.frameworks or list(runner.FRAMEWORKS.keys())

    # Run tests
    results = runner.run_tests(
        frameworks=frameworks,
        pytest_args=args.pytest_args,
        install_deps=args.install_deps,
        verbose=args.verbose,
    )

    # Print summary
    runner.print_summary(results)

    # Exit with error if any tests failed
    if any(r["status"] == "failed" for r in results.values()):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
