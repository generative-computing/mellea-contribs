#!/usr/bin/env python3
"""Validate package version and detect build system from pyproject.toml"""
import sys
import tomllib
from pathlib import Path


def extract_version(toml_path):
    """Extract version from pyproject.toml"""
    with open(toml_path, 'rb') as f:
        data = tomllib.load(f)
    return data.get('project', {}).get('version', '')


def extract_build_backend(toml_path):
    """Extract build backend from pyproject.toml"""
    with open(toml_path, 'rb') as f:
        data = tomllib.load(f)
    return data.get('build-system', {}).get('build-backend', '')


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_build.py <action> [args]")
        print("  validate_build.py version <tag_version>")
        print("  validate_build.py backend")
        sys.exit(1)

    action = sys.argv[1]
    toml_path = Path('pyproject.toml')

    if not toml_path.exists():
        print("Error: pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    if action == 'version':
        if len(sys.argv) < 3:
            print("Error: tag_version required", file=sys.stderr)
            sys.exit(1)

        tag_version = sys.argv[2]
        pyproject_version = extract_version(toml_path)

        print(f"Version in pyproject.toml: {pyproject_version}")
        print(f"Version from tag: {tag_version}")

        if pyproject_version != tag_version:
            print("Error: Version mismatch!", file=sys.stderr)
            print(f"  pyproject.toml version: {pyproject_version}", file=sys.stderr)
            print(f"  Tag version: {tag_version}", file=sys.stderr)
            sys.exit(1)

        print(f"version={pyproject_version}")

    elif action == 'backend':
        build_backend = extract_build_backend(toml_path)
        print(f"Detected build backend: {build_backend}")
        print(f"build_backend={build_backend}")

    else:
        print(f"Error: Unknown action '{action}'", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
