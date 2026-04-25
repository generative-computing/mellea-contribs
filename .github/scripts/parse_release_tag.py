#!/usr/bin/env python3
import os
import re
import sys
import tomllib
from pathlib import Path

def main():
    tag = os.environ.get("GITHUB_REF", "").replace("refs/tags/", "")
    print(f"Full tag: {tag}")

    # Validate tag format
    pattern = r"^[a-z0-9-]+/v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$"
    if not re.match(pattern, tag):
        print("Error: Tag must be in format 'package-name/vX.Y.Z' or 'package-name/vX.Y.Z-prerelease'")
        print("Example: mellea-dspy/v0.1.0 or mellea-dspy/v0.1.0-beta.1")
        sys.exit(1)

    # Extract package name and version
    package_name = tag.split("/v")[0]
    version = tag.split("/v",1)[1]

    print(f"Package name: {package_name}")
    print(f"Version: {version}")

    # Discover packages
    print("Discovering packages in mellea_contribs/...")
    package_dir = None
    found_packages = []

    mellea_contribs = Path("mellea_contribs")
    for dir_path in sorted(mellea_contribs.iterdir()):
        if not dir_path.is_dir():
            continue

        pyproject = dir_path / "pyproject.toml"
        if not pyproject.exists():
            continue

        try:
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
                pkg_name = data.get("project", {}).get("name", "")

                if pkg_name:
                    found_packages.append(f"  - {pkg_name} -> {dir_path}")
                    if pkg_name == package_name:
                        package_dir = str(dir_path)
                        print(f"Found matching package: {pkg_name} in {package_dir}")
        except Exception as e:
            print(f"Warning: Failed to parse {pyproject}: {e}", file=sys.stderr)

    # Validate that we found the package
    if not package_dir:
        print(f"Error: Package '{package_name}' not found in any pyproject.toml file")
        print("")
        print("Available packages:")
        for pkg in found_packages:
            print(pkg)
        sys.exit(1)

    print(f"Package directory: {package_dir}")

    # Output for next jobs
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"package_name={package_name}\n")
            f.write(f"version={version}\n")
            f.write(f"package_dir={package_dir}\n")

if __name__ == "__main__":
    main()
