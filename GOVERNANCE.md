# Governance

## Overview

The `mellea-contribs` repository is an incubation point for contributions to the Mellea ecosystem. It contains multiple packages — framework integrations, tools, and libraries — each of which may be maintained by different individuals or teams.

This document defines how packages are structured and maintained, who has authority over what, and how the project is governed.

## Packages

### Package Requirements

Every package in the repository must meet the following standards:

- **`pyproject.toml`** — Each package is a self-contained Python project with its own `pyproject.toml` defining dependencies, metadata, and tool configuration.
- **Tests** — Each package must include a `tests/` (or `test/`) directory with at least one test module. All tests must be runnable via `pytest`. See [Testing](#testing) for details.
- **CI workflow** — Each package must have a CI workflow file (`.github/workflows/ci-<package-name>.yml`) that calls the shared `quality-generic.yml` reusable workflow. A validation workflow automatically checks that every subpackage has both a CI workflow and a test directory.
- **Documentation** — Each package must include a README with usage documentation. Packages may also include a CONTRIBUTING.md if their contribution workflow has package-specific requirements.
- **License** — Each package must use a license compatible with the project (Apache 2.0 preferred).

### Package Structure

Packages live under `mellea_contribs/` and follow this general layout:

```
mellea_contribs/<package_name>/
├── pyproject.toml
├── README.md
├── src/
│   └── <import_name>/
│       └── ...
└── tests/
    ├── test_*.py
    └── integration/     # optional
        └── test_*.py
```

### Package Ownership

Each package has one or more designated maintainers who are responsible for its health and direction. Package maintainers have autonomy over technical decisions within their package, including:

- API design and implementation approach
- Dependency choices (within project-wide constraints)
- Release cadence for their package
- Internal test organization and contribution guidelines

The core team retains override authority on any decision and may intervene when:

- A package decision affects other packages or the project as a whole
- Project-wide standards or policies are not being met
- A package becomes unmaintained

### Package Maintenance

Maintainers are expected to keep their packages in good working order. This means:

- **CI stays green** — Tests should be passing on the main branch. Flaky or broken tests should be addressed promptly, not left failing.
- **Dependencies stay current** — Packages should remain compatible with supported versions of Mellea core and their framework dependencies. When Mellea core releases a new version, maintainers should verify compatibility and update as needed.
- **Python version support** — Packages should support the Python versions tested in CI (currently 3.11, 3.12, and 3.13). When the project adds or drops Python versions, maintainers should update accordingly.
- **Documentation stays accurate** — READMEs and usage docs should reflect the current state of the package. Outdated documentation is a maintenance issue.
- **Security issues are addressed** — Dependabot alerts, reported vulnerabilities, and security-related issues should be treated as high priority.

#### When a Package Falls Behind

If a package is not meeting the standards above, the following escalation path applies:

1. **The core team flags the issue** — by opening an issue on the package describing what needs attention.
2. **The maintainer has a reasonable window to respond** — either by fixing the issue, providing a timeline, or explaining why the current state is acceptable.
3. **If there is no response**, the core team may seek a new maintainer by posting a call for volunteers.
4. **If no new maintainer steps up**, the core team may retire the package (see [Retiring Packages](#retiring-packages)).

The goal is not to be punitive — maintainers are volunteers and life happens. The escalation path exists to ensure that packages in the repository remain usable and trustworthy for the broader community.

### Accepting New Packages

To propose a new package, open an issue describing:

1. The package's purpose and scope
2. How it fits within the Mellea ecosystem (i.e., what need it addresses that is not already covered)
3. Who will maintain it

The core team will review and decide on acceptance. Accepted packages must meet all the [Package Requirements](#package-requirements) before merging.

### Retiring Packages

A package may be retired by the core team if:

- It has no active maintainer and no one volunteers to take over
- It is superseded by functionality in another package or in Mellea core
- It no longer meets project-wide standards and the maintainer is unresponsive

Retired packages will be removed from the repository. The core team will make reasonable efforts to notify users and provide a migration path before removal.

## Testing

Every package is required to have its own test suite. This section describes project-wide testing conventions; individual packages may layer on additional requirements.

### Requirements

1. **Test directory** — Every package must contain a `tests/` (or `test/`) directory with at least one test module.
2. **Test runner** — All tests must be runnable via `pytest`.
3. **CI workflow** — Every package must have a CI workflow file that calls the shared `quality-generic.yml` workflow.
4. **Tests must pass** — A package's CI checks must pass for any PR that touches that package. Failures in one package do not block PRs to other packages.

### Conventions

- **Integration tests** — Tests that require external services (LLM providers, Ollama, APIs) should be placed in a `tests/integration/` subdirectory or marked with `@pytest.mark.integration`. Integration tests are excluded from CI by default (`--ignore=tests/integration`).
- **Async support** — Use `asyncio_mode = "auto"` in `pyproject.toml` so async tests do not need explicit `@pytest.mark.asyncio` decorators.
- **Custom markers** — Packages that define custom pytest markers (e.g., `qualitative`, `slow`, `requires_api_key`) should document them in their `pyproject.toml` under `[tool.pytest.ini_options]`.

### How CI Runs Tests

Each package's CI workflow passes the package path to the shared `quality-generic.yml` workflow, which:

1. Sets up the Python matrix (3.11, 3.12, 3.13 by default)
2. Installs the package's dependencies via `uv sync --dev`
3. Runs `pytest -v tests/ --ignore=tests/integration` (or `test/`)

Packages that need Ollama or other services for unit tests can configure this through workflow inputs. See the existing CI workflows for examples.

## Merge Policy

Pull requests require the following before merging:

1. **At least one review** from a member, maintainer, or core team member
2. **Approval from a package maintainer** (or core team member) for the affected package(s)
3. **All CI checks pass**
4. **No unresolved "request changes" reviews**

For pull requests that span multiple packages, approval is required from a maintainer of each affected package (or a core team member).

### Contributor Responsibility

When a PR is accepted, the maintainers inherit both an asset and a liability: the new functionality is (hopefully) an asset, but the code and documentation required to support it are ongoing liabilities. Reviewers invest real time reading and understanding every PR.

Contributors are responsible for every line they submit. Do not ask others to read and maintain code or documentation that you have not taken the time to read, understand, and refine yourself. This applies regardless of how the code was produced — whether written by hand or generated with AI tools.

**Use of AI tools:** We neither prohibit nor discourage the use of AI coding assistants. However, AI-generated code is not exempt from this standard. If you use AI tools to produce code or documentation, you are expected to review, understand, and take full ownership of the output before submitting it for review. The reviewer's time is not the place to discover what the AI wrote on your behalf.

## Roles

### Contributor

Anyone who contributes to the project in any form — code, documentation, bug reports, reviews, or discussion. No formal requirements. All contributors are expected to follow the project's code of conduct.

### Member

An established contributor who has demonstrated sustained interest in the project through multiple contributions. Members are added to the GitHub organization and gain:

- The ability to be assigned to issues and pull requests
- Automatic CI runs on their pull requests (no manual approval needed)
- Eligibility to review pull requests

**Requirements:**
- Multiple meaningful contributions to the repository
- Sponsorship by at least one existing member, package maintainer, or core team member

### Package Maintainer

A maintainer is responsible for the health and direction of a specific package. Maintainers have authority to approve and merge pull requests for their package.

**Responsibilities:**
- Review and approve pull requests for their package
- Triage issues related to their package
- Ensure their package meets project-wide quality standards (tests, documentation, CI)
- Mentor contributors working within their package

**How to become a maintainer:**
1. **Contribute** — Submit pull requests, report bugs, review code, participate in discussions
2. **Become a member** — After sustained contributions, request membership (or be nominated)
3. **Become a package maintainer** — After demonstrating familiarity with a package through contributions and reviews, you may be nominated by an existing maintainer or core team member. Nomination is accepted if there are no objections from other maintainers of that package within one week

### Core Team

The [`generative-computing/mellea-contributors`](https://github.com/orgs/generative-computing/teams/mellea-contributors) GitHub team serves as the core team for this repository. The core team has overall oversight of the project and is responsible for:

- Project-wide policies, standards, and direction
- Accepting or retiring packages
- Resolving cross-package conflicts and escalations
- Managing releases and project infrastructure
- Approving changes to governance

Core team members may approve and merge pull requests for any package in the repository.

## Decision Making

**Within a package:** Package maintainers use lazy consensus — a proposal is considered accepted if no maintainer objects within a reasonable timeframe. For significant design decisions (new APIs, breaking changes), maintainers should open an issue or discussion to solicit feedback before proceeding.

**Project-wide decisions:** The core team decides on matters that affect the project as a whole, including governance changes, accepting or retiring packages, and cross-package concerns. The core team aims for consensus; if consensus cannot be reached, decisions are made by majority vote of the core team.

## Evolution of This Document

This governance model is intentionally lightweight. As the contributor community grows, we expect to add:

- Formal inactivity and emeritus policies
- A CODEOWNERS file mapping packages to maintainers
- Company diversity guidelines for maintainer nominations
- A structured proposal process for significant cross-package changes
- Detailed voting and escalation procedures

Changes to this document require approval from the core team.
