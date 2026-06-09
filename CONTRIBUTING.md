# Contributing to `mellea-contribs`

`mellea-contribs` is a monorepo of bleeding-edge contributions that extend Mellea
without going through core review. Subpackages here are opinionated, leaf-installed,
and evictable: when one falls behind a recent mellea release or breaks against
`mellea@main` for too long, it gets archived. Code that is stable, broadly useful,
and on the core roadmap belongs in [mellea](https://github.com/generative-computing/mellea)
itself, not here. If a contribs subpackage matures into something everyone wants
in core, file an issue in the core repo and propose graduation.

## Where to go next

- **Adding a new contribs subpackage?** See [`docs/contributing/new-subpackage.md`](docs/contributing/new-subpackage.md).
- **Cutting a release?** See [`RELEASING.md`](RELEASING.md).

## Quick links

- **Cookiecutter template**: [`cookiecutter/`](cookiecutter/) — scaffolds a new subpackage onto the layout `validate-structure` enforces.
- **Daily smoke matrix**: [`.github/smoke-matrix.json`](.github/smoke-matrix.json) — subpackages exercised every night against `mellea@main`.
- **Structural contract**: [`.github/scripts/validate_package_contract.py`](.github/scripts/validate_package_contract.py) — what every PR is checked against.
- **Issues**: <https://github.com/generative-computing/mellea-contribs/issues>.
- **Owners**: each subpackage's `OWNERS` file lists the GitHub usernames on the hook for it. Auto-issues @-mention every owner.
- **Code of Conduct**: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Applies to everyone, human or AI.

## Developer Certificate of Origin (DCO)

`mellea-contribs` uses the [Developer Certificate of Origin](https://developercertificate.org/)
to certify that contributors have the right to submit their work under the project's
license. By signing off on a commit, you are agreeing to the terms of the DCO (full
text below).

**Sign off every commit** using `-s` or `--signoff`:

```bash
git commit -s -m "feat: your commit message"
```

This appends a `Signed-off-by` trailer using your `user.name` and `user.email` from
git config:

```text
Signed-off-by: Jane Doe <jane@example.com>
```

Use your real name and a reachable email. PRs with unsigned commits will be blocked
by the DCO check until every commit is signed off. To retroactively sign existing
commits, use `git rebase --signoff <base>` and force-push.

<details>
<summary>Developer Certificate of Origin v1.1 (full text)</summary>

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

</details>

# Contribute to the Mellea Core Repository

Check out the documentation for contributing to [Mellea](https://github.com/generative-computing/mellea/blob/main/CONTRIBUTING.md).
