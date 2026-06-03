"""Auto-issue bot for the daily smoke job against ``mellea@main``.

Lifecycle (per the contribs maintainer playbook):

* The smoke workflow records a failure for a subpackage by invoking
  ``auto_issue_bot.py --action record-failure``. The first red is silent;
  on the second consecutive red the bot opens a tracking issue labelled
  ``contribs-broken`` and assigns the package's OWNERS. Subsequent reds
  add a comment to the existing issue.
* The smoke workflow records a recovery by invoking
  ``--action record-recovery``. The bot adds a "smoke green again on
  <date>, fixed in <sha>" comment, removes the ``contribs-broken`` label,
  and resets the consecutive-failure counter and the label-applied-days
  counter. The issue itself stays open until a human closes it.
* A separate daily workflow invokes ``--action apply-archival``. The bot
  walks each tracked package: if the issue still carries the
  ``contribs-broken`` label it bumps ``label_applied_days`` and posts the
  appropriate milestone comment (day 7 reminder, day 14 release-notes
  mention, day 21 archival label).

The bot exposes two execution modes:

* **Fake mode** (the default in tests, and the fallback when no
  ``GITHUB_TOKEN`` is set in the environment) keeps state in memory via
  :class:`FakeGitHub`. The unit tests drive the bot end to end through
  this fake.
* **Real mode** uses :class:`RealGitHub`, a thin wrapper over PyGithub.
  Persistent state is stored as JSON on a bot-managed branch in the
  repository (see :func:`_load_persistent_state` /
  :func:`_save_persistent_state`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


# Threshold for opening an issue: at least this many consecutive failed runs.
FAILURE_THRESHOLD = 2

# Milestones (in days of contribs-broken label persistence).
DAY_7_REMINDER = 7
DAY_14_RELEASE_NOTES_MENTION = 14
DAY_21_ARCHIVAL = 21

CONTRIBS_BROKEN_LABEL = "contribs-broken"
ARCHIVED_LABEL = "archived"

# Path inside the bot-managed branch where persistent state lives.
STATE_FILE_PATH = ".github/bot-state/auto_issue_bot.json"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class BotState:
    """Persistent state tracked across bot invocations.

    Attributes:
        consecutive_failures: Per-package count of consecutive failed
            smoke runs since the last green.
        open_issue_numbers: Per-package issue number for the current
            tracking issue (``None`` if the bot has never opened one).
        label_applied_days: Per-package days the ``contribs-broken``
            label has continuously been present on the tracking issue.
            Reset to 0 on recovery and bumped by ``apply-archival``.
        last_failure_run_url: Per-package URL of the most recent failed
            workflow run (used in issue bodies and follow-up comments).
        milestones_posted: Per-package list of milestone names already
            commented on the tracking issue (e.g. ``["day-7"]``); used
            to keep ``apply-archival`` idempotent.
    """

    consecutive_failures: dict[str, int] = field(default_factory=dict)
    open_issue_numbers: dict[str, int] = field(default_factory=dict)
    label_applied_days: dict[str, int] = field(default_factory=dict)
    last_failure_run_url: dict[str, str] = field(default_factory=dict)
    milestones_posted: dict[str, list[str]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "BotState":
        data = json.loads(text)
        return cls(
            consecutive_failures=dict(data.get("consecutive_failures", {})),
            open_issue_numbers=dict(data.get("open_issue_numbers", {})),
            label_applied_days=dict(data.get("label_applied_days", {})),
            last_failure_run_url=dict(data.get("last_failure_run_url", {})),
            milestones_posted={
                k: list(v) for k, v in data.get("milestones_posted", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# GitHub client interface
# ---------------------------------------------------------------------------


class GitHubClient(Protocol):
    """Narrow interface the bot needs from GitHub."""

    def open_issue(
        self,
        *,
        title: str,
        body: str,
        labels: list[str],
        assignees: list[str],
    ) -> int: ...

    def add_comment(self, issue_number: int, body: str) -> None: ...

    def add_label(self, issue_number: int, label: str) -> None: ...

    def remove_label(self, issue_number: int, label: str) -> None: ...

    def get_issue_labels(self, issue_number: int) -> list[str]: ...


@dataclass
class FakeGitHub:
    """In-memory ``GitHubClient`` used by tests and dry runs."""

    issues_opened: list[dict[str, Any]] = field(default_factory=list)
    comments_added: list[str] = field(default_factory=list)
    next_issue_number: int = 1

    def open_issue(
        self,
        *,
        title: str,
        body: str,
        labels: list[str],
        assignees: list[str],
    ) -> int:
        n = self.next_issue_number
        self.next_issue_number += 1
        self.issues_opened.append(
            {
                "number": n,
                "title": title,
                "body": body,
                "labels": list(labels),
                "assignees": list(assignees),
                "state": "open",
            }
        )
        return n

    def add_comment(self, issue_number: int, body: str) -> None:
        self.comments_added.append(body)
        for issue in self.issues_opened:
            if issue["number"] == issue_number:
                issue.setdefault("comments", []).append(body)

    def add_label(self, issue_number: int, label: str) -> None:
        for issue in self.issues_opened:
            if issue["number"] == issue_number and label not in issue["labels"]:
                issue["labels"].append(label)

    def remove_label(self, issue_number: int, label: str) -> None:
        for issue in self.issues_opened:
            if issue["number"] == issue_number and label in issue["labels"]:
                issue["labels"].remove(label)

    def get_issue_labels(self, issue_number: int) -> list[str]:
        for issue in self.issues_opened:
            if issue["number"] == issue_number:
                return list(issue["labels"])
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _load_owners(package_dir: Path) -> list[str]:
    """Read the OWNERS file for a subpackage; return ``@handle`` lines.

    A missing or empty OWNERS file results in an empty list (the issue
    is opened unassigned).
    """
    owners_file = package_dir / "OWNERS"
    if not owners_file.exists():
        return []
    handles: list[str] = []
    for raw in owners_file.read_text().splitlines():
        line = raw.strip()
        if line.startswith("@"):
            handles.append(line)
    return handles


def _issue_title(package: str) -> str:
    return f"[{CONTRIBS_BROKEN_LABEL}] {package}: smoke red since {_today_iso()}"


def _issue_body(package: str, run_url: str, owners: list[str]) -> str:
    owners_line = " ".join(owners) if owners else "(no OWNERS file found)"
    return (
        f"The daily smoke job has failed against `mellea@main` for the\n"
        f"`{package}` subpackage on {FAILURE_THRESHOLD} consecutive runs.\n\n"
        f"Latest failing run: {run_url}\n\n"
        f"OWNERS: {owners_line}\n\n"
        f"This issue is tracked by the auto-issue bot. While the\n"
        f"`{CONTRIBS_BROKEN_LABEL}` label is present the archival timeline\n"
        f"runs: a reminder is posted on day {DAY_7_REMINDER}, the\n"
        f"subpackage is mentioned in the next release notes on\n"
        f"day {DAY_14_RELEASE_NOTES_MENTION}, and the\n"
        f"`{ARCHIVED_LABEL}` label is applied on day {DAY_21_ARCHIVAL}.\n"
        f"Removing the `{CONTRIBS_BROKEN_LABEL}` label resets that clock.\n"
    )


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------


def record_failure(
    gh: GitHubClient,
    state: BotState,
    *,
    package: str,
    run_url: str,
) -> None:
    """Record a failed smoke run for ``package``.

    Below the failure threshold this only bumps the in-memory counter.
    At the threshold it opens a fresh tracking issue. Above the
    threshold it appends a "still red" comment to the existing issue.
    """
    state.consecutive_failures[package] = (
        state.consecutive_failures.get(package, 0) + 1
    )
    state.last_failure_run_url[package] = run_url
    n = state.consecutive_failures[package]

    if n < FAILURE_THRESHOLD:
        return

    if n == FAILURE_THRESHOLD and package not in state.open_issue_numbers:
        owners = _load_owners(Path(package))
        issue_n = gh.open_issue(
            title=_issue_title(package),
            body=_issue_body(package, run_url, owners),
            labels=[CONTRIBS_BROKEN_LABEL],
            assignees=[o.lstrip("@") for o in owners],
        )
        state.open_issue_numbers[package] = issue_n
        state.label_applied_days[package] = 0
        state.milestones_posted[package] = []
        return

    issue_n = state.open_issue_numbers.get(package)
    if issue_n is not None:
        gh.add_comment(
            issue_n,
            f"Still red after run {run_url} (consecutive failure #{n}).",
        )


def record_recovery(
    gh: GitHubClient,
    state: BotState,
    *,
    package: str,
    commit_sha: str,
) -> None:
    """Record a green smoke run for ``package``.

    Resets the consecutive-failure counter and, if a tracking issue is
    open, posts a recovery comment, removes the ``contribs-broken``
    label, and resets the archival clock. The issue itself is left
    open for a human to close.
    """
    had_failures = state.consecutive_failures.get(package, 0) > 0
    state.consecutive_failures[package] = 0

    issue_n = state.open_issue_numbers.get(package)
    if issue_n is None or not had_failures:
        return

    gh.add_comment(
        issue_n,
        f"smoke green again on {_today_iso()}, fixed in {commit_sha}",
    )
    gh.remove_label(issue_n, CONTRIBS_BROKEN_LABEL)
    state.label_applied_days[package] = 0
    state.milestones_posted[package] = []


def apply_archival_timeline(
    gh: GitHubClient,
    state: BotState,
    *,
    today_iso: str | None = None,
) -> None:
    """Walk tracked packages and post the day-7 / 14 / 21 milestones.

    For each package with an open tracking issue still carrying the
    ``contribs-broken`` label, this function bumps
    ``label_applied_days`` by one and, when a milestone is hit for the
    first time, posts the corresponding comment (and applies the
    ``archived`` label on day 21). Packages whose label has been
    cleared are skipped (the recovery path zeroed the counter).
    """
    today = today_iso or _today_iso()

    for package in list(state.open_issue_numbers):
        issue_n = state.open_issue_numbers[package]
        labels = gh.get_issue_labels(issue_n)
        if CONTRIBS_BROKEN_LABEL not in labels:
            # Label was cleared — recovery already reset the clock.
            continue

        days = state.label_applied_days.get(package, 0) + 1
        state.label_applied_days[package] = days
        posted = state.milestones_posted.setdefault(package, [])

        if days >= DAY_7_REMINDER and "day-7" not in posted:
            gh.add_comment(
                issue_n,
                f"Day {DAY_7_REMINDER} reminder ({today}): the smoke job "
                f"for `{package}` has been red for one week. Please "
                f"investigate or escalate to the contribs maintainers.",
            )
            posted.append("day-7")

        if (
            days >= DAY_14_RELEASE_NOTES_MENTION
            and "day-14" not in posted
        ):
            gh.add_comment(
                issue_n,
                f"Day {DAY_14_RELEASE_NOTES_MENTION} notice ({today}): "
                f"`{package}` will be called out as broken in the next "
                f"contribs release notes.",
            )
            posted.append("day-14")

        if days >= DAY_21_ARCHIVAL and "day-21" not in posted:
            gh.add_label(issue_n, ARCHIVED_LABEL)
            gh.add_comment(
                issue_n,
                f"Archival triggered on {today} after {days} days of "
                f"`{CONTRIBS_BROKEN_LABEL}` label persistence. The "
                f"`{ARCHIVED_LABEL}` label has been applied; contribs "
                f"maintainers will move the subpackage to the archived "
                f"layout in a follow-up PR.",
            )
            posted.append("day-21")


# ---------------------------------------------------------------------------
# Real-mode (PyGithub) — wired but optional
# ---------------------------------------------------------------------------


@dataclass
class RealGitHub:
    """Thin wrapper over PyGithub implementing :class:`GitHubClient`.

    Constructed only inside :func:`_real_main` so that fake-mode runs
    (and the test suite) do not require ``PyGithub`` to be installed.
    """

    repo: Any  # github.Repository.Repository (kept loose to avoid import-time dep)

    def open_issue(
        self,
        *,
        title: str,
        body: str,
        labels: list[str],
        assignees: list[str],
    ) -> int:
        issue = self.repo.create_issue(
            title=title,
            body=body,
            labels=labels,
            assignees=assignees or [],
        )
        return int(issue.number)

    def add_comment(self, issue_number: int, body: str) -> None:
        self.repo.get_issue(issue_number).create_comment(body)

    def add_label(self, issue_number: int, label: str) -> None:
        self.repo.get_issue(issue_number).add_to_labels(label)

    def remove_label(self, issue_number: int, label: str) -> None:
        self.repo.get_issue(issue_number).remove_from_labels(label)

    def get_issue_labels(self, issue_number: int) -> list[str]:
        return [lbl.name for lbl in self.repo.get_issue(issue_number).labels]


def _load_persistent_state(repo: Any, branch: str) -> BotState:
    """Fetch the bot-state JSON from a managed branch (real-mode only).

    Returns a fresh :class:`BotState` if the file (or the branch) does
    not exist yet.
    """
    try:
        contents = repo.get_contents(STATE_FILE_PATH, ref=branch)
    except Exception:  # pragma: no cover - first run / missing branch
        return BotState()
    decoded = contents.decoded_content.decode("utf-8")
    return BotState.from_json(decoded)


def _save_persistent_state(
    repo: Any,
    branch: str,
    state: BotState,
    *,
    commit_message: str,
) -> None:
    """Write the bot-state JSON back to the managed branch."""
    payload = state.to_json()
    try:
        existing = repo.get_contents(STATE_FILE_PATH, ref=branch)
        repo.update_file(
            path=STATE_FILE_PATH,
            message=commit_message,
            content=payload,
            sha=existing.sha,
            branch=branch,
        )
    except Exception:  # pragma: no cover - first commit on the branch
        repo.create_file(
            path=STATE_FILE_PATH,
            message=commit_message,
            content=payload,
            branch=branch,
        )


def _real_main(args: argparse.Namespace) -> int:  # pragma: no cover
    """Run the bot against a real GitHub repository via PyGithub."""
    try:
        from github import Github  # type: ignore[import-not-found]
    except ImportError:
        print(
            "PyGithub is required for real-mode operation. "
            "Install it with `uv sync` (it lives in the dev group), "
            "or rerun with --fake to use the in-memory client.",
            file=sys.stderr,
        )
        return 2

    token = os.environ["GITHUB_TOKEN"]
    repo_full_name = os.environ["GITHUB_REPOSITORY"]
    branch = os.environ.get("BOT_STATE_BRANCH", "bot-state/auto-issue-bot")

    gh_api = Github(token)
    repo = gh_api.get_repo(repo_full_name)
    gh = RealGitHub(repo=repo)

    state = _load_persistent_state(repo, branch)

    if args.action == "record-failure":
        record_failure(gh, state, package=args.package, run_url=args.run_url)
        msg = f"bot: record failure for {args.package}"
    elif args.action == "record-recovery":
        record_recovery(
            gh, state, package=args.package, commit_sha=args.commit_sha
        )
        msg = f"bot: record recovery for {args.package}"
    elif args.action == "apply-archival":
        apply_archival_timeline(gh, state)
        msg = "bot: apply archival timeline"
    else:
        print(f"Unknown action: {args.action}", file=sys.stderr)
        return 2

    _save_persistent_state(repo, branch, state, commit_message=msg)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        required=True,
        choices=["record-failure", "record-recovery", "apply-archival"],
    )
    parser.add_argument("--package", default=None)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--commit-sha", default="")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Force fake (in-memory) mode even if GITHUB_TOKEN is set.",
    )
    return parser


def _fake_main(args: argparse.Namespace) -> int:
    """Run the bot against an in-memory ``FakeGitHub`` (no API calls)."""
    gh = FakeGitHub()
    state = BotState()
    if args.action == "record-failure":
        if not args.package:
            print("--package is required for record-failure", file=sys.stderr)
            return 2
        record_failure(gh, state, package=args.package, run_url=args.run_url)
    elif args.action == "record-recovery":
        if not args.package:
            print("--package is required for record-recovery", file=sys.stderr)
            return 2
        record_recovery(
            gh, state, package=args.package, commit_sha=args.commit_sha
        )
    elif args.action == "apply-archival":
        apply_archival_timeline(gh, state)

    print(
        json.dumps(
            {
                "issues_opened": gh.issues_opened,
                "comments_added": gh.comments_added,
                "state": asdict(state),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.fake or not os.environ.get("GITHUB_TOKEN"):
        return _fake_main(args)
    return _real_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
