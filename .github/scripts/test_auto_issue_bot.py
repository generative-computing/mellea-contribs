"""Tests for ``auto_issue_bot.py`` — drives the bot via the in-memory fake."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from github import GithubException

# Make the sibling module importable when pytest is invoked from the
# repo root (``uv run pytest .github/scripts/test_auto_issue_bot.py``).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from auto_issue_bot import (  # noqa: E402
    ARCHIVED_LABEL,
    CONTRIBS_BROKEN_LABEL,
    DAY_7_REMINDER,
    DAY_14_RELEASE_NOTES_MENTION,
    DAY_21_ARCHIVAL,
    BotState,
    FakeGitHub,
    apply_archival_timeline,
    record_failure,
    record_recovery,
    _EMPTY_BLOB_SHA,
    _ensure_state_branch,
)


# ---------------------------------------------------------------------------
# record_failure
# ---------------------------------------------------------------------------


def test_first_failure_does_not_open_issue() -> None:
    """A single red is silent — issues only open on the second consecutive red."""
    gh = FakeGitHub()
    state = BotState()
    record_failure(gh, state, package="dspy", run_url="https://example/run/1")
    assert gh.issues_opened == []
    assert state.consecutive_failures["dspy"] == 1
    assert "dspy" not in state.open_issue_numbers


def test_second_consecutive_failure_opens_issue() -> None:
    gh = FakeGitHub()
    state = BotState()
    record_failure(gh, state, package="dspy", run_url="https://example/run/1")
    record_failure(gh, state, package="dspy", run_url="https://example/run/2")
    assert len(gh.issues_opened) == 1
    issue = gh.issues_opened[0]
    assert issue["title"].startswith(f"[{CONTRIBS_BROKEN_LABEL}] dspy")
    assert CONTRIBS_BROKEN_LABEL in issue["labels"]
    assert "https://example/run/2" in issue["body"]
    assert state.open_issue_numbers["dspy"] == issue["number"]
    assert state.label_applied_days["dspy"] == 0


def test_team_owner_is_not_used_as_issue_assignee(tmp_path, monkeypatch) -> None:
    """A team OWNERS entry (@org/team) can't be an issue assignee — skip it.

    GitHub 422s if you try to assign an issue to a team; only individual
    users are dropped-in as assignees, though the team stays mentioned in
    the body via `owners`.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "OWNERS").write_text("@generative-computing/mellea-maintainers\n@akihikokuroda\n")
    monkeypatch.chdir(tmp_path)

    gh = FakeGitHub()
    state = BotState()
    record_failure(gh, state, package="pkg", run_url="r1")
    record_failure(gh, state, package="pkg", run_url="r2")

    issue = gh.issues_opened[0]
    # Only the individual is assignable; the team is filtered out.
    assert issue["assignees"] == ["akihikokuroda"]
    # But the team is still surfaced in the body for visibility.
    assert "mellea-maintainers" in issue["body"]


def test_third_failure_comments_on_existing_issue() -> None:
    gh = FakeGitHub()
    state = BotState()
    for i in range(3):
        record_failure(gh, state, package="dspy", run_url=f"https://example/run/{i}")
    assert len(gh.issues_opened) == 1
    assert len(gh.comments_added) == 1
    assert "Still red" in gh.comments_added[0]


def test_failures_are_per_package() -> None:
    gh = FakeGitHub()
    state = BotState()
    record_failure(gh, state, package="dspy", run_url="r1")
    record_failure(gh, state, package="langchain", run_url="r1")
    # Each package needs its own second failure to open an issue.
    assert gh.issues_opened == []
    record_failure(gh, state, package="dspy", run_url="r2")
    assert len(gh.issues_opened) == 1
    assert "dspy" in gh.issues_opened[0]["title"]


# ---------------------------------------------------------------------------
# record_recovery
# ---------------------------------------------------------------------------


def test_recovery_clears_label_but_leaves_issue_open() -> None:
    """On recovery the bot comments + removes the label; the issue stays open."""
    gh = FakeGitHub()
    state = BotState()
    record_failure(gh, state, package="dspy", run_url="r1")
    record_failure(gh, state, package="dspy", run_url="r2")
    record_recovery(gh, state, package="dspy", commit_sha="abc1234")

    assert state.consecutive_failures["dspy"] == 0
    issue = gh.issues_opened[0]
    assert CONTRIBS_BROKEN_LABEL not in issue["labels"]
    assert issue["state"] == "open"
    assert any("smoke green again" in c for c in gh.comments_added)
    assert any("abc1234" in c for c in gh.comments_added)


def test_rebreak_after_recovery_opens_a_new_issue() -> None:
    """A fresh breakage after recovery opens a new issue, not a comment on the old."""
    gh = FakeGitHub()
    state = BotState()
    record_failure(gh, state, package="dspy", run_url="r1")
    record_failure(gh, state, package="dspy", run_url="r2")  # opens issue #1
    record_recovery(gh, state, package="dspy", commit_sha="abc")
    assert "dspy" not in state.open_issue_numbers  # stopped tracking

    record_failure(gh, state, package="dspy", run_url="r3")
    record_failure(gh, state, package="dspy", run_url="r4")  # should open issue #2
    assert len(gh.issues_opened) == 2
    assert state.open_issue_numbers["dspy"] == gh.issues_opened[1]["number"]


def test_recovery_with_no_open_issue_is_a_noop() -> None:
    gh = FakeGitHub()
    state = BotState()
    record_recovery(gh, state, package="dspy", commit_sha="abc")
    assert gh.issues_opened == []
    assert gh.comments_added == []


def test_recovery_after_single_failure_does_not_comment() -> None:
    """A single red never opened an issue, so recovery should be silent."""
    gh = FakeGitHub()
    state = BotState()
    record_failure(gh, state, package="dspy", run_url="r1")
    record_recovery(gh, state, package="dspy", commit_sha="abc")
    assert gh.comments_added == []
    assert state.consecutive_failures["dspy"] == 0


# ---------------------------------------------------------------------------
# apply_archival_timeline — milestones based on label persistence
# ---------------------------------------------------------------------------


def _open_issue(gh: FakeGitHub, state: BotState, package: str) -> int:
    record_failure(gh, state, package=package, run_url="r1")
    record_failure(gh, state, package=package, run_url="r2")
    return state.open_issue_numbers[package]


def test_day_7_reminder_posted_once() -> None:
    gh = FakeGitHub()
    state = BotState()
    issue_n = _open_issue(gh, state, "dspy")

    # Tick days 1..7 — only the day-7 tick should produce a milestone comment.
    for _ in range(DAY_7_REMINDER):
        apply_archival_timeline(gh, state, today_iso="2026-06-08")

    day_7_comments = [c for c in gh.comments_added if "Day 7 reminder" in c]
    assert len(day_7_comments) == 1
    assert state.label_applied_days["dspy"] == DAY_7_REMINDER
    assert "day-7" in state.milestones_posted["dspy"]
    assert ARCHIVED_LABEL not in gh.get_issue_labels(issue_n)


def test_day_14_release_notes_mention() -> None:
    gh = FakeGitHub()
    state = BotState()
    _open_issue(gh, state, "dspy")
    state.label_applied_days["dspy"] = DAY_14_RELEASE_NOTES_MENTION - 1
    state.milestones_posted["dspy"] = ["day-7"]

    apply_archival_timeline(gh, state, today_iso="2026-06-15")
    assert any("Day 14 notice" in c for c in gh.comments_added)
    assert "day-14" in state.milestones_posted["dspy"]


def test_archival_after_21_days_of_label_persistence() -> None:
    """21 days of contribs-broken label persistence triggers archival."""
    gh = FakeGitHub()
    state = BotState()
    issue_n = _open_issue(gh, state, "dspy")
    state.label_applied_days["dspy"] = DAY_21_ARCHIVAL - 1
    state.milestones_posted["dspy"] = ["day-7", "day-14"]

    apply_archival_timeline(gh, state, today_iso="2026-06-22")
    assert ARCHIVED_LABEL in gh.get_issue_labels(issue_n)
    assert any("Archival triggered" in c for c in gh.comments_added)


def test_recovery_resets_archival_clock() -> None:
    """A recovery clears the label so subsequent ticks do not advance."""
    gh = FakeGitHub()
    state = BotState()
    issue_n = _open_issue(gh, state, "dspy")
    record_recovery(gh, state, package="dspy", commit_sha="abc")

    # 30 ticks pass — but the label is gone, so nothing should change.
    for _ in range(30):
        apply_archival_timeline(gh, state, today_iso="2026-06-29")

    assert ARCHIVED_LABEL not in gh.get_issue_labels(issue_n)
    # The counter never advanced past 0 once the label was cleared.
    assert state.label_applied_days["dspy"] == 0


def test_apply_archival_skips_packages_with_label_cleared() -> None:
    """Manually removing the label out-of-band should also skip the timeline."""
    gh = FakeGitHub()
    state = BotState()
    issue_n = _open_issue(gh, state, "dspy")
    gh.remove_label(issue_n, CONTRIBS_BROKEN_LABEL)

    apply_archival_timeline(gh, state, today_iso="2026-06-22")
    assert state.label_applied_days["dspy"] == 0
    assert ARCHIVED_LABEL not in gh.get_issue_labels(issue_n)


def test_apply_archival_is_idempotent() -> None:
    """Re-running on the same day after day-21 should not re-add comments."""
    gh = FakeGitHub()
    state = BotState()
    _open_issue(gh, state, "dspy")
    state.label_applied_days["dspy"] = DAY_21_ARCHIVAL - 1
    state.milestones_posted["dspy"] = ["day-7", "day-14"]

    apply_archival_timeline(gh, state, today_iso="2026-06-22")
    archival_comments_first = [
        c for c in gh.comments_added if "Archival triggered" in c
    ]
    assert len(archival_comments_first) == 1

    # A second tick on the same day adds no new archival comment because
    # day-21 has already been recorded in milestones_posted.
    apply_archival_timeline(gh, state, today_iso="2026-06-23")
    archival_comments_second = [
        c for c in gh.comments_added if "Archival triggered" in c
    ]
    assert len(archival_comments_second) == 1


# ---------------------------------------------------------------------------
# BotState round-trip
# ---------------------------------------------------------------------------


def test_bot_state_json_round_trip() -> None:
    state = BotState(
        consecutive_failures={"dspy": 3},
        open_issue_numbers={"dspy": 42},
        label_applied_days={"dspy": 5},
        last_failure_run_url={"dspy": "https://example/run/3"},
        milestones_posted={"dspy": ["day-7"]},
    )
    restored = BotState.from_json(state.to_json())
    assert restored.consecutive_failures == state.consecutive_failures
    assert restored.open_issue_numbers == state.open_issue_numbers
    assert restored.label_applied_days == state.label_applied_days
    assert restored.last_failure_run_url == state.last_failure_run_url
    assert restored.milestones_posted == state.milestones_posted


# ---------------------------------------------------------------------------
# _ensure_state_branch (real-mode branch bootstrap)
# ---------------------------------------------------------------------------


class _FakePyGithubRepo:
    """Stand-in for a PyGithub Repository covering _ensure_state_branch."""

    def __init__(self, *, ref_exists: bool, seed_status: int | None = None) -> None:
        self.ref_exists = ref_exists
        self.seed_status = seed_status
        self.tree: list[dict[str, str]] | None = None
        self.commit_parents: list[str] | None = None
        self.created_ref: tuple[str, str] | None = None

    def get_git_ref(self, ref: str) -> Any:
        if self.ref_exists:
            return object()
        raise GithubException(404, {"message": "Not Found"}, {})

    def create_git_tree(self, tree: list[dict[str, str]]) -> Any:
        self.tree = tree
        return SimpleNamespace(sha="tree-sha")

    def create_git_commit(self, message: str, tree: Any, parents: list[str]) -> Any:
        self.commit_parents = parents
        return SimpleNamespace(sha="seed-sha")

    def create_git_ref(self, ref: str, sha: str) -> None:
        if self.seed_status is not None:
            raise GithubException(self.seed_status, {"message": "exists"}, {})
        self.created_ref = (ref, sha)


def test_ensure_state_branch_seeds_orphan_when_missing() -> None:
    """A missing branch is seeded as an orphan commit holding only .gitkeep."""
    repo = _FakePyGithubRepo(ref_exists=False)
    _ensure_state_branch(repo, "bot-state/auto-issue-bot")
    assert repo.tree == [
        {
            "path": ".gitkeep",
            "mode": "100644",
            "type": "blob",
            "sha": _EMPTY_BLOB_SHA,
        }
    ]
    assert repo.commit_parents == []
    assert repo.created_ref == ("refs/heads/bot-state/auto-issue-bot", "seed-sha")


def test_ensure_state_branch_is_noop_when_ref_exists() -> None:
    """An existing branch is left untouched."""
    repo = _FakePyGithubRepo(ref_exists=True)
    _ensure_state_branch(repo, "bot-state/auto-issue-bot")
    assert repo.tree is None
    assert repo.created_ref is None


def test_ensure_state_branch_tolerates_race_on_create() -> None:
    """A concurrent leg seeding first shows up as 422, not an error."""
    repo = _FakePyGithubRepo(ref_exists=False, seed_status=422)
    _ensure_state_branch(repo, "bot-state/auto-issue-bot")
    assert repo.created_ref is None


def test_ensure_state_branch_raises_other_errors() -> None:
    """Non-404/422 failures (e.g. 403 permissions) surface to the caller."""
    repo = _FakePyGithubRepo(ref_exists=True)
    repo.get_git_ref = lambda ref: (_ for _ in ()).throw(  # type: ignore[assignment]
        GithubException(403, {"message": "Forbidden"}, {})
    )
    with pytest.raises(GithubException):
        _ensure_state_branch(repo, "bot-state/auto-issue-bot")
