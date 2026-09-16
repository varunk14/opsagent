"""
The trend: one line per accepted baseline, committed beside it.

`evals/history.jsonl` is append-only. Every acceptance adds one line -- the date, the commit
it was accepted on, the model and prompt versions, the completion, the violations, the failure
mix and the golden set's hash -- so the failure chart shows the mix moving from one accepted
baseline to the next, and git shows who accepted what. The gate checks the file ends with the
baseline it compares against, so a history edited on its own, or a baseline accepted without
its line, fails rather than passing quietly.

The commit is context, not a measure: outside a git checkout it reads "unknown" and nothing
else changes.
"""

import json
from datetime import date

import pytest

from evals.history import COMPARED, append, code_version, ends_with, entry, last
from evals.scoring import empty_mix, scoreboard_of


@pytest.fixture
def board():
    from evals.golden import load_cases
    from tests.test_evals_scoring import perfect

    cases = load_cases()[:3]
    return scoreboard_of(cases, [perfect(case) for case in cases])


def test_an_empty_history_has_no_last_line(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("\n  \n", encoding="utf-8")

    assert last(path) is None


def test_a_history_without_a_line_never_ends_with_a_baseline(tmp_path, board):
    path = tmp_path / "history.jsonl"
    path.write_text("", encoding="utf-8")

    assert ends_with(path, board) is False


def test_a_line_that_is_not_an_object_is_not_a_last_line(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")

    assert last(path) is None


def test_the_appended_line_is_the_one_the_gate_compares(tmp_path, board):
    path = tmp_path / "history.jsonl"
    path.touch()

    written = append(path, board)

    assert ends_with(path, board)
    assert all(name in written for name in COMPARED)
    assert last(path) == written


def test_a_line_written_for_another_scoreboard_does_not_end_the_history(tmp_path, board):
    from dataclasses import replace

    path = tmp_path / "history.jsonl"
    path.touch()
    append(path, replace(board, completed=board.completed - 1, failure_mix=empty_mix()))

    assert ends_with(path, board) is False


def test_the_date_is_recorded_as_given_rather_than_today(board):
    assert entry(board, date(2026, 1, 2))["accepted_on"] == "2026-01-02"


def test_outside_a_git_checkout_the_commit_is_unknown(monkeypatch):
    def refuses(*args, **kwargs):
        raise OSError("git is not on this machine")

    monkeypatch.setattr("evals.history.subprocess.run", refuses)

    assert code_version() == "unknown"


def test_an_empty_mix_names_every_category_with_no_count():
    mix = empty_mix()

    assert set(mix) == {"hallucinated_field", "tool_misuse", "loop", "context_overflow", "wrong_escalation", "drift"}
    assert set(mix.values()) == {0}
