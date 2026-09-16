"""
The failure chart on the screen: which kind of failure is growing, and what we would fix.

/failures shows the failure mix of live runs per week -- how many runs rested in each of the
six categories -- as plain bars, and under it one line per category saying what the fix for
that kind is. With no failed run yet it says so rather than showing an empty chart. It shows
counts and category names only: no customer text, no worker names.
"""

import json
from datetime import timedelta

import psycopg
import pytest
from psycopg.types.json import Jsonb

from app.failures import FailureCategory
from tests.test_web import client_for

pytestmark = pytest.mark.db


def rested(dsn: str, category: str | None, *, days_ago: int) -> None:
    """A run that rested `days_ago` days ago with this category, inserted as the worker would have left it."""
    with psycopg.connect(dsn) as connection:
        connection.execute(
            "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key, failure_category, created_at) "
            "VALUES (gen_random_uuid(), 'email', 'waiting_approval', 'plan', %s, gen_random_uuid()::text, %s, now() - %s)",
            (Jsonb({"untrusted": {"body": "<script>alert('pwned')</script>"}}), category, timedelta(days=days_ago)),
        )


def test_the_failures_page_is_linked_from_every_screen(fresh_database):
    assert 'href="/failures"' in client_for(fresh_database).get("/runs").text


def test_with_no_failed_run_the_page_says_so(fresh_database):
    page = client_for(fresh_database).get("/failures")

    assert page.status_code == 200
    assert "No failed runs yet" in page.text


def test_every_category_is_named_with_what_we_would_fix(fresh_database):
    page = client_for(fresh_database).get("/failures").text

    for category in FailureCategory:
        assert category.value in page
    assert "what we would fix" in page.lower()


def test_the_mix_is_shown_per_week_with_bars_scaled_to_the_largest_count(fresh_database):
    rested(fresh_database, "loop", days_ago=1)
    rested(fresh_database, "loop", days_ago=1)
    rested(fresh_database, "tool_misuse", days_ago=1)
    rested(fresh_database, "wrong_escalation", days_ago=21)
    rested(fresh_database, None, days_ago=1)  # did what it should: not on the chart

    page = client_for(fresh_database).get("/failures").text

    assert 'style="width: 100%"' in page  # loop, 2 of 2
    assert 'style="width: 50%"' in page  # tool_misuse and wrong_escalation, 1 of 2 each
    assert "4 failed runs" in page
    assert page.count("week of ") == 2


def test_the_page_shows_no_customer_text_and_no_worker(fresh_database):
    rested(fresh_database, "loop", days_ago=1)
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET locked_by = 'worker-7'")

    page = client_for(fresh_database).get("/failures").text

    assert "pwned" not in page
    assert "worker-7" not in page


def history_file(path, *lines: dict):
    """A history of accepted baselines, written as `python -m evals accept` writes it."""
    path.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8")
    return path


def baseline(day: str, code: str, mix: dict[str, int], completed: int) -> dict:
    return {"accepted_on": day, "code": code, "cases": 150, "completed": completed, "failure_mix": mix}


def test_with_no_accepted_baseline_the_trend_says_so(fresh_database, tmp_path):
    page = client_for(fresh_database, history_path=tmp_path / "absent.jsonl").get("/failures").text

    assert "No accepted baseline yet" in page


def test_the_golden_trend_shows_one_row_per_accepted_baseline(fresh_database, tmp_path):
    path = history_file(
        tmp_path / "history.jsonl",
        baseline("2026-09-16", "c12eee8", {"loop": 17, "tool_misuse": 3, "wrong_escalation": 41}, completed=89),
        baseline("2026-09-17", "abc1234", {"loop": 17, "tool_misuse": 3}, completed=130),
    )

    page = client_for(fresh_database, history_path=path).get("/failures").text

    assert "2026-09-16" in page and "c12eee8" in page
    assert "2026-09-17" in page and "abc1234" in page
    assert "89 of 150" in page and "130 of 150" in page
    assert page.count("wrong_escalation") >= 2  # once per baseline, plus the fix list


def test_the_trend_is_separate_from_the_live_runs_chart(fresh_database, tmp_path):
    rested(fresh_database, "loop", days_ago=1)
    path = history_file(tmp_path / "history.jsonl", baseline("2026-09-16", "c12eee8", {"loop": 17}, completed=89))

    page = client_for(fresh_database, history_path=path).get("/failures").text

    assert "1 failed run" in page  # the live chart counts one
    assert "17" in page  # the trend counts seventeen
