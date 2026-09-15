"""
The failure chart on the screen: which kind of failure is growing, and what we would fix.

/failures shows the failure mix of live runs per week -- how many runs rested in each of the
six categories -- as plain bars, and under it one line per category saying what the fix for
that kind is. With no failed run yet it says so rather than showing an empty chart. It shows
counts and category names only: no customer text, no worker names.
"""

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
