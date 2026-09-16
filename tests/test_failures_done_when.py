"""
The acceptance criterion for the failure taxonomy, word for word, as tests.

  "The dashboard shows the mix trending, and you can point at one category and say
   what you would fix."

Trending needs more than one measurement, so the test that matters uses the history this
repository actually committed: every baseline anyone accepted, in order, as the operator sees
it on the screen. A category that a change moved has to be visible as a shorter bar, and every
category has to name the work its bar would buy -- a chart that only says "17" tells nobody
what to do on Monday.
"""

import json

import pytest

from app.failures import FIXES, GOLDEN_HISTORY, FailureCategory, golden_trend
from tests.test_web import client_for

pytestmark = pytest.mark.db


def committed_history() -> list[dict]:
    lines = GOLDEN_HISTORY.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_the_committed_history_holds_more_than_one_accepted_baseline():
    """Without a second measurement there is no trend, only a number."""
    assert len(committed_history()) >= 2


def test_the_page_shows_every_accepted_baseline_the_repository_committed(fresh_database):
    page = client_for(fresh_database).get("/failures").text

    for accepted in committed_history():
        assert accepted["accepted_on"] in page
        assert accepted["code"] in page
        assert f"{accepted['completed']} of {accepted['cases']} complete" in page


def test_a_category_that_was_worked_on_is_visibly_shorter_than_it_was(fresh_database):
    """
    The point of the chart: a bar that fell says the work landed.

    Read from the committed history rather than hard-coded, so this keeps meaning what it says
    after the next accepted baseline instead of pinning today's numbers forever.
    """
    history = committed_history()
    first, last = history[0]["failure_mix"], history[-1]["failure_mix"]

    moved = {kind for kind in first if last[kind] < first[kind]}

    assert moved, "no category improved across the accepted baselines"
    trend = golden_trend(GOLDEN_HISTORY)
    for kind in moved:
        widths = [width for row in (trend[0], trend[-1]) for category, _, width in row.mix if category == kind]
        assert widths[-1] < widths[0], f"{kind} fell but its bar did not"


def test_every_category_says_what_we_would_fix(fresh_database):
    page = client_for(fresh_database).get("/failures").text

    assert [category for category, _ in FIXES] == list(FailureCategory)
    for category, fix in FIXES:
        assert category.value in page
        assert fix[:40] in page, f"{category.value} names no fix on the page"


def test_the_trend_survives_a_history_that_only_grows(fresh_database, tmp_path):
    """One line per accepted baseline, oldest first, however many there come to be."""
    path = tmp_path / "history.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "accepted_on": f"2026-09-{day:02d}",
                    "code": f"{day:07d}",
                    "cases": 150,
                    "completed": 80 + day,
                    "failure_mix": {"loop": 20 - day, "wrong_escalation": 40 - day},
                },
                sort_keys=True,
            )
            + "\n"
            for day in (14, 15, 16, 17)
        ),
        encoding="utf-8",
    )

    page = client_for(fresh_database, history_path=path).get("/failures").text

    assert [f"2026-09-{day:02d}" in page for day in (14, 15, 16, 17)] == [True] * 4
    assert page.index("2026-09-14") < page.index("2026-09-17"), "oldest first"
