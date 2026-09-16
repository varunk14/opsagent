"""
What the burn-down shows, and what it refuses to believe.

The tokens on this page are read out of `runs.state`, which the agent writes as JSON from what a
model said. A page that renders whatever it finds there is a page that can be drawn by whatever
reached the agent, so the reading is as suspicious of that column as app/failures.py is of the
golden history.

Runs are inserted directly rather than produced by a scripted case: this is a read-side page, and
what is under test is the arithmetic over rows, not the worker that writes them.
"""

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import psycopg
import pytest

from app.costs import Day, spend_by_day, spend_chart, tokens_by_model

# Relative to now, not written down: the queries only look back SHOWN_DAYS, so fixed dates would
# pass today and start failing a month from today for a reason no one would look for here.
TODAY = datetime.now(UTC).date()
YESTERDAY = TODAY - timedelta(days=1)

INSERT = """
    INSERT INTO runs (id, channel, status, current_node, state, cost_usd, created_at)
    VALUES (gen_random_uuid(), 'email', 'done', 'rest', %s::jsonb, %s, %s)
"""


def a_run(connection, *, on: date = TODAY, cost: str = "0.001000", tokens: dict | None = None) -> None:
    """One rested run on a given day, optionally carrying what each model was asked for."""
    agent: dict = {} if tokens is None else {"tokens_by_model": tokens}
    connection.execute(
        INSERT,
        (json.dumps({"agent": agent}), Decimal(cost), datetime(on.year, on.month, on.day, 12, tzinfo=UTC)),
    )


# --- the burn-down ----------------------------------------------------------------------------


def test_each_day_says_what_it_spent_and_on_how_many_runs():
    chart = spend_chart([(YESTERDAY, 4, Decimal("0.004000")), (TODAY, 2, Decimal("0.001000"))])

    assert [(day.on, day.runs, day.spent) for day in chart] == [
        (YESTERDAY, 4, Decimal("0.004000")),
        (TODAY, 2, Decimal("0.001000")),
    ]


def test_a_day_is_measured_by_what_a_run_cost_not_by_the_day_s_total():
    """Four cheap runs must not look worse than one dear one; the page is about cost per run."""
    chart = spend_chart([(YESTERDAY, 4, Decimal("0.004000")), (TODAY, 1, Decimal("0.003000"))])

    assert [day.each for day in chart] == [Decimal("0.001000"), Decimal("0.003000")]
    assert [day.width for day in chart] == [33, 100], "the dearest run fills the bar, not the busiest day"


def test_a_day_that_ran_nothing_is_not_divided_by_nothing():
    chart = spend_chart([(TODAY, 0, Decimal("0.000000"))])

    assert chart == [Day(on=TODAY, runs=0, spent=Decimal("0.000000"), each=Decimal(0), width=0)]


def test_nothing_spent_anywhere_draws_no_bars_rather_than_failing():
    """Every cost zero makes the dearest run zero, and a share of zero is a division by it."""
    chart = spend_chart([(YESTERDAY, 2, Decimal(0)), (TODAY, 3, Decimal(0))])

    assert [day.width for day in chart] == [0, 0]


def test_no_runs_at_all_charts_nothing():
    assert spend_chart([]) == []


@pytest.mark.db
def test_the_burn_down_groups_runs_by_the_day_they_ran(fresh_database: str):
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, on=YESTERDAY, cost="0.002000")
        a_run(connection, on=YESTERDAY, cost="0.004000")
        a_run(connection, on=TODAY, cost="0.001000")

        assert spend_by_day(connection) == [
            (YESTERDAY, 2, Decimal("0.006000")),
            (TODAY, 1, Decimal("0.001000")),
        ]


@pytest.mark.db
def test_a_run_that_has_not_rested_costs_nothing_rather_than_nothing_known(fresh_database: str):
    """`cost_usd` is NOT NULL and defaults to zero, so an unfinished run lowers the day's average."""
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, on=TODAY, cost="0.003000")
        connection.execute(
            "INSERT INTO runs (id, channel, status, current_node, state, created_at)"
            " VALUES (gen_random_uuid(), 'email', 'running', 'plan', '{}'::jsonb, %s)",
            (datetime(TODAY.year, TODAY.month, TODAY.day, 13, tzinfo=UTC),),
        )

        assert spend_by_day(connection) == [(TODAY, 2, Decimal("0.003000"))]


# --- tokens by model --------------------------------------------------------------------------


@pytest.mark.db
def test_tokens_are_added_up_per_model(fresh_database: str):
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, tokens={"llama3.1:8b": [100, 20], "llama3.2": [10, 5]})
        a_run(connection, tokens={"llama3.1:8b": [50, 8]})

        assert tokens_by_model(connection) == [("llama3.1:8b", (150, 28)), ("llama3.2", (10, 5))]


@pytest.mark.db
def test_a_run_from_before_the_tokens_were_kept_is_not_counted(fresh_database: str):
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, tokens=None)

        assert tokens_by_model(connection) == []


@pytest.mark.db
@pytest.mark.parametrize(
    "written",
    [
        {"llama3.1:8b": [-100, 20]},  # a negative number of tokens
        {"llama3.1:8b": ["100", "20"]},  # tokens as text
        {"llama3.1:8b": [True, False]},  # booleans, which Python would otherwise count as 1 and 0
        {"llama3.1:8b": [100]},  # half a pair
        {"llama3.1:8b": "100"},  # not a pair at all
        {"llama3.1:8b": None},
        # One bad half beside a good one. These are the shapes that catch a guard deciding by what
        # it returned rather than by what it was given: 0.0 and False are each rejected, and each is
        # numerically equal to a zero it would have accepted.
        {"llama3.1:8b": [0.0, 20]},
        {"llama3.1:8b": [1, False]},
        {"llama3.1:8b": [20, 0.0]},
    ],
)
def test_a_token_count_that_is_not_a_count_is_read_as_none(fresh_database: str, written: dict):
    """`state` is written by the agent from what a model said, so the page may not trust its shape."""
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, tokens=written)

        assert tokens_by_model(connection) == [("llama3.1:8b", (0, 0))]


@pytest.mark.db
def test_a_model_name_longer_than_a_model_name_is_not_drawn(fresh_database: str):
    """A name is a key in JSON the agent wrote; an essay there would be rendered onto the page."""
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, tokens={"m" * 500: [10, 5], "llama3.1:8b": [1, 1]})

        assert tokens_by_model(connection) == [("llama3.1:8b", (1, 1))]


# --- what the page is willing to read ------------------------------------------------------------
#
# Both questions are asked of a table that only grows. Unbounded, one page view reads every run ever
# and the burn-down gets slower for the rest of the agent's life -- on a screen whose other pages are
# how a person approves a refund while something is going wrong.


@pytest.mark.db
def test_the_burn_down_covers_a_window_not_all_of_history(fresh_database: str):
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, on=date(2020, 1, 1), cost="0.009000")
        a_run(connection, on=TODAY, cost="0.001000")

        charted = spend_by_day(connection)

    assert [day for day, _, _ in charted] == [TODAY], "2020 is outside the window"


@pytest.mark.db
def test_tokens_are_counted_over_the_same_window_as_the_chart(fresh_database: str):
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, on=date(2020, 1, 1), tokens={"llama3.1:8b": [9000, 900]})
        a_run(connection, on=TODAY, tokens={"llama3.1:8b": [10, 2]})

        assert tokens_by_model(connection) == [("llama3.1:8b", (10, 2))]


@pytest.mark.db
def test_only_so_many_runs_are_ever_read_for_one_page(fresh_database: str):
    """The newest are read first, so a cap loses the oldest rather than an arbitrary slice."""
    with psycopg.connect(fresh_database) as connection:
        for _ in range(4):
            a_run(connection, tokens={"llama3.1:8b": [10, 1]})

        assert tokens_by_model(connection, limit=2) == [("llama3.1:8b", (20, 2))]


@pytest.mark.db
def test_more_models_than_the_page_draws_keeps_the_ones_that_spent(fresh_database: str):
    """A row per model is a row an attacker could ask for; the page shows the biggest, not all."""
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, tokens={f"model-{index}": [index, index] for index in range(1, 6)})

        shown = tokens_by_model(connection, most=2)

    assert [name for name, _ in shown] == ["model-4", "model-5"], "the two that spent, still sorted by name"


def test_the_cost_per_run_is_rounded_to_money_not_left_as_a_division():
    """
    Found by mutation: nothing pinned the rounding, because Decimal compares by value.

    `Decimal("0.000333") == Decimal("0.00033333333...")` is False, but the test that looked at this
    used values that divided exactly, so dropping the quantize changed nothing it could see. A
    division that does not come out renders every digit it has onto the page.
    """
    chart = spend_chart([(TODAY, 3, Decimal("0.001000"))])

    assert str(chart[0].each) == "0.000333"


@pytest.mark.db
def test_the_run_cap_loses_the_oldest_runs_not_the_newest(fresh_database: str):
    """
    Found by mutation: the earlier cap test used identical runs, so the order could not matter.

    A burn-down that answered from the oldest rows would go stale the moment the table outgrew the
    cap, and keep reporting a number from whenever that happened.
    """
    with psycopg.connect(fresh_database) as connection:
        a_run(connection, on=TODAY - timedelta(days=3), tokens={"llama3.1:8b": [9000, 900]})
        a_run(connection, on=TODAY - timedelta(days=1), tokens={"llama3.1:8b": [20, 2]})
        a_run(connection, on=TODAY, tokens={"llama3.1:8b": [10, 1]})

        assert tokens_by_model(connection, limit=2) == [("llama3.1:8b", (30, 3))]
