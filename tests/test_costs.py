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
from datetime import UTC, date, datetime
from decimal import Decimal

import psycopg
import pytest

from app.costs import Day, spend_by_day, spend_chart, tokens_by_model

TODAY = date(2026, 9, 16)
YESTERDAY = date(2026, 9, 15)

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
