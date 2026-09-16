"""
What the burn-down shows, and what it refuses to believe.

Two of the numbers on this page are read out of `runs.state`, which the agent writes as JSON. A page
that renders whatever it finds there is a page that can be drawn by whatever reached the agent, so
the reading is as suspicious of that column as app/failures.py is of the golden history.
"""

from datetime import date
from decimal import Decimal

import psycopg
import pytest

from app.costs import Day, spend_by_day, spend_chart, tokens_by_model

TODAY = date(2026, 9, 16)
YESTERDAY = date(2026, 9, 15)


# --- the burn-down ----------------------------------------------------------------------------


def test_each_day_says_what_it_spent_and_on_how_many_runs():
    chart = spend_chart([(YESTERDAY, 4, Decimal("0.004000")), (TODAY, 2, Decimal("0.001000"))])

    assert [(day.on, day.runs, day.spent) for day in chart] == [
        (YESTERDAY, 4, Decimal("0.004000")),
        (TODAY, 2, Decimal("0.001000")),
    ]


def test_a_day_is_measured_by_what_a_run_cost_not_by_the_day_s_total():
    """Four cheap runs must not look worse than one expensive one; the page is about cost per run."""
    chart = spend_chart([(YESTERDAY, 4, Decimal("0.004000")), (TODAY, 1, Decimal("0.003000"))])

    assert [day.each for day in chart] == [Decimal("0.001000"), Decimal("0.003000")]
    assert [day.width for day in chart] == [33, 100], "the dearest run fills the bar, not the busiest day"


def test_a_day_that_ran_nothing_is_not_divided_by_nothing():
    chart = spend_chart([(TODAY, 0, Decimal("0.000000"))])

    assert chart == [Day(on=TODAY, runs=0, spent=Decimal("0.000000"), each=Decimal("0"), width=0)]


def test_nothing_spent_anywhere_draws_no_bars_rather_than_failing():
    """Every cost zero makes the dearest run zero, and a share of zero is a division by it."""
    chart = spend_chart([(YESTERDAY, 2, Decimal("0")), (TODAY, 3, Decimal("0"))])

    assert [day.width for day in chart] == [0, 0]


def test_no_runs_at_all_charts_nothing():
    assert spend_chart([]) == []


@pytest.mark.db
def test_the_burn_down_counts_the_runs_in_the_database(fresh_database: str):
    with psycopg.connect(fresh_database) as connection:
        rows = spend_by_day(connection)

    assert rows, "the seeded database has runs, so the burn-down cannot be empty"
    assert all(isinstance(spent, Decimal) for _, _, spent in rows)
    assert all(runs > 0 for _, runs, _ in rows)


@pytest.mark.db
def test_a_run_charged_nothing_yet_counts_as_nothing_not_as_unknown(fresh_database: str):
    """cost_usd is null until a run rests; summing it must not make the day's total null."""
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET cost_usd = NULL")
        rows = spend_by_day(connection)

    assert rows
    assert all(spent == Decimal(0) for _, _, spent in rows)


# --- tokens by model --------------------------------------------------------------------------


@pytest.mark.db
def test_tokens_are_added_up_per_model(fresh_database: str):
    with psycopg.connect(fresh_database) as connection:
        connection.execute(
            """UPDATE runs SET state = jsonb_set(state, '{agent,tokens_by_model}',
               '{"llama3.1:8b": [100, 20], "llama3.2": [10, 5]}'::jsonb)"""
        )
        counted = dict(tokens_by_model(connection))
        (runs,) = connection.execute("SELECT count(*) FROM runs").fetchone()

    assert counted["llama3.1:8b"] == (100 * runs, 20 * runs)
    assert counted["llama3.2"] == (10 * runs, 5 * runs)


@pytest.mark.db
def test_a_run_from_before_the_tokens_were_kept_is_not_counted(fresh_database: str):
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET state = state #- '{agent,tokens_by_model}'")

        assert tokens_by_model(connection) == []


@pytest.mark.db
@pytest.mark.parametrize(
    "written",
    [
        '{"llama3.1:8b": [-100, 20]}',  # a negative number of tokens
        '{"llama3.1:8b": ["100", "20"]}',  # tokens as text
        '{"llama3.1:8b": [true, false]}',  # booleans, which Python would count as 1 and 0
        '{"llama3.1:8b": [100]}',  # half a pair
        '{"llama3.1:8b": "100"}',  # not a pair at all
        '{"llama3.1:8b": null}',
    ],
)
def test_a_token_count_that_is_not_a_count_is_read_as_none(fresh_database: str, written: str):
    """`state` is written by the agent from what a model said, so the page may not trust its shape."""
    with psycopg.connect(fresh_database) as connection:
        connection.execute(
            "UPDATE runs SET state = jsonb_set(state, '{agent,tokens_by_model}', %s::jsonb)", (written,)
        )
        counted = dict(tokens_by_model(connection))

    assert counted.get("llama3.1:8b", (0, 0)) == (0, 0)


@pytest.mark.db
def test_a_model_name_longer_than_a_model_name_is_not_drawn(fresh_database: str):
    """A name is a key in JSON the agent wrote; an essay there would be rendered onto the page."""
    with psycopg.connect(fresh_database) as connection:
        connection.execute(
            "UPDATE runs SET state = jsonb_set(state, '{agent,tokens_by_model}', %s::jsonb)",
            ('{"' + "m" * 500 + '": [10, 5]}',),
        )
        counted = tokens_by_model(connection)

    assert all(len(name) <= 64 for name, _ in counted)
