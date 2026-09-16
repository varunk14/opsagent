"""
The burn-down on the screen: what a run costs, and where the tokens went.

/costs shows cost per run by the day it ran, as plain bars, and under it the tokens each model was
asked for. It shows money, counts and model names only -- no customer text ever reaches it.

It also states what the number leaves out. Embedding tokens are not in a run's cost, and a page that
published a cost per run without saying so would be publishing a number that is quietly too low.
"""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import psycopg
import pytest

from tests.test_web import client_for

pytestmark = pytest.mark.db


def a_run(dsn: str, *, cost: str, days_ago: int = 0, tokens: dict | None = None) -> None:
    """A rested run as the worker would have left it, carrying text no page may ever show."""
    agent: dict = {"untrusted": {"body": "<script>alert('pwned')</script>"}}
    if tokens is not None:
        agent["tokens_by_model"] = tokens
    with psycopg.connect(dsn) as connection:
        connection.execute(
            "INSERT INTO runs (id, channel, status, current_node, state, cost_usd, created_at)"
            " VALUES (gen_random_uuid(), 'email', 'done', 'rest', %s::jsonb, %s, %s)",
            (
                json.dumps({"agent": agent}),
                Decimal(cost),
                datetime.now(UTC) - timedelta(days=days_ago),
            ),
        )


def test_the_costs_page_is_linked_from_every_screen(fresh_database):
    assert 'href="/costs"' in client_for(fresh_database).get("/runs").text


def test_with_no_run_yet_the_page_says_so(fresh_database):
    page = client_for(fresh_database).get("/costs")

    assert page.status_code == 200
    assert "No runs yet" in page.text


def test_the_page_shows_what_a_run_cost_on_each_day(fresh_database):
    a_run(fresh_database, cost="0.002000", days_ago=1)
    a_run(fresh_database, cost="0.004000", days_ago=1)
    a_run(fresh_database, cost="0.001000")

    page = client_for(fresh_database).get("/costs").text

    assert "0.003000" in page, "yesterday's two runs averaged three thousandths of a cent each"
    assert "0.001000" in page


def test_the_page_shows_where_the_tokens_went(fresh_database):
    a_run(fresh_database, cost="0.001000", tokens={"llama3.1:8b": [3000, 200]})

    page = client_for(fresh_database).get("/costs").text

    assert "llama3.1:8b" in page
    assert "3000" in page and "200" in page


def test_the_page_says_what_the_cost_leaves_out(fresh_database):
    """The decision was to publish embeddings as uncounted, which obliges the page to admit it."""
    a_run(fresh_database, cost="0.001000")

    page = client_for(fresh_database).get("/costs").text.lower()

    assert "embedding" in page


def test_the_burn_down_never_asks_for_the_customer_s_text(fresh_database):
    """
    Not an escaping test, and it would pass with escaping switched off.

    What it pins is narrower and worth pinning on its own: the page's queries select `cost_usd` and
    `tokens_by_model` and nothing else, so the customer's words are never fetched and have no render
    path to be escaped on. If someone later widens a query to `SELECT state`, this fails.
    The escaping itself is held up by the model-name test below, which renders a real field.
    """
    a_run(fresh_database, cost="0.001000", tokens={"llama3.1:8b": [10, 2]})

    page = client_for(fresh_database).get("/costs").text

    assert "pwned" not in page
    assert "<script>" not in page


def test_a_model_name_from_the_state_column_cannot_write_the_page(fresh_database):
    """The names are keys in JSON the agent wrote, and they are rendered. They must be escaped."""
    a_run(fresh_database, cost="0.001000", tokens={"<script>alert(1)</script>": [10, 2]})

    page = client_for(fresh_database).get("/costs").text

    assert "<script>alert(1)</script>" not in page
