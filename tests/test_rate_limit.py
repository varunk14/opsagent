"""
Rate limiting, per sender, counted in Postgres.

Anyone can email in, as often as they like, quoting as many order numbers as
they like. Each lookup is a query against the ledger. So one sender gets at most
RATE_LIMIT lookups in any RATE_WINDOW; past that, their run is deferred -- put
back in the queue with a time to come back, executing nothing and spending no
attempt, because being busy is not a failure of the case.

Handing a case to a person is never limited. The count is taken under an
advisory lock on the sender, so two workers cannot both take the last slot.

These tests commit, so each one gets its own scratch database. The limit's
names are read from the module when each test runs, so a missing one fails
that test rather than the whole file.
"""

import threading
from uuid import uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

import app.run_agent as agent
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    EXTRACTED_4821,
    PROPOSED_ESCALATE,
    ScriptedModel,
)
from tests.test_run_agent import graph_of, happy_graph, keys, ledger, queue, row

pytestmark = pytest.mark.db

PRIYA = "priya@example.com"


def past_lookups(dsn: str, sender: str, count: int, age: str = "1 minute") -> None:
    """Lookups this sender already made, `age` ago, on runs of their own."""
    with psycopg.connect(dsn) as connection:
        for number in range(count):
            run_id = uuid4()
            connection.execute(
                "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
                "VALUES (%s, 'email', 'waiting_approval', 'plan', %s, %s)",
                (run_id, Jsonb({"untrusted": {"sender": sender}}), f"email_msg_past_{sender}_{number}"),
            )
            connection.execute(
                "INSERT INTO tool_calls (idempotency_key, run_id, tool, args, result, created_at) "
                "VALUES (%s, %s, 'get_order', '{}', '{}', now() - %s::interval)",
                (f"{run_id}:step_1:get_order", run_id, age),
            )


def test_a_sender_under_the_limit_is_served(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    past_lookups(fresh_database, PRIYA, agent.RATE_LIMIT - 1)

    with psycopg.connect(fresh_database) as connection:
        outcome = agent.work_next(connection, happy_graph())

    assert outcome.status == "waiting_approval"
    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:get_order"]


def test_a_sender_at_the_limit_is_deferred_without_spending_an_attempt(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    past_lookups(fresh_database, PRIYA, agent.RATE_LIMIT)

    with psycopg.connect(fresh_database) as connection:
        outcome = agent.work_next(connection, happy_graph())

    stored = row(fresh_database, run_id)
    assert (outcome.status, outcome.tool) == ("queued", "get_order")
    assert (stored["status"], stored["attempt"], stored["locked_by"]) == ("queued", 0, None)
    assert keys(fresh_database, run_id) == [], "nothing is executed while the sender is limited"


def test_a_deferred_run_comes_back_when_the_oldest_lookup_leaves_the_window(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    past_lookups(fresh_database, PRIYA, agent.RATE_LIMIT, age="10 minutes")

    with psycopg.connect(fresh_database) as connection:
        agent.work_next(connection, happy_graph())

    with psycopg.connect(fresh_database) as connection:
        wait = connection.execute(
            "SELECT next_retry_at - (SELECT min(created_at) FROM tool_calls) FROM runs WHERE id = %s", (run_id,)
        ).fetchone()[0]
    assert wait == agent.RATE_WINDOW


def test_a_deferred_run_is_not_claimed_before_it_is_due(fresh_database):
    ledger(fresh_database)
    queue(fresh_database)
    past_lookups(fresh_database, PRIYA, agent.RATE_LIMIT)
    with psycopg.connect(fresh_database) as connection:
        agent.work_next(connection, happy_graph())

    with psycopg.connect(fresh_database) as connection:
        assert agent.claim_next(connection, worker="worker-b") is None


def test_another_sender_is_not_limited(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    past_lookups(fresh_database, "dev@example.com", agent.RATE_LIMIT)

    with psycopg.connect(fresh_database) as connection:
        agent.work_next(connection, happy_graph())

    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:get_order"]


def test_lookups_older_than_the_window_do_not_count(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    past_lookups(fresh_database, PRIYA, agent.RATE_LIMIT, age="2 hours")

    with psycopg.connect(fresh_database) as connection:
        agent.work_next(connection, happy_graph())

    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:get_order"]


def test_handing_a_case_to_a_person_is_never_limited(fresh_database):
    run_id = queue(fresh_database)
    past_lookups(fresh_database, PRIYA, agent.RATE_LIMIT)
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_ESCALATE)

    with psycopg.connect(fresh_database) as connection:
        outcome = agent.work_next(connection, graph_of(model))

    assert (outcome.status, outcome.tool) == ("waiting_approval", "escalate_to_human")
    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:escalate_to_human"]


def test_two_workers_racing_for_the_last_slot_use_it_once(fresh_database):
    ledger(fresh_database)
    first = queue(fresh_database, "one", minute=1)
    second = queue(fresh_database, "two", minute=2)
    past_lookups(fresh_database, PRIYA, agent.RATE_LIMIT - 1)

    barrier = threading.Barrier(2)
    outcomes = []
    errors: list[BaseException] = []

    def worker(name: str) -> None:
        try:
            with psycopg.connect(fresh_database) as connection:
                barrier.wait()
                outcomes.append(agent.work_next(connection, happy_graph(), worker=name))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("worker-a", "worker-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    executed = len(keys(fresh_database, first)) + len(keys(fresh_database, second))
    assert executed == 1, "exactly one of the two runs got the last lookup"
    assert sorted(outcome.status for outcome in outcomes) == ["queued", "waiting_approval"]
