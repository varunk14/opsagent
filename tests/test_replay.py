"""
Replaying a run: a fresh run with the old message, and the original left exactly as it was.

The point of replay is to ask "would it go differently now?" -- after a prompt, a policy, or a
line of code changed. So the replay must be a genuine fresh start (the customer's message and
nothing the first run worked out), it must be a different run (its own id and key), it must be
threaded back to the original, and the original must not move by a single byte -- the whole value
is in comparing the two, and a comparison against something that was itself edited proves nothing.
"""

import psycopg
import pytest

from app.replay import replay
from tests.test_run_agent import happy_graph, ledger, queue, work_next

pytestmark = pytest.mark.db


def run_row(dsn: str, run_id) -> dict:
    with psycopg.connect(dsn) as connection:
        cursor = connection.execute(
            "SELECT id, channel, status, current_node, state, idempotency_key, replay_of, "
            "attempt, cost_usd, failure_class, prompt_version FROM runs WHERE id = %s",
            (run_id,),
        )
        names = [column.name for column in cursor.description]
        return dict(zip(names, cursor.fetchone(), strict=True))


def do_replay(dsn: str, original) -> str:
    with psycopg.connect(dsn) as connection:
        new_id = replay(connection, original)
    return str(new_id)


# --- the fresh run ------------------------------------------------------------


def test_a_replay_carries_the_same_message(fresh_database):
    original = queue(fresh_database)
    new_id = do_replay(fresh_database, original)

    old, new = run_row(fresh_database, original), run_row(fresh_database, new_id)
    assert new_id != original
    assert new["channel"] == old["channel"]
    assert new["state"]["untrusted"] == old["state"]["untrusted"]
    assert new["state"]["received_at"] == old["state"]["received_at"]


def test_a_replay_starts_from_nothing_the_first_run_worked_out(fresh_database):
    """A fresh start: only the customer's message, none of the original's steps or proposal."""
    ledger(fresh_database)
    original = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())  # the original refunds and rests, filling state.agent

    new_id = do_replay(fresh_database, original)
    new = run_row(fresh_database, new_id)
    assert "agent" not in new["state"]
    assert new["status"] == "queued"
    assert new["current_node"] == "intake"
    assert new["attempt"] == 0
    assert new["cost_usd"] == 0
    assert new["failure_class"] is None
    assert new["prompt_version"] is None


def test_a_replay_is_a_different_run_with_its_own_key(fresh_database):
    original = queue(fresh_database)
    old_key = run_row(fresh_database, original)["idempotency_key"]
    new_id = do_replay(fresh_database, original)
    new_key = run_row(fresh_database, new_id)["idempotency_key"]

    assert new_key != old_key
    assert new_key.startswith("replay:")


def test_a_replay_is_threaded_back_to_the_original(fresh_database):
    original = queue(fresh_database)
    new_id = do_replay(fresh_database, original)
    assert str(run_row(fresh_database, new_id)["replay_of"]) == original


# --- the original is untouched ------------------------------------------------


def test_replaying_does_not_move_the_original_by_a_byte(fresh_database):
    ledger(fresh_database)
    original = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    before = run_row(fresh_database, original)
    do_replay(fresh_database, original)
    after = run_row(fresh_database, original)
    assert before == after


# --- edges --------------------------------------------------------------------


def test_replaying_a_run_that_does_not_exist_is_refused(fresh_database):
    import uuid

    with psycopg.connect(fresh_database) as connection, pytest.raises(ValueError, match="no run"):
        replay(connection, uuid.uuid4())


def test_a_replay_can_itself_be_replayed(fresh_database):
    original = queue(fresh_database)
    first = do_replay(fresh_database, original)
    second = do_replay(fresh_database, first)

    assert str(run_row(fresh_database, second)["replay_of"]) == first
    assert run_row(fresh_database, second)["state"]["untrusted"] == run_row(fresh_database, original)["state"]["untrusted"]


# --- the worker can take it ---------------------------------------------------


def test_the_worker_processes_a_replay_through_the_same_guardrails(fresh_database):
    """
    A replay is a run like any other, so it meets every guardrail the original did. Here the
    original already refunded order 4821; replaying it and proposing the same refund is correctly
    held for a person rather than paid twice -- the duplicate-refund guard, reached through replay.
    """
    ledger(fresh_database)
    original = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())  # the original refunds 4821 and rests

    new_id = do_replay(fresh_database, original)
    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, happy_graph())  # only the replay is queued now

    assert str(outcome.run_id) == new_id
    assert outcome.status == "waiting_approval"
