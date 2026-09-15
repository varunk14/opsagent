"""
Dead letters: what cannot be processed is kept, explained, and can be sent back.

A run that runs out of attempts, or whose lock expires on its last attempt,
becomes dead -- and gets exactly one letter saying why, holding the message it
was about. An operator can list the open letters and requeue a run, which gives
it a fresh set of attempts. Requeueing is done once: asking twice changes
nothing, and a run that dies again after being requeued gets a new letter.

These tests commit, so each one gets its own scratch database.
"""

import psycopg
import pytest

from app.dead_letters import list_open, requeue
from app.llm import ModelUnavailable
from app.run_agent import claim_next, work_next
from tests.test_lock_expiry import expire
from tests.test_retry import Down
from tests.test_run_agent import graph_of, queue, row

pytestmark = pytest.mark.db


def exhaust(dsn: str, run_id: str) -> None:
    """Put the run on its last attempt and let the model be down for it."""
    with psycopg.connect(dsn) as connection:
        connection.execute("UPDATE runs SET attempt = max_attempts - 1 WHERE id = %s", (run_id,))
    with psycopg.connect(dsn) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph_of(Down()))


def letters(dsn: str, run_id: str) -> list[tuple]:
    with psycopg.connect(dsn) as connection:
        return connection.execute(
            "SELECT kind, idempotency_key, payload, reason, failure_class, requeued_at "
            "FROM dead_letters WHERE run_id = %s ORDER BY id",
            (run_id,),
        ).fetchall()


# --- how a run becomes a letter ---------------------------------------------------


def test_a_run_out_of_attempts_gets_one_letter_saying_why(fresh_database):
    run_id = queue(fresh_database)

    exhaust(fresh_database, run_id)

    [(kind, key, payload, reason, failure_class, requeued_at)] = letters(fresh_database, run_id)
    assert (kind, key, failure_class, requeued_at) == ("run", "email_msg_9f2a", "model_unavailable", None)
    assert "attempts" in reason
    assert payload["untrusted"]["body"].startswith("Hi, I think I was charged twice")


def test_a_run_that_has_only_failed_gets_no_letter(fresh_database):
    """A failed run is waiting to be tried again; nothing is dead yet."""
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph_of(Down()))

    assert row(fresh_database, run_id)["status"] == "failed"
    assert letters(fresh_database, run_id) == []


def test_a_run_whose_lock_expired_on_its_last_attempt_gets_a_letter(fresh_database):
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        claim_next(connection, worker="worker-a")
        connection.execute("UPDATE runs SET attempt = max_attempts WHERE id = %s", (run_id,))
    expire(fresh_database, run_id)

    with psycopg.connect(fresh_database) as connection:
        assert claim_next(connection, worker="worker-b") is None

    [(kind, _, _, reason, failure_class, _)] = letters(fresh_database, run_id)
    assert (kind, failure_class) == ("run", "lock_expired")
    assert "lock" in reason


# --- what an operator can do --------------------------------------------------------


def test_open_letters_are_listed_oldest_first(fresh_database):
    first = queue(fresh_database, "first", minute=1)
    exhaust(fresh_database, first)
    second = queue(fresh_database, "second", minute=2)
    exhaust(fresh_database, second)

    with psycopg.connect(fresh_database) as connection:
        listed = list_open(connection)

    assert [str(letter.run_id) for letter in listed] == [first, second]
    assert {letter.failure_class for letter in listed} == {"model_unavailable"}


def test_requeue_makes_a_dead_run_claimable_with_fresh_attempts(fresh_database):
    run_id = queue(fresh_database)
    exhaust(fresh_database, run_id)

    with psycopg.connect(fresh_database) as connection:
        assert requeue(connection, run_id) is True

    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["attempt"]) == ("queued", 0)
    with psycopg.connect(fresh_database) as connection:
        assert str(claim_next(connection, worker="worker-b").run_id) == run_id


def test_requeueing_twice_changes_nothing_the_second_time(fresh_database):
    run_id = queue(fresh_database)
    exhaust(fresh_database, run_id)
    with psycopg.connect(fresh_database) as connection:
        requeue(connection, run_id)
    with psycopg.connect(fresh_database) as connection:
        claim_next(connection, worker="worker-b")

    with psycopg.connect(fresh_database) as connection:
        assert requeue(connection, run_id) is False

    assert row(fresh_database, run_id)["status"] == "running", "the second requeue must not disturb the run"
    with psycopg.connect(fresh_database) as connection:
        assert list_open(connection) == []


def test_a_requeued_run_that_dies_again_gets_a_new_letter(fresh_database):
    run_id = queue(fresh_database)
    exhaust(fresh_database, run_id)
    with psycopg.connect(fresh_database) as connection:
        requeue(connection, run_id)

    exhaust(fresh_database, run_id)

    found = letters(fresh_database, run_id)
    assert len(found) == 2
    assert [requeued_at is None for (*_, requeued_at) in found] == [False, True]


def test_only_a_dead_run_can_be_requeued(fresh_database):
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        assert requeue(connection, run_id) is False

    assert row(fresh_database, run_id)["status"] == "queued"


def test_requeue_leaves_the_commit_to_its_caller(fresh_database):
    run_id = queue(fresh_database)
    exhaust(fresh_database, run_id)

    with psycopg.connect(fresh_database) as connection:
        requeue(connection, run_id)
        connection.rollback()

    assert row(fresh_database, run_id)["status"] == "dead"
    with psycopg.connect(fresh_database) as connection:
        assert len(list_open(connection)) == 1
