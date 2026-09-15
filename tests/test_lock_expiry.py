"""
Lock expiry: a run whose worker died is picked up by another.

The handbook's picture (2.7): a worker claims a run, executes one step, commits,
and dies. Its lock expires after five minutes, another worker claims the run and
continues from the step that was committed. A worker that is merely slow keeps
its run, because every committed step refreshes the lock; one that finds its
claim taken executes nothing.

These tests commit, so each one gets its own scratch database.
"""

from datetime import timedelta

import psycopg
import pytest

from app.run_agent import LOCK_TIMEOUT, LostClaim, claim_next, tick, work_next
from tests.fakes import PROPOSED_REFUND, ScriptedModel
from tests.test_run_agent import graph_of, happy_graph, keys, ledger, queue, row

pytestmark = pytest.mark.db

FIVE_MINUTES = timedelta(minutes=5)


class WorkerDied(BaseException):
    """Stands in for SIGKILL in-process: nothing after it runs, nothing uncommitted survives."""


def die(run_id, steps: int) -> None:
    raise WorkerDied


def expire(dsn: str, run_id: str) -> None:
    """Age the run's lock past the timeout, as ten minutes of silence would."""
    with psycopg.connect(dsn) as connection:
        connection.execute(
            "UPDATE runs SET locked_at = now() - interval '10 minutes' WHERE id = %s", (run_id,)
        )


def test_the_lock_timeout_is_five_minutes():
    """The handbook's figure: long enough for slow model calls, short enough for a customer."""
    assert timedelta(minutes=5) == LOCK_TIMEOUT


def test_a_fresh_lock_is_not_taken(fresh_database):
    queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        assert claim_next(connection, worker="worker-a") is not None

    with psycopg.connect(fresh_database) as connection:
        assert claim_next(connection, worker="worker-b", lock_timeout=FIVE_MINUTES) is None


def test_an_expired_lock_is_reclaimed_by_another_worker(fresh_database):
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        claim_next(connection, worker="worker-a")
    expire(fresh_database, run_id)

    with psycopg.connect(fresh_database) as connection:
        reclaimed = claim_next(connection, worker="worker-b", lock_timeout=FIVE_MINUTES)

    stored = row(fresh_database, run_id)
    assert str(reclaimed.run_id) == run_id
    assert (stored["status"], stored["locked_by"], stored["attempt"]) == ("running", "worker-b", 2)


def test_the_worker_whose_lock_was_taken_executes_nothing(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        stale = claim_next(connection, worker="worker-a")
    expire(fresh_database, run_id)
    with psycopg.connect(fresh_database) as connection:
        claim_next(connection, worker="worker-b", lock_timeout=FIVE_MINUTES)

    with psycopg.connect(fresh_database) as connection, pytest.raises(LostClaim):
        tick(connection, happy_graph(), stale, max_steps=4)

    assert keys(fresh_database, run_id) == []
    assert row(fresh_database, run_id)["locked_by"] == "worker-b"


def test_a_dead_workers_run_continues_from_its_last_committed_step(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection, pytest.raises(WorkerDied):
        work_next(connection, happy_graph(), worker="worker-a", after_step=die)

    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["locked_by"]) == ("running", "worker-a")
    assert len(stored["state"]["agent"]["steps"]) == 1

    expire(fresh_database, run_id)
    resumed = ScriptedModel(plan=PROPOSED_REFUND)
    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(resumed), worker="worker-b", lock_timeout=FIVE_MINUTES)

    assert resumed.tasks() == ["plan"], "classify and extract were already on record"
    assert (outcome.status, outcome.tool) == ("done", "issue_refund")
    assert keys(fresh_database, run_id) == [
        f"{run_id}:step_1:get_order",
        f"{run_id}:step_2:issue_refund",
    ], "the lookup is not repeated"
    assert row(fresh_database, run_id)["attempt"] == 2


def test_an_expired_lock_on_the_last_attempt_is_dead_not_reclaimed(fresh_database):
    """A run that kills every worker that touches it must stop being handed out."""
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        claim_next(connection, worker="worker-a")
        connection.execute("UPDATE runs SET attempt = max_attempts WHERE id = %s", (run_id,))
    expire(fresh_database, run_id)

    with psycopg.connect(fresh_database) as connection:
        assert claim_next(connection, worker="worker-b", lock_timeout=FIVE_MINUTES) is None

    with psycopg.connect(fresh_database) as connection:
        status, failure_class, locked_by = connection.execute(
            "SELECT status, failure_class, locked_by FROM runs WHERE id = %s", (run_id,)
        ).fetchone()
    assert (status, failure_class, locked_by) == ("dead", "lock_expired", None)


def test_every_committed_step_refreshes_the_lock(fresh_database):
    """A slow run that keeps committing steps is working, not stuck."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        claimed = claim_next(connection, worker="worker-a")
    expire(fresh_database, run_id)
    aged = row(fresh_database, run_id)["locked_at"]

    with psycopg.connect(fresh_database) as connection:
        outcome = tick(connection, happy_graph(), claimed, max_steps=4)

    assert outcome.status == "running"
    assert row(fresh_database, run_id)["locked_at"] > aged + timedelta(minutes=9)
