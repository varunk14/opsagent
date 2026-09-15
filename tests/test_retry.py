"""
Retry with backoff: an outage schedules the run for later, not the front of the queue.

Before this, a run whose model was down went straight back to `queued` and the
next claim picked it up at once -- a tight loop against a service that is down.
Now it waits in `failed` until `next_retry_at`: thirty seconds, then twice as
long each time, never more than an hour, spread by jitter so that many runs
failing together do not all come back in the same second.
"""

from datetime import timedelta

import psycopg
import pytest

from app.llm import ModelUnavailable, Reply
from app.run_agent import claim_next, retry_delay, work_next
from tests.test_run_agent import graph_of, queue, row


class Down:
    def generate(self, prompt: str) -> Reply:
        raise ModelUnavailable("connection refused")


def full_delay() -> float:
    return 1.0


def outage(dsn: str) -> None:
    with psycopg.connect(dsn) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph_of(Down()))


def retry_row(dsn: str, run_id: str) -> tuple:
    with psycopg.connect(dsn) as connection:
        return connection.execute(
            "SELECT status, failure_class, next_retry_at, next_retry_at - now() FROM runs WHERE id = %s",
            (run_id,),
        ).fetchone()


# --- the schedule ---------------------------------------------------------------


def test_the_first_retry_waits_thirty_seconds():
    assert retry_delay(1, jitter=full_delay) == timedelta(seconds=30)


def test_each_retry_waits_twice_as_long_as_the_last():
    delays = [retry_delay(attempt, jitter=full_delay) for attempt in (1, 2, 3, 4)]

    assert delays == [timedelta(seconds=seconds) for seconds in (30, 60, 120, 240)]


def test_no_retry_waits_longer_than_an_hour():
    assert retry_delay(30, jitter=full_delay) == timedelta(hours=1)


def test_jitter_spreads_a_retry_between_half_and_all_of_its_delay():
    """Many runs failing together must not all come back in the same second."""
    assert retry_delay(3, jitter=lambda: 0.0) == timedelta(seconds=60)
    assert retry_delay(3, jitter=lambda: 0.5) == timedelta(seconds=90)


# --- what an outage does to the run -----------------------------------------------


@pytest.mark.db
def test_an_outage_marks_the_run_failed_with_a_time_to_try_again(fresh_database):
    run_id = queue(fresh_database)

    outage(fresh_database)

    status, failure_class, _, wait = retry_row(fresh_database, run_id)
    assert (status, failure_class) == ("failed", "model_unavailable")
    assert timedelta(seconds=14) < wait <= timedelta(seconds=30), "first attempt: 15 to 30 seconds"


@pytest.mark.db
def test_a_failed_run_is_not_claimed_before_it_is_due(fresh_database):
    queue(fresh_database)
    outage(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        assert claim_next(connection, worker="worker-b") is None


@pytest.mark.db
def test_a_failed_run_is_claimed_once_it_is_due(fresh_database):
    run_id = queue(fresh_database)
    outage(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET next_retry_at = now() - interval '1 second' WHERE id = %s", (run_id,))

    with psycopg.connect(fresh_database) as connection:
        claimed = claim_next(connection, worker="worker-b")

    stored = row(fresh_database, run_id)
    assert str(claimed.run_id) == run_id
    assert (stored["status"], stored["locked_by"], stored["attempt"]) == ("running", "worker-b", 2)


@pytest.mark.db
def test_a_run_out_of_attempts_is_dead_and_not_scheduled(fresh_database):
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET attempt = max_attempts - 1 WHERE id = %s", (run_id,))

    outage(fresh_database)

    status, failure_class, next_retry_at, _ = retry_row(fresh_database, run_id)
    assert (status, failure_class, next_retry_at) == ("dead", "model_unavailable", None)
