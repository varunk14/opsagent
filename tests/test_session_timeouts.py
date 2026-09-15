"""
Session timeouts: a worker that hangs inside a transaction cannot hold its run forever.

Found in Unit B's review. LOCK_TIMEOUT covers a worker that goes quiet between
steps. A worker whose connection hangs *inside* its act transaction still holds
the run's row lock, and FOR UPDATE SKIP LOCKED will skip that row for as long as
the session lives -- whatever the timeout says. So every worker session is given
an idle-in-transaction timeout and a statement timeout: Postgres itself ends a
session that sits in a transaction too long, the row lock goes with it, and lock
expiry can do its job.

The limits are read from the module when each test runs, so a missing one fails
that test rather than the whole file.
"""

import time
from datetime import timedelta

import psycopg
import pytest

import app.run_agent as agent
from tests.test_run_agent import queue

pytestmark = pytest.mark.db


def setting_ms(connection: psycopg.Connection, name: str) -> int:
    return int(connection.execute("SELECT setting FROM pg_settings WHERE name = %s", (name,)).fetchone()[0])


def test_a_worker_session_bounds_idle_transactions_and_statements(fresh_database):
    with psycopg.connect(fresh_database) as connection:
        agent.configure_session(connection)

        idle = setting_ms(connection, "idle_in_transaction_session_timeout")
        statement = setting_ms(connection, "statement_timeout")

    assert idle == agent.IDLE_IN_TRANSACTION_TIMEOUT // timedelta(milliseconds=1)
    assert statement == agent.STATEMENT_TIMEOUT // timedelta(milliseconds=1)
    assert idle > 0 and statement > 0


def test_work_next_configures_its_own_session(fresh_database):
    """The protection must not depend on every caller remembering to ask for it."""
    with psycopg.connect(fresh_database) as connection:
        assert agent.work_next(connection, graph=None) is None

        assert setting_ms(connection, "idle_in_transaction_session_timeout") > 0


def test_a_worker_hung_inside_a_transaction_gives_its_run_up(fresh_database):
    run_id = queue(fresh_database)
    hung = psycopg.connect(fresh_database)
    try:
        agent.configure_session(hung, idle_in_transaction=timedelta(seconds=1))
        agent.claim_next(hung, worker="worker-a")
        hung.execute("SELECT 1 FROM runs WHERE id = %s FOR UPDATE", (run_id,))  # as HOLD_CLAIM does
        time.sleep(2.5)  # the worker hangs; Postgres ends its session after 1 second idle

        with psycopg.connect(fresh_database) as connection:
            # If the hung session still held the row, this would wait on it forever;
            # a lock timeout turns that into a quick, readable failure.
            connection.execute("SET lock_timeout = '5s'")
            connection.execute(
                "UPDATE runs SET locked_at = now() - interval '10 minutes' WHERE id = %s", (run_id,)
            )
        with psycopg.connect(fresh_database) as connection:
            reclaimed = agent.claim_next(connection, worker="worker-b")

        assert reclaimed is not None, "the hung session's row lock must not outlive the timeout"
        assert str(reclaimed.run_id) == run_id
    finally:
        hung.close()
