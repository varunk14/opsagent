"""
Acceptance test for durability: kill the worker mid-run, and it resumes.

A real process runs Priya's case and is killed by the operating system with
SIGKILL the moment its first step -- the order lookup -- has committed. Nothing
in that process gets to react. Another worker then finds the run once its lock
has expired and finishes it from where the database says it stopped: planning,
with the lookup's result, without looking the order up again.

The lock is aged in the database rather than waited out, so the test takes
seconds, not five minutes, and does not depend on timing.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import psycopg
import pytest

from app.run_agent import work_next
from tests.fakes import PROPOSED_REFUND, ScriptedModel
from tests.test_run_agent import count, graph_of, keys, ledger, queue, row
from tests.workers.crashing_worker import WORKER

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parent.parent


def run_doomed_worker(dsn: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tests.workers.crashing_worker"],
        cwd=ROOT,
        env={**os.environ, "OPSAGENT_CRASH_DSN": dsn},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def root_spans(dsn: str, run_id: str) -> list[tuple]:
    """Each tick recorded for the run: its trace, and what the tick ended as."""
    with psycopg.connect(dsn) as connection:
        return connection.execute(
            "SELECT trace_id, attributes ->> 'opsagent.outcome' FROM spans "
            "WHERE trace_id = %s AND parent_span_id IS NULL ORDER BY started_at",
            (run_id,),
        ).fetchall()


def test_a_worker_killed_mid_run_is_resumed_by_another(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    doomed = run_doomed_worker(fresh_database)

    assert doomed.returncode == -signal.SIGKILL, f"the worker was not killed: {doomed.stderr}"
    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["locked_by"]) == ("running", WORKER)
    assert len(stored["state"]["agent"]["steps"]) == 1
    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:get_order"]
    # The lookup's tick committed with its spans; nothing of the killed process's next tick exists.
    assert root_spans(fresh_database, run_id) == [(UUID(run_id), "running")]

    with psycopg.connect(fresh_database) as connection:
        connection.execute(
            "UPDATE runs SET locked_at = now() - interval '6 minutes' WHERE id = %s", (run_id,)
        )

    resumed = ScriptedModel(plan=PROPOSED_REFUND)
    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(resumed), worker="rescuer")

    assert str(outcome.run_id) == run_id
    assert resumed.tasks() == ["plan"], "classify and extract were already on record"
    # Rs 3,600 is under the default guardrail, so the rescuer pays it -- once.
    assert (outcome.status, outcome.tool) == ("done", "issue_refund")
    assert keys(fresh_database, run_id) == [
        f"{run_id}:step_1:get_order",
        f"{run_id}:step_2:issue_refund",
    ], "the lookup was not repeated"
    assert count(fresh_database, "refunds") == 1
    assert row(fresh_database, run_id)["attempt"] == 2
    # One trace across both processes: the killed worker's tick, then the rescuer's.
    assert root_spans(fresh_database, run_id) == [(UUID(run_id), "running"), (UUID(run_id), "done")]
