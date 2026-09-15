"""
A worker started from the command line traces what it does.

Tracing is installed per process, so these run a real worker process. Without
OPSAGENT_OTLP_ENDPOINT its spans are still recorded with its steps, which is what
the runs page reads. With it, a copy reaches Langfuse on this machine before the
worker exits. Told to send spans anywhere else, the worker refuses to start, and no
run is touched.
"""

import os
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest

from tests.test_run_agent import ledger, queue
from tests.test_tracing import Listener

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parent.parent
TRACING_VARS = ("OPSAGENT_OTLP_ENDPOINT", "OPSAGENT_LANGFUSE_PUBLIC_KEY", "OPSAGENT_LANGFUSE_SECRET_KEY")


def run_worker(dsn: str, **settings: str) -> subprocess.CompletedProcess:
    env = {name: value for name, value in os.environ.items() if name not in TRACING_VARS}
    return subprocess.run(
        [sys.executable, "-m", "tests.workers.traced_worker"],
        cwd=ROOT,
        env={**env, "OPSAGENT_DATABASE_URL": dsn, **settings},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def ticks(dsn: str, run_id: str) -> list[str]:
    with psycopg.connect(dsn) as connection:
        rows = connection.execute(
            "SELECT attributes ->> 'opsagent.outcome' FROM spans "
            "WHERE trace_id = %s AND parent_span_id IS NULL ORDER BY started_at",
            (run_id,),
        ).fetchall()
    return [outcome for (outcome,) in rows]


def status(dsn: str, run_id: str) -> str:
    with psycopg.connect(dsn) as connection:
        (value,) = connection.execute("SELECT status FROM runs WHERE id = %s", (run_id,)).fetchone()
    return value


def test_a_worker_records_each_runs_trace_with_its_steps(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    worker = run_worker(fresh_database)

    assert worker.returncode == 0, worker.stderr
    assert status(fresh_database, run_id) == "done"
    assert ticks(fresh_database, run_id) == ["running", "done"]


def test_a_worker_sends_its_trace_to_langfuse_on_this_machine_before_it_exits(fresh_database):
    ledger(fresh_database)
    queue(fresh_database)

    with Listener() as langfuse:
        worker = run_worker(
            fresh_database,
            OPSAGENT_OTLP_ENDPOINT=f"{langfuse.url}/api/public/otel",
            OPSAGENT_LANGFUSE_PUBLIC_KEY="pk-lf-local",
            OPSAGENT_LANGFUSE_SECRET_KEY="sk-lf-local",
        )

    assert worker.returncode == 0, worker.stderr
    assert "/api/public/otel/v1/traces" in langfuse.hits


def test_a_worker_told_to_send_traces_off_this_machine_does_not_start(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    worker = run_worker(
        fresh_database,
        OPSAGENT_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel",
        OPSAGENT_LANGFUSE_PUBLIC_KEY="pk-lf-local",
        OPSAGENT_LANGFUSE_SECRET_KEY="sk-lf-local",
    )

    assert worker.returncode == 2
    assert "this machine" in worker.stderr
    assert "Traceback" not in worker.stderr
    assert status(fresh_database, run_id) == "queued"
