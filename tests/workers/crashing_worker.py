"""
A real worker process that is killed right after its first step commits.

Started by tests/test_resume.py, never by hand. It runs the real driver against
the scratch database named in OPSAGENT_CRASH_DSN, and the moment its first
lookup has been committed it sends itself SIGKILL: no exception, no finally
block, no connection closed politely. Whatever survives is only what the
database already had.

The address is passed in the environment rather than on the command line, so
the throwaway password does not show in the process list.
"""

import os
import signal
import sys

import psycopg

from app.graph.build import build_graph
from app.run_agent import work_next
from app.tracing import Tracing
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    EXTRACTED_4821,
    PROPOSED_LOOKUP,
    PROPOSED_REFUND,
    FakeRetriever,
    ScriptedModel,
)

WORKER = "doomed-worker"


def die_now(run_id, steps: int) -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def main() -> int:
    model = ScriptedModel(
        classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, PROPOSED_REFUND]
    )
    # Traced like any worker, so the step that commits before the kill is recorded with its spans.
    Tracing().install()
    with psycopg.connect(os.environ["OPSAGENT_CRASH_DSN"]) as connection:
        work_next(connection, build_graph(model, FakeRetriever()), worker=WORKER, after_step=die_now)
    return 0  # reached only if the kill did not happen, which the test reports


if __name__ == "__main__":
    sys.exit(main())
