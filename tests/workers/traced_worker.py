"""
A real worker process, started the way `python -m app.run_agent` starts one.

Started by tests/test_worker_tracing.py, never by hand. It runs the driver's own
entry point against the database in OPSAGENT_DATABASE_URL, with a scripted model in
place of Ollama, so what tracing a worker installs -- and what it sends on exit -- is
exactly what the command line gets.
"""

import os
import sys

from app.graph.build import build_graph
from app.run_agent import run_worker
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    EXTRACTED_4821,
    PROPOSED_LOOKUP,
    PROPOSED_REFUND,
    FakeRetriever,
    ScriptedModel,
)


def main() -> int:
    model = ScriptedModel(
        classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, PROPOSED_REFUND]
    )
    return run_worker(build_graph(model, FakeRetriever()), limit=5, environ=os.environ)


if __name__ == "__main__":
    sys.exit(main())
