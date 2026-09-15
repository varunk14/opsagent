"""
Golden cases through the real system.

Each case's message is accepted by the real intake into the database given, with the eval
ledger and the policies loaded, and worked by the real driver until every run rests. What
the model says comes from whichever Model is passed -- a live one while recording, a
RecordedModel on replay -- and nothing else is stood in for: the executor, the guardrail,
ownership and the refund cap are the ones production uses.

A CaseResult is what the run left behind, read from the database: where it came to rest,
what it understood, which tools it ran, what was paid and what was put to a person. It
holds nothing that differs between two runs of the same recording, so replays compare equal.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg

from app.db import apply_migrations, connect
from app.embeddings import Embedder
from app.graph.build import build_graph
from app.intake import accept
from app.llm import Model
from app.policies import POLICY_DIR, ingest, load_policies
from app.retrieval import PolicyRetriever
from app.run_agent import work_next
from app.seed import load_ledger
from evals.golden import LEDGER, GoldenCase

RUN = "SELECT status, cost_usd, state -> 'agent' FROM runs WHERE id = %s"
REFUNDS = "SELECT amount_paise FROM refunds WHERE run_id = %s ORDER BY id"
LAST_APPROVAL = """
    SELECT (action -> 'args' ->> 'amount_paise')::bigint
      FROM approvals
     WHERE run_id = %s
     ORDER BY id DESC
     LIMIT 1
"""


@dataclass(frozen=True)
class CaseResult:
    """What one case's run left behind."""

    case_id: str
    status: str
    failure: str | None
    intent: str | None
    order_id: str | None
    stated_amount_paise: int | None
    tools: tuple[str, ...]
    refunds_paise: tuple[int, ...]
    approval_paise: int | None
    model_calls: int
    cost_usd: Decimal


def run_cases(
    dsn: str, cases: Sequence[GoldenCase], model: Model, embedder: Embedder, ledger: Path = LEDGER
) -> list[CaseResult]:
    """Accept every case, work every run until it rests, and read back each case's result in order."""
    with psycopg.connect(dsn) as connection:
        apply_migrations(connection)
    with psycopg.connect(dsn) as connection:
        ingest(connection, embedder, load_policies(POLICY_DIR))
    with psycopg.connect(dsn) as connection:
        load_ledger(connection, ledger)
    with psycopg.connect(dsn) as connection:
        runs = {case.id: accept(connection, case.message).run_id for case in cases}

    graph = build_graph(model, PolicyRetriever(lambda: connect(dsn), embedder))
    with psycopg.connect(dsn) as connection:
        while work_next(connection, graph) is not None:
            pass
        return [read_back(connection, case.id, runs[case.id]) for case in cases]


def read_back(connection: psycopg.Connection, case_id: str, run_id: UUID) -> CaseResult:
    row = connection.execute(RUN, (run_id,)).fetchone()
    if row is None:
        raise LookupError(f"case {case_id}: its run {run_id} is not in the database")
    status, cost_usd, stored = row
    agent: dict[str, Any] = stored or {}
    classification = agent.get("classification") or {}
    extraction = agent.get("extraction") or {}
    approval = connection.execute(LAST_APPROVAL, (run_id,)).fetchone()
    return CaseResult(
        case_id=case_id,
        status=status,
        failure=agent.get("failure"),
        intent=classification.get("intent"),
        order_id=extraction.get("order_id"),
        stated_amount_paise=extraction.get("amount_paise"),
        tools=tuple(step["tool"] for step in agent.get("steps", [])),
        refunds_paise=tuple(amount for (amount,) in connection.execute(REFUNDS, (run_id,)).fetchall()),
        approval_paise=approval[0] if approval else None,
        model_calls=int(agent.get("model_calls", 0)),
        cost_usd=cost_usd,
    )
