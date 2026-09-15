"""
Taking a queued run through the graph and recording what it proposes.

Three transactions, deliberately separate. The claim is one statement --
UPDATE ... WHERE id = (SELECT ... FOR UPDATE SKIP LOCKED) -- so two workers can
never take the same run, and it commits at once. The graph then runs with no
transaction open, because model calls take seconds and a lock held that long
blocks everyone else. Finally the proposal is recorded in its own transaction.

Proposing only: the run ends waiting for a decision. A model that cannot be
reached puts the run back in the queue, since an outage is not a verdict on the
case -- until its attempts run out, when it is marked dead instead of sitting
at the head of the queue for good. A crash between claim and record leaves the
run marked running; week 4's lock expiry recovers that. Every write here checks
the worker still holds the run, so a slow worker cannot overwrite a reclaim.

Run:  .venv/bin/python -m app.run_agent [how many runs, default 10]
"""

import os
import socket
import sys
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb

from app.baseline import REFERENCE_RATE, token_cost
from app.db import connect
from app.graph.build import build_graph, run_graph
from app.graph.state import AgentState
from app.llm import ModelUnavailable, Ollama

MICRO_DOLLAR = Decimal("0.000001")  # matches runs.cost_usd numeric(10, 6)

CLAIM = """
    UPDATE runs
       SET status = 'running', current_node = 'classify', attempt = attempt + 1,
           locked_by = %s, locked_at = now()
     WHERE id = (
            SELECT id FROM runs
             WHERE status = 'queued'
             ORDER BY created_at, id
               FOR UPDATE SKIP LOCKED
             LIMIT 1
           )
    RETURNING id, state
"""

RECORD = """
    UPDATE runs
       SET status = 'waiting_approval', current_node = %s, state = state || %s,
           cost_usd = cost_usd + %s, locked_by = NULL, locked_at = NULL
     WHERE id = %s AND status = 'running' AND locked_by = %s
"""

# One statement decides requeue or dead, so nothing can change attempt in between.
RELEASE = """
    UPDATE runs
       SET status = CASE WHEN attempt >= max_attempts THEN 'dead' ELSE 'queued' END,
           failure_class = CASE WHEN attempt >= max_attempts
                                THEN 'model_unavailable' ELSE failure_class END,
           current_node = 'intake', locked_by = NULL, locked_at = NULL
     WHERE id = %s AND status = 'running' AND locked_by = %s
"""


class LostClaim(RuntimeError):
    """The run was taken by another worker before this one could record its result."""


@dataclass(frozen=True)
class ClaimedRun:
    run_id: UUID
    subject: str | None
    body: str
    worker: str


@dataclass(frozen=True)
class ProposalOutcome:
    run_id: UUID
    tool: str
    failure: str | None
    cost_usd: Decimal


def default_worker() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def require_idle(connection: psycopg.Connection) -> None:
    if connection.pgconn.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError(
            "the driver commits, so it needs its own transaction: call it on a "
            "connection with no work already open"
        )


def claim_next(connection: psycopg.Connection, worker: str) -> ClaimedRun | None:
    """Take the oldest queued run nobody else holds, or None if there is none."""
    require_idle(connection)
    with connection.transaction():
        claimed = connection.execute(CLAIM, (worker,)).fetchone()
    if claimed is None:
        return None

    run_id, state = claimed
    untrusted = state.get("untrusted", {})
    return ClaimedRun(
        run_id=run_id, subject=untrusted.get("subject"), body=untrusted.get("body", ""), worker=worker
    )


def summarise_agent(state: AgentState) -> tuple[dict[str, Any], Decimal]:
    """What gets stored on the run, and what the run cost."""
    replies = state.get("replies", [])
    prompt_tokens = sum(reply.prompt_tokens for reply in replies)
    completion_tokens = sum(reply.completion_tokens for reply in replies)
    cost = token_cost(prompt_tokens, completion_tokens, REFERENCE_RATE).quantize(MICRO_DOLLAR)

    classification = state.get("classification")
    extraction = state.get("extraction")
    agent = {
        "classification": classification.model_dump(mode="json") if classification else None,
        "extraction": extraction.model_dump(mode="json") if extraction else None,
        "policy": state.get("policy", []),
        "proposal": state["proposal"].model_dump(mode="json"),
        "failure": state.get("failure"),
        "model_calls": len(replies),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "rate": REFERENCE_RATE.name,
    }
    return agent, cost


def propose_next(
    connection: psycopg.Connection, graph: Any, worker: str | None = None
) -> ProposalOutcome | None:
    """Claim one run, walk it through the graph, record the proposal."""
    claimed = claim_next(connection, worker or default_worker())
    if claimed is None:
        return None

    try:
        state = run_graph(graph, claimed.subject, claimed.body)
    except ModelUnavailable:
        with connection.transaction():
            connection.execute(RELEASE, (claimed.run_id, claimed.worker))
        raise

    agent, cost = summarise_agent(state)
    failure = state.get("failure")
    # A run that escalated early stopped at the step that failed.
    node = failure.split(":", 1)[0] if failure else "plan"
    with connection.transaction():
        recorded = connection.execute(
            RECORD, (node, Jsonb({"agent": agent}), cost, claimed.run_id, claimed.worker)
        )
        if recorded.rowcount == 0:
            raise LostClaim(f"run {claimed.run_id} was reclaimed before its proposal was recorded")

    return ProposalOutcome(
        run_id=claimed.run_id, tool=state["proposal"].tool, failure=failure, cost_usd=cost
    )


def main(argv: list[str]) -> int:  # pragma: no cover - the interactive driver
    limit = int(argv[1]) if len(argv) > 1 else 10
    graph = build_graph(Ollama())

    with connect() as connection:
        for _ in range(limit):
            try:
                outcome = propose_next(connection, graph)
            except ModelUnavailable as exc:
                print(f"  model unavailable, run returned to the queue: {exc}")
                return 1
            if outcome is None:
                print("  queue empty")
                break
            note = f"  ESCALATED ({outcome.failure})" if outcome.failure else ""
            print(f"  {outcome.run_id}  proposes {outcome.tool:<18} ${outcome.cost_usd}{note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
