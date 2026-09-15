"""
Working a queued run one committed step at a time.

A run is claimed once, then worked in ticks. Each tick reads what the run already
holds, walks the graph from there, and acts on the single proposal that comes
back. The transactions are kept deliberately separate and short:

  claim  UPDATE ... WHERE id = (SELECT ... FOR UPDATE SKIP LOCKED), committed at
         once, so two workers can never take the same run.
  graph  no transaction open, because model calls take seconds and a lock held
         that long blocks everyone else.
  act    one transaction: lock the run row and confirm this worker still holds
         it, execute the tool through the keyed executor, append the step, add
         the cost. A worker that lost its claim executes nothing.

get_order runs and the run goes round again, its lock refreshed so a run making
progress is never mistaken for a stuck one. escalate_to_human runs and the run
waits for a person. issue_refund is recorded and the run waits for a person
too: approval is week 5, and the sender is only a From header. A proposal that
repeats an earlier step, a run past its step budget, or a tool this worker does
not run are all handed to a person instead of executed.

A model or policy store that cannot be reached marks the run failed, charged for
the calls that completed and keeping every committed step, and schedules it for
another attempt after a backoff delay -- until its attempts run out and it is
marked dead. A crash leaves the run marked
running with its committed steps intact. Once its lock is older than
LOCK_TIMEOUT another worker reclaims it, spending an attempt, and carries on
from the last committed step; on its last attempt it is marked dead instead.
The timeout covers a worker that has gone quiet between steps. One whose
connection hangs inside the act transaction keeps its row lock, and SKIP LOCKED
will not take the run until Postgres ends that session -- session timeouts to
bound that are planned.

Run:  .venv/bin/python -m app.run_agent [how many runs, default 10]
"""

import os
import random
import socket
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb

from app.baseline import REFERENCE_RATE, token_cost
from app.contracts import Classification, ExtractedRefund, ProposedAction, StepRecord
from app.db import apply_migrations, connect
from app.embeddings import OllamaEmbedder
from app.executor import execute
from app.graph.build import build_graph, run_graph
from app.graph.nodes import escalation
from app.graph.state import AgentState
from app.llm import Ollama, Reply, ServiceUnavailable
from app.retrieval import PolicyRetriever

MICRO_DOLLAR = Decimal("0.000001")  # matches runs.cost_usd numeric(10, 6)

# Executed tools a run may use before a person takes over. Small models can wander.
MAX_STEPS = 4

# A failed run waits RETRY_BASE after its first failure, doubling each time, never
# longer than RETRY_CAP. The exponent is capped too, so no attempt count overflows.
RETRY_BASE = timedelta(seconds=30)
RETRY_CAP = timedelta(hours=1)
MAX_DOUBLINGS = 20

# Every tool the model can propose is in exactly one of these, and a test holds it
# there, so a new tool in app/tools.py cannot run -- or fail to run -- by default.
RUNS_NOW = frozenset({"get_order", "escalate_to_human"})
WAITS_FOR_APPROVAL = frozenset({"issue_refund"})
# Retrieval already ran in the graph; a planner asking to search again is handed over.
NOT_RUN_HERE = frozenset({"search_policy"})

# How long a lock may go without a committed step before the run is taken back.
# Every committed step refreshes it, so only a worker that has gone quiet loses it.
LOCK_TIMEOUT = timedelta(minutes=5)

# A run whose worker went quiet on its last attempt is not handed out again.
BURY_EXPIRED = """
    UPDATE runs
       SET status = 'dead', failure_class = 'lock_expired', locked_by = NULL, locked_at = NULL
     WHERE status = 'running' AND locked_at < now() - %s AND attempt >= max_attempts
"""

CLAIM = """
    UPDATE runs
       SET status = 'running', current_node = 'classify', attempt = attempt + 1,
           locked_by = %s, locked_at = now()
     WHERE id = (
            SELECT id FROM runs
             WHERE status = 'queued'
                OR (status = 'failed' AND next_retry_at <= now())
                OR (status = 'running' AND locked_at < now() - %s AND attempt < max_attempts)
             ORDER BY created_at, id
               FOR UPDATE SKIP LOCKED
             LIMIT 1
           )
    RETURNING id, state, attempt
"""

# Taken at the start of every act transaction. Holding the row until commit means
# nothing can reclaim the run between the check and the tool call.
HOLD_CLAIM = """
    SELECT 1 FROM runs
     WHERE id = %s AND status = 'running' AND locked_by = %s
       FOR UPDATE
"""

CONTINUE = """
    UPDATE runs
       SET current_node = 'act', state = state || %s, cost_usd = cost_usd + %s, locked_at = now()
     WHERE id = %s AND locked_by = %s
"""

PARK = """
    UPDATE runs
       SET status = 'waiting_approval', current_node = %s, state = state || %s,
           cost_usd = cost_usd + %s, locked_by = NULL, locked_at = NULL
     WHERE id = %s AND locked_by = %s
"""

# One statement decides retry or dead, so nothing can change attempt in between.
RELEASE = """
    UPDATE runs
       SET status = CASE WHEN attempt >= max_attempts THEN 'dead' ELSE 'failed' END,
           failure_class = %s,
           next_retry_at = CASE WHEN attempt >= max_attempts THEN NULL ELSE now() + %s END,
           current_node = 'intake', locked_by = NULL, locked_at = NULL,
           cost_usd = cost_usd + %s
     WHERE id = %s AND status = 'running' AND locked_by = %s
"""


class LostClaim(RuntimeError):
    """The run was taken by another worker before this one could act on it."""


@dataclass(frozen=True)
class ClaimedRun:
    run_id: UUID
    subject: str | None
    body: str
    worker: str
    attempt: int


@dataclass(frozen=True)
class RunOutcome:
    """Where one tick left the run. `running` means the worker goes round again."""

    run_id: UUID
    status: str
    tool: str
    steps: int
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


def claim_next(
    connection: psycopg.Connection, worker: str, lock_timeout: timedelta = LOCK_TIMEOUT
) -> ClaimedRun | None:
    """Take the oldest run that is queued, or whose worker went quiet; None if there is none."""
    require_idle(connection)
    with connection.transaction():
        connection.execute(BURY_EXPIRED, (lock_timeout,))
        claimed = connection.execute(CLAIM, (worker, lock_timeout)).fetchone()
    if claimed is None:
        return None

    run_id, state, attempt = claimed
    untrusted = state.get("untrusted", {})
    return ClaimedRun(
        run_id=run_id,
        subject=untrusted.get("subject"),
        body=untrusted.get("body", ""),
        worker=worker,
        attempt=attempt,
    )


def retry_delay(attempt: int, jitter: Callable[[], float] = random.random) -> timedelta:
    """
    How long a run that failed on `attempt` waits before it may be claimed again.

    Thirty seconds after the first failure, twice as long after each one since,
    never more than an hour. Jitter then places it between half and all of that,
    so runs that failed together in one outage do not all return in the same second.
    """
    delay = min(RETRY_CAP, RETRY_BASE * 2 ** min(attempt - 1, MAX_DOUBLINGS))
    return delay * (0.5 + jitter() / 2)


def cost_of(replies: list[Reply]) -> Decimal:
    """Every reply priced once, at the baseline rate, to the column's precision."""
    prompt_tokens = sum(reply.prompt_tokens for reply in replies)
    completion_tokens = sum(reply.completion_tokens for reply in replies)
    return token_cost(prompt_tokens, completion_tokens, REFERENCE_RATE).quantize(MICRO_DOLLAR)


def prior_from(agent: Mapping[str, Any]) -> AgentState:
    """What earlier ticks of this run found, rebuilt from its row. Empty before the first."""
    prior: AgentState = {}
    if agent.get("classification") is None:
        return prior
    extraction = agent.get("extraction")
    prior["classification"] = Classification.model_validate(agent["classification"])
    prior["extraction"] = ExtractedRefund.model_validate(extraction) if extraction is not None else None
    prior["policy"] = list(agent.get("policy", []))
    prior["policy_sources"] = list(agent.get("policy_sources", []))
    prior["observations"] = [dict(step) for step in agent.get("steps", [])]
    return prior


def hand_over(why: str, reason: str) -> tuple[ProposedAction, str]:
    """A driver-made escalation, shaped exactly like one a graph step makes."""
    escalated = escalation("plan", reason, why, [])
    return escalated["proposal"], escalated["failure"]


def decide(state: AgentState, steps: list[dict[str, Any]], max_steps: int) -> tuple[ProposedAction, str | None]:
    """The action this tick takes: the proposal, or a hand-over to a person in its place."""
    proposal = state["proposal"]
    failure = state.get("failure")
    if failure is not None or proposal.tool == "escalate_to_human":
        return proposal, failure

    if proposal.tool not in RUNS_NOW | WAITS_FOR_APPROVAL:
        return hand_over(
            f"{proposal.tool} is not a tool this worker runs",
            f"The planner asked for {proposal.tool}, which does not run here.",
        )
    if any(step["tool"] == proposal.tool and step["args"] == dict(proposal.args) for step in steps):
        return hand_over("repeated an earlier step", f"The planner asked for {proposal.tool} again with the same arguments.")
    if proposal.tool in RUNS_NOW and len(steps) >= max_steps:
        return hand_over(f"step budget of {max_steps} used", f"The run used all {max_steps} of its steps without deciding.")
    return proposal, None


def charge(before: Mapping[str, Any], replies: list[Reply]) -> tuple[Decimal, int, int]:
    """
    What this tick adds to the run's cost, and the run's token totals after it.

    Charged as the difference between the rounded totals, so the stored cost is
    always the whole run's tokens priced once, however many ticks it took.
    """
    prompt_before = before.get("prompt_tokens", 0)
    completion_before = before.get("completion_tokens", 0)
    prompt_after = prompt_before + sum(reply.prompt_tokens for reply in replies)
    completion_after = completion_before + sum(reply.completion_tokens for reply in replies)
    delta = token_cost(prompt_after, completion_after, REFERENCE_RATE).quantize(MICRO_DOLLAR) - token_cost(
        prompt_before, completion_before, REFERENCE_RATE
    ).quantize(MICRO_DOLLAR)
    return delta, prompt_after, completion_after


def summarise_agent(
    state: AgentState,
    before: Mapping[str, Any],
    proposal: ProposedAction,
    failure: str | None,
    steps: list[dict[str, Any]],
) -> tuple[dict[str, Any], Decimal]:
    """What gets stored on the run after this tick, and what the tick cost."""
    replies = state.get("replies", [])
    cost, prompt_tokens, completion_tokens = charge(before, replies)
    classification = state.get("classification")
    extraction = state.get("extraction")
    agent = {
        "classification": classification.model_dump(mode="json") if classification else None,
        "extraction": extraction.model_dump(mode="json") if extraction else None,
        "policy": state.get("policy", []),
        "policy_sources": state.get("policy_sources", []),
        "proposal": proposal.model_dump(mode="json"),
        "failure": failure,
        "steps": steps,
        "model_calls": before.get("model_calls", 0) + len(replies),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "rate": REFERENCE_RATE.name,
    }
    return agent, cost


def agent_of(connection: psycopg.Connection, run_id: UUID) -> dict[str, Any]:
    with connection.transaction():
        row = connection.execute("SELECT state -> 'agent' FROM runs WHERE id = %s", (run_id,)).fetchone()
    return dict(row[0]) if row and row[0] else {}


def tick(connection: psycopg.Connection, graph: Any, claimed: ClaimedRun, max_steps: int) -> RunOutcome:
    """Walk the graph from what the run holds, then act on one proposal in one transaction."""
    before = agent_of(connection, claimed.run_id)
    try:
        state = run_graph(graph, claimed.subject, claimed.body, prior_from(before))
    except ServiceUnavailable as outage:
        # Failed, to be tried again later: charged for the calls that did complete, steps kept.
        with connection.transaction():
            connection.execute(
                RELEASE,
                (
                    outage.failure_class,
                    retry_delay(claimed.attempt),
                    cost_of(outage.replies),
                    claimed.run_id,
                    claimed.worker,
                ),
            )
        raise

    steps = list(before.get("steps", []))
    proposal, failure = decide(state, steps, max_steps)
    with connection.transaction():
        if connection.execute(HOLD_CLAIM, (claimed.run_id, claimed.worker)).fetchone() is None:
            raise LostClaim(f"run {claimed.run_id} was reclaimed before this worker could act on it")

        if proposal.tool in RUNS_NOW:
            number = len(steps) + 1
            done = execute(connection, claimed.run_id, number, proposal)
            record = StepRecord(
                step=number, tool=proposal.tool, args=dict(proposal.args), result=done.result, replayed=done.replayed
            )
            steps.append(record.model_dump(mode="json"))

        agent, cost = summarise_agent(state, before, proposal, failure, steps)
        if proposal.tool == "get_order":
            status = "running"
            connection.execute(CONTINUE, (Jsonb({"agent": agent}), cost, claimed.run_id, claimed.worker))
        else:
            status = "waiting_approval"
            # A run that escalated early stopped at the step that failed.
            node = failure.split(":", 1)[0] if failure else "plan"
            connection.execute(PARK, (node, Jsonb({"agent": agent}), cost, claimed.run_id, claimed.worker))

    return RunOutcome(
        run_id=claimed.run_id, status=status, tool=proposal.tool, steps=len(steps), failure=failure, cost_usd=cost
    )


def work_next(
    connection: psycopg.Connection,
    graph: Any,
    worker: str | None = None,
    max_steps: int = MAX_STEPS,
    after_step: Callable[[UUID, int], None] | None = None,
    lock_timeout: timedelta = LOCK_TIMEOUT,
) -> RunOutcome | None:
    """
    Claim one run and work it until it waits for a person.

    `after_step` is called once each continuing step has been committed, with the
    run id and how many steps it now has -- the moment a dying worker loses nothing.
    """
    claimed = claim_next(connection, worker or default_worker(), lock_timeout)
    if claimed is None:
        return None

    while True:
        outcome = tick(connection, graph, claimed, max_steps)
        if outcome.status != "running":
            return outcome
        if after_step is not None:
            after_step(claimed.run_id, outcome.steps)


def prepare_database(connection: psycopg.Connection) -> int:
    """
    Bring the schema up to date and report how many policy passages are loaded.

    Migrations are safe to run on every start-up, and running them here means a
    deploy that restarts only the driver cannot leave it querying columns that
    do not exist yet. Zero passages means app.policies has not been run.
    """
    apply_migrations(connection)
    with connection.transaction():
        row = connection.execute("SELECT count(*) FROM policy_chunks").fetchone()
    return int(row[0]) if row else 0


def main(argv: list[str]) -> int:  # pragma: no cover - the interactive driver
    limit = int(argv[1]) if len(argv) > 1 else 10
    graph = build_graph(Ollama(), PolicyRetriever(connect, OllamaEmbedder()))

    with connect() as connection:
        if prepare_database(connection) == 0:
            print("  warning: no policy passages loaded; run `python -m app.policies` first")
        for _ in range(limit):
            try:
                outcome = work_next(connection, graph)
            except ServiceUnavailable as exc:
                print(f"  {exc.failure_class}, run returned to the queue: {exc}")
                return 1
            if outcome is None:
                print("  queue empty")
                break
            note = f"  ESCALATED ({outcome.failure})" if outcome.failure else ""
            print(
                f"  {outcome.run_id}  {outcome.steps} step(s), now {outcome.status}, "
                f"proposes {outcome.tool:<18}{note}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
