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
waits for a person. issue_refund is judged by the guardrail (app/guardrails.py)
with the limits in force at that moment: allowed, it is paid through the keyed
executor and the run is done; not allowed, an approval is opened and the run
waits for a person. Once a person approves, the next worker to claim the run
pays exactly what was approved, asking no model and judging nothing again. A
refund the ledger refuses, however it was allowed, goes to a person. A proposal that
repeats an earlier step, a run past its step budget, or a tool this worker does
not run are all handed to a person instead of executed. A sender who has caused
RATE_LIMIT lookups within RATE_WINDOW has further lookups deferred: the run goes
back to the queue until the window frees, executing nothing and spending no
attempt.

A model or policy store that cannot be reached marks the run failed, charged for
the calls that completed and keeping every committed step, and schedules it for
another attempt after a backoff delay -- until its attempts run out and it is
marked dead, with a dead letter saying why (app/dead_letters.py). A crash leaves the run marked
running with its committed steps intact. Once its lock is older than
LOCK_TIMEOUT another worker reclaims it, spending an attempt, and carries on
from the last committed step; on its last attempt it is marked dead instead.
The timeout covers a worker that has gone quiet between steps. One whose
connection hangs inside the act transaction keeps its row lock, and SKIP LOCKED
will not take the run until Postgres ends that session -- which, with the
session timeouts every worker sets, it does after IDLE_IN_TRANSACTION_TIMEOUT.

Run:  .venv/bin/python -m app.run_agent [how many runs, default 10]
"""

import os
import random
import socket
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg
from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from app.approvals import (
    EXCERPT_CHARS,
    ApprovedAction,
    approved_unexecuted,
    mark_executed,
    open_approval,
)
from app.baseline import REFERENCE_RATE, token_cost
from app.contracts import Classification, ExtractedRefund, ProposedAction, StepRecord
from app.db import apply_migrations, connect
from app.embeddings import OllamaEmbedder
from app.executor import OWNED_ORDER, ToolOutcome, execute
from app.failures import record_category
from app.graph.build import build_graph, failure_recorded, run_graph
from app.graph.nodes import escalation
from app.graph.prompts import run_prompt_version
from app.graph.state import AgentState
from app.guardrails import Evidence, Verdict, evidence_of, judge, justified
from app.guardrails import load as load_guardrails
from app.llm import Ollama, Reply, ServiceUnavailable
from app.retrieval import PolicyRetriever
from app.tracing import (
    Attr,
    Tracing,
    discard,
    exporter_from_env,
    record_spans,
    run_context,
    tracer,
)

MICRO_DOLLAR = Decimal("0.000001")  # matches runs.cost_usd numeric(10, 6)

# Executed tools a run may use before a person takes over. Small models can wander.
MAX_STEPS = 4

# A failed run waits RETRY_BASE after its first failure, doubling each time, never
# longer than RETRY_CAP. The exponent is capped too, so no attempt count overflows.
RETRY_BASE = timedelta(seconds=30)
RETRY_CAP = timedelta(hours=1)
MAX_DOUBLINGS = 20

# Order lookups one sender may cause in any RATE_WINDOW. Past that their run waits
# for the window to free, spending no attempt. Handing a case to a person is never
# limited: it is the one action that must always be possible.
RATE_LIMIT = 10
RATE_WINDOW = timedelta(hours=1)
RATE_LIMITED = frozenset({"get_order"})

# A run the rate limit defers this many times is handed to a person instead of
# waiting again, so a sender kept over the limit cannot keep a run cycling forever.
MAX_DEFERRALS = 3

# Postgres ends a worker session that sits inside a transaction this long, and the
# run's row lock goes with it; no single statement may run longer than the other.
IDLE_IN_TRANSACTION_TIMEOUT = timedelta(seconds=60)
STATEMENT_TIMEOUT = timedelta(seconds=30)

# Every tool the model can propose is in exactly one of these, and a test holds it
# there, so a new tool in app/tools.py cannot run -- or fail to run -- by default.
RUNS_NOW = frozenset({"get_order", "escalate_to_human"})
# Judged by the guardrail at the act step: paid now, or put to a person to approve.
GUARDED = frozenset({"issue_refund"})
# Retrieval already ran in the graph; a planner asking to search again is handed over.
NOT_RUN_HERE = frozenset({"search_policy"})

# How long a lock may go without a committed step before the run is taken back.
# Every committed step refreshes it, so only a worker that has gone quiet loses it.
LOCK_TIMEOUT = timedelta(minutes=5)

# A run whose worker went quiet on its last attempt is not handed out again; it is
# dead-lettered in the same statement, so no dead run is ever without its letter.
BURY_EXPIRED = """
    WITH buried AS (
        UPDATE runs
           SET status = 'dead', failure_class = 'lock_expired', locked_by = NULL, locked_at = NULL
         WHERE status = 'running' AND locked_at < now() - %s AND attempt >= max_attempts
        RETURNING id, idempotency_key, state
    )
    INSERT INTO dead_letters (kind, run_id, idempotency_key, payload, reason, failure_class)
    SELECT 'run', id, idempotency_key, state, 'lock expired on its last attempt', 'lock_expired'
      FROM buried
"""

CLAIM = """
    UPDATE runs
       SET status = 'running', current_node = 'classify', attempt = attempt + 1,
           locked_by = %s, locked_at = now(), next_retry_at = NULL
     WHERE id = (
            SELECT id FROM runs
             WHERE (status = 'queued' AND (next_retry_at IS NULL OR next_retry_at <= now()))
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

# Taken inside the act transaction before a rate-limited tool runs. Every worker
# counting for the same sender queues behind this lock until the one before it
# commits, so two cannot both see the last free slot.
LOCK_SENDER = """
    SELECT pg_advisory_xact_lock(hashtext(lower(coalesce(state -> 'untrusted' ->> 'sender', ''))))
      FROM runs
     WHERE id = %s
"""

RECENT_LOOKUPS = """
    SELECT count(*), min(tool_calls.created_at)
      FROM tool_calls
      JOIN runs ON runs.id = tool_calls.run_id
     WHERE tool_calls.tool = 'get_order'
       AND tool_calls.created_at > now() - %s
       AND lower(runs.state -> 'untrusted' ->> 'sender') =
           (SELECT lower(state -> 'untrusted' ->> 'sender') FROM runs WHERE id = %s)
"""

# Every statement that records a tick's model work also records the prompt version
# it was made under. Paying an approved action asks no model, so it passes NULL and
# the run keeps the version it was planned under. Each also drops state.billing: the
# totals it writes were counted on from billing's, so billing has nothing left to add.
#
# Busy is not a failure: the attempt the claim spent is given back.
DEFER = """
    UPDATE runs
       SET status = 'queued', attempt = attempt - 1, next_retry_at = %s + %s, state = (state - 'billing') || %s,
           current_node = 'act', cost_usd = cost_usd + %s, prompt_version = coalesce(%s, prompt_version),
           locked_by = NULL, locked_at = NULL
     WHERE id = %s AND locked_by = %s
"""

CONTINUE = """
    UPDATE runs
       SET current_node = 'act', state = (state - 'billing') || %s, cost_usd = cost_usd + %s,
           prompt_version = coalesce(%s, prompt_version), locked_at = now()
     WHERE id = %s AND locked_by = %s
"""

PARK = """
    UPDATE runs
       SET status = 'waiting_approval', current_node = %s, state = (state - 'billing') || %s,
           cost_usd = cost_usd + %s, prompt_version = coalesce(%s, prompt_version),
           locked_by = NULL, locked_at = NULL
     WHERE id = %s AND locked_by = %s
"""

FINISH = """
    UPDATE runs
       SET status = 'done', current_node = 'act', state = (state - 'billing') || %s,
           cost_usd = cost_usd + %s, prompt_version = coalesce(%s, prompt_version),
           locked_by = NULL, locked_at = NULL
     WHERE id = %s AND locked_by = %s
"""

# Taken before a refund is judged. Two runs refunding one order queue here, so the
# second judges against a total that includes the first; the ledger's cap trigger
# takes the same lock later in the transaction. The sum is a separate statement so
# that, once the lock is ours, its snapshot includes whatever the holder committed.
LOCK_OWNED_ORDER = OWNED_ORDER + "   FOR UPDATE OF o"
REFUNDED_SO_FAR = "SELECT coalesce(sum(amount_paise), 0)::bigint FROM refunds WHERE order_id = %s"

# How much of the customer's message is kept with an approval, for the person deciding.
EVIDENCE_BODY_CHARS = EXCERPT_CHARS

# One statement decides retry or dead, so nothing can change attempt in between,
# and a run that dies is dead-lettered by that same statement. It returns the status
# it set, and nothing when this worker no longer held the run. An outage's calls are
# charged like any tick's, counting on from what the run was already charged, and the
# new totals are kept under state.billing so the next tick counts on from them.
RELEASE = """
    WITH released AS (
        UPDATE runs
           SET status = CASE WHEN attempt >= max_attempts THEN 'dead' ELSE 'failed' END,
               failure_class = %s,
               next_retry_at = CASE WHEN attempt >= max_attempts THEN NULL ELSE now() + %s END,
               current_node = 'intake', locked_by = NULL, locked_at = NULL,
               cost_usd = cost_usd + %s, prompt_version = coalesce(%s, prompt_version),
               state = state || %s
         WHERE id = %s AND status = 'running' AND locked_by = %s
        RETURNING id, status, idempotency_key, state, failure_class
    ), lettered AS (
        INSERT INTO dead_letters (kind, run_id, idempotency_key, payload, reason, failure_class)
        SELECT 'run', id, idempotency_key, state, 'out of attempts', failure_class
          FROM released
         WHERE status = 'dead'
    )
    SELECT status FROM released
"""


class LostClaim(RuntimeError):
    """The run was taken by another worker before this one could act on it."""


class ApprovalAlreadyExecuted(RuntimeError):
    """The approval this worker read had been executed before it could act on it."""


@dataclass(frozen=True)
class ClaimedRun:
    run_id: UUID
    subject: str | None
    body: str
    worker: str
    attempt: int
    sender: str | None = None


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


def milliseconds(duration: timedelta) -> str:
    return str(duration // timedelta(milliseconds=1))


def configure_session(
    connection: psycopg.Connection,
    idle_in_transaction: timedelta = IDLE_IN_TRANSACTION_TIMEOUT,
    statement: timedelta = STATEMENT_TIMEOUT,
) -> None:
    """
    Bound how long this worker's session may sit on locks without making progress.

    LOCK_TIMEOUT only helps once a run's row lock is free. A connection hung inside
    the act transaction would keep that lock for as long as its session lived, and
    SKIP LOCKED would pass the run over indefinitely. With these set, Postgres ends
    such a session itself. Set for the session, and committed, so they outlast this call.
    """
    require_idle(connection)
    with connection.transaction():
        connection.execute(
            "SELECT set_config('idle_in_transaction_session_timeout', %s, false)",
            (milliseconds(idle_in_transaction),),
        )
        connection.execute("SELECT set_config('statement_timeout', %s, false)", (milliseconds(statement),))


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
        sender=untrusted.get("sender"),
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

    if proposal.tool not in RUNS_NOW | GUARDED:
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
        "deferrals": before.get("deferrals", 0),
        "model_calls": before.get("model_calls", 0) + len(replies),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "rate": REFERENCE_RATE.name,
    }
    return agent, cost


# What a run has been charged for so far. A tick records these with its agent state; an
# outage has no agent state to record, so it records them alone, under state.billing.
CHARGED = ("prompt_tokens", "completion_tokens", "model_calls")


def agent_of(connection: psycopg.Connection, run_id: UUID) -> dict[str, Any]:
    """
    What earlier ticks of the run recorded, with its charged totals the larger of the
    agent's and an outage's: totals only grow, and the next charge counts on from all of it.
    """
    with connection.transaction():
        row = connection.execute(
            "SELECT state -> 'agent', state -> 'billing' FROM runs WHERE id = %s", (run_id,)
        ).fetchone()
    if row is None:
        return {}
    agent = dict(row[0]) if row[0] else {}
    billing = dict(row[1]) if row[1] else {}
    for key in CHARGED:
        if key in billing:
            agent[key] = max(agent.get(key, 0), billing[key])
    return agent


def record_step(
    connection: psycopg.Connection, run_id: UUID, steps: list[dict[str, Any]], action: ProposedAction
) -> ToolOutcome:
    """Execute `action` as the run's next keyed step and append what it did."""
    number = len(steps) + 1
    done = execute(connection, run_id, number, action)
    record = StepRecord(step=number, tool=action.tool, args=dict(action.args), result=done.result, replayed=done.replayed)
    steps.append(record.model_dump(mode="json"))
    return done


def pay(
    connection: psycopg.Connection, run_id: UUID, steps: list[dict[str, Any]], action: ProposedAction
) -> str | None:
    """Pay a refund as the next step. None if it was paid; otherwise why a person must look."""
    done = record_step(connection, run_id, steps, action)
    if done.result.get("refunded"):
        return None
    return f"act: the ledger refused the refund: {done.result.get('error', 'no reason given')}"


def refunded_so_far(connection: psycopg.Connection, run_id: UUID, action: ProposedAction) -> int:
    """
    What the order has already had back, read under a lock on the order.

    An order this run's sender does not own is neither locked nor counted: the
    executor refuses it exactly as it refuses a missing one.
    """
    order_id = action.args["order_id"]
    if connection.execute(LOCK_OWNED_ORDER, (run_id, order_id)).fetchone() is None:
        return 0
    row = connection.execute(REFUNDED_SO_FAR, (order_id,)).fetchone()
    return int(row[0]) if row else 0


def evidence_for(claimed: ClaimedRun, agent: Mapping[str, Any]) -> dict[str, Any]:
    """What a person needs to decide, kept with the approval as it stood when they were asked."""
    return {
        "sender": claimed.sender,
        "subject": claimed.subject,
        "body": claimed.body[:EVIDENCE_BODY_CHARS],
        "classification": agent["classification"],
        "extraction": agent["extraction"],
        "policy_sources": agent["policy_sources"],
        "steps": agent["steps"],
    }


@contextmanager
def traced_tick(run_id: UUID, attributes: dict[str, Any]) -> Iterator[Span]:
    """
    A tick's root span, current for everything the tick does, in its run's trace.

    A tick that commits closes this span and writes its spans in that transaction
    (close_tick). Leaving any other way -- a lost claim, an approval already paid, an
    error -- commits nothing, so what the tick traced is dropped with it.
    """
    with run_context(run_id):
        span = tracer().start_span("tick", attributes=attributes)
        try:
            with trace.use_span(span, end_on_exit=False, record_exception=False, set_status_on_exception=False):
                yield span
        finally:
            if span.is_recording():
                span.end()
            discard(run_id)


def close_tick(
    connection: psycopg.Connection, run_id: UUID, tick_span: Span, outcome: str, failure_class: str | None = None
) -> None:
    """End the tick's span and write its spans, in the transaction that commits what the tick did."""
    tick_span.set_attribute(Attr.OUTCOME, outcome)
    if failure_class is not None:
        tick_span.set_status(StatusCode.ERROR, failure_class)
    tick_span.end()
    record_spans(connection, run_id)


def oldest_lookup_if_limited(connection: psycopg.Connection, run_id: UUID) -> datetime | None:
    """
    When the run's sender is at the lookup limit, the time of their oldest lookup in the
    window -- the run may come back once it leaves; otherwise None.

    The sender's advisory lock is taken first and held until the transaction ends, so
    two workers counting for one sender cannot both see the last free slot.
    """
    connection.execute(LOCK_SENDER, (run_id,))
    recent = connection.execute(RECENT_LOOKUPS, (RATE_WINDOW, run_id)).fetchone()
    if recent is not None and recent[0] >= RATE_LIMIT:
        return recent[1]
    return None


def defer(
    connection: psycopg.Connection,
    claimed: ClaimedRun,
    before: Mapping[str, Any],
    state: AgentState,
    steps: list[dict[str, Any]],
    proposal: ProposedAction,
    version: str,
    oldest_lookup: datetime,
) -> Decimal:
    """
    Send the run back to the queue until the sender's window frees, and say what the tick cost.

    What this tick found is kept, so the next one plans from it and each model call is
    charged once. Nothing is executed and no attempt is spent.
    """
    deferred, cost = summarise_agent(state, before, proposal, None, steps)
    deferred["deferrals"] = before.get("deferrals", 0) + 1
    connection.execute(
        DEFER,
        (oldest_lookup, RATE_WINDOW, Jsonb({"agent": deferred}), cost, version, claimed.run_id, claimed.worker),
    )
    return cost


def evidence_in(state: AgentState, steps: list[dict[str, Any]], proposal: ProposedAction) -> Evidence:
    """What this run established about the order it proposes to refund: how it read the message, and what it looked up."""
    classification = state.get("classification")
    return evidence_of(classification.intent if classification else None, steps, str(proposal.args["order_id"]))


def judged(
    connection: psycopg.Connection,
    run_id: UUID,
    proposal: ProposedAction,
    evidence: Evidence,
) -> Verdict:
    """The guardrail's verdict on a refund, recorded as the act's guardrail span."""
    with failure_recorded("guardrail", "guardrail") as span:
        # Read now, inside this transaction: a limit changed a second ago applies.
        limits = load_guardrails(connection)
        verdict = judge(proposal, limits, refunded_so_far(connection, run_id, proposal))
        # The limit and the confidence decide whether a person is asked; the conditions decide
        # whether there is anything to ask about. Only a payment that would otherwise have run on
        # its own is checked against them: one already going to a person is a person's to judge.
        if verdict.runs:
            verdict = justified(proposal, evidence)
        span.set_attributes(
            {
                Attr.VERDICT: "runs" if verdict.runs else "needs a person",
                Attr.LIMIT_PAISE: limits.auto_refund_limit_paise,
                Attr.MIN_CONFIDENCE: str(limits.min_confidence),
            }
        )
        if verdict.reason:
            span.set_attribute(Attr.REASON, verdict.reason)
    return verdict


def act(
    connection: psycopg.Connection,
    claimed: ClaimedRun,
    before: Mapping[str, Any],
    state: AgentState,
    steps: list[dict[str, Any]],
    proposal: ProposedAction,
    failure: str | None,
    version: str,
) -> tuple[str, ProposedAction, str | None, Decimal]:
    """
    Carry out one proposal in the tick's transaction, as the tick's act span, and record the run.

    Returns the status the run was left in, the action actually taken (a hand-over can
    replace the proposal), why a person must look if one must, and what the tick cost.
    """
    with failure_recorded("act", "tool") as span:
        span.set_attribute(Attr.TOOL, proposal.tool)
        if proposal.tool in RATE_LIMITED:
            oldest_lookup = oldest_lookup_if_limited(connection, claimed.run_id)
            if oldest_lookup is not None:
                deferrals = before.get("deferrals", 0)
                if deferrals < MAX_DEFERRALS:
                    cost = defer(connection, claimed, before, state, steps, proposal, version, oldest_lookup)
                    span.set_attribute(Attr.RESULT, "deferred by the rate limit")
                    return "queued", proposal, None, cost
                proposal, failure = hand_over(
                    f"deferred {deferrals} times by the rate limit",
                    "The sender kept this case over the lookup limit; a person should look at it.",
                )
                span.set_attribute(Attr.TOOL, proposal.tool)

        if proposal.tool in RUNS_NOW:
            record_step(connection, claimed.run_id, steps, proposal)

        verdict = judged(connection, claimed.run_id, proposal, evidence_in(state, steps, proposal)) if proposal.tool in GUARDED else None
        if verdict is not None and verdict.refused:
            # Nothing to approve: the agent cannot say this refund is owed, so the case goes to a person.
            proposal, failure = hand_over("the conditions for paying it were not met", verdict.reason or "")
            span.set_attribute(Attr.TOOL, proposal.tool)
            # Dropped deliberately, and the rest of this function depends on it: from here on there is
            # no refund under consideration, so the run must be recorded exactly like any other
            # hand-over. Everything below reads `verdict is None` as "no refund to pay or approve".
            verdict = None
        if verdict is not None and verdict.runs:
            failure = pay(connection, claimed.run_id, steps, proposal)

        agent, cost = summarise_agent(state, before, proposal, failure, steps)
        stored = (Jsonb({"agent": agent}), cost, version, claimed.run_id, claimed.worker)
        if proposal.tool == "get_order":
            status, result = "running", "looked up"
            connection.execute(CONTINUE, stored)
        elif verdict is not None and not verdict.runs:
            status, result = "waiting_approval", "approval opened"
            reason = verdict.reason or ""
            approval_id = open_approval(connection, claimed.run_id, proposal, evidence_for(claimed, agent), reason)
            span.set_attribute(Attr.APPROVAL_ID, approval_id)
            connection.execute(PARK, ("approval", *stored))
        elif verdict is not None and failure is None:
            status, result = "done", "refunded"
            connection.execute(FINISH, stored)
        else:
            status = "waiting_approval"
            result = "refused by the ledger" if verdict is not None else "handed to a person"
            # A run that escalated early stopped at the step that failed.
            node = failure.split(":", 1)[0] if failure else "plan"
            connection.execute(PARK, (node, *stored))
        span.set_attribute(Attr.RESULT, result)
        if status != "running":
            # The run has rested: name how it failed, if it did, in this same transaction.
            category = record_category(connection, claimed.run_id)
            if category is not None:
                span.set_attribute(Attr.FAILURE_CATEGORY, category.value)
    return status, proposal, failure, cost


def tick(connection: psycopg.Connection, graph: Any, claimed: ClaimedRun, max_steps: int) -> RunOutcome:
    """Walk the graph from what the run holds, then act on one proposal in one transaction."""
    with traced_tick(claimed.run_id, {Attr.ATTEMPT: claimed.attempt}) as tick_span:
        before = agent_of(connection, claimed.run_id)
        # Read before the walk: the version this tick's prompts were built from.
        version = run_prompt_version()
        try:
            state = run_graph(graph, claimed.subject, claimed.body, prior_from(before))
        except ServiceUnavailable as outage:
            # Failed, to be tried again later: charged for the calls that did complete, steps kept.
            with connection.transaction():
                # Rounded as part of the run's running total, not on its own: charged apart,
                # fractions of a micro-dollar here and on the next tick would each round away.
                cost, prompt_tokens, completion_tokens = charge(before, outage.replies)
                billing = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "model_calls": before.get("model_calls", 0) + len(outage.replies),
                }
                released = connection.execute(
                    RELEASE,
                    (
                        outage.failure_class,
                        retry_delay(claimed.attempt),
                        cost,
                        version,
                        Jsonb({"billing": billing}),
                        claimed.run_id,
                        claimed.worker,
                    ),
                ).fetchone()
                if released is not None:
                    close_tick(connection, claimed.run_id, tick_span, released[0], outage.failure_class)
            raise

        steps = list(before.get("steps", []))
        proposal, failure = decide(state, steps, max_steps)
        with connection.transaction():
            if connection.execute(HOLD_CLAIM, (claimed.run_id, claimed.worker)).fetchone() is None:
                raise LostClaim(f"run {claimed.run_id} was reclaimed before this worker could act on it")
            status, proposal, failure, cost = act(connection, claimed, before, state, steps, proposal, failure, version)
            close_tick(connection, claimed.run_id, tick_span, status)

    return RunOutcome(
        run_id=claimed.run_id, status=status, tool=proposal.tool, steps=len(steps), failure=failure, cost_usd=cost
    )


def act_on_approval(connection: psycopg.Connection, claimed: ClaimedRun, approved: ApprovedAction) -> RunOutcome:
    """
    Pay exactly what a person approved: no model is asked, and the guardrail is not
    applied again, since a person has already overruled it. The keyed executor, the
    sender's ownership of the order and the ledger cap all still apply.
    """
    attributes = {Attr.ATTEMPT: claimed.attempt, Attr.APPROVAL_ID: approved.id}
    with traced_tick(claimed.run_id, attributes) as tick_span:
        before = agent_of(connection, claimed.run_id)
        steps = list(before.get("steps", []))
        tool = str(approved.action.get("tool", "unknown"))
        with connection.transaction():
            if connection.execute(HOLD_CLAIM, (claimed.run_id, claimed.worker)).fetchone() is None:
                raise LostClaim(f"run {claimed.run_id} was reclaimed before this worker could act on it")
            # Stamped first, in the transaction that pays: a copy of an approval that has
            # since been executed pays nothing and records nothing.
            if not mark_executed(connection, approved.id):
                raise ApprovalAlreadyExecuted(f"approval {approved.id} was already executed")

            with failure_recorded("act", "tool") as act:
                act.set_attribute(Attr.TOOL, tool)
                failure: str | None
                try:
                    action = ProposedAction.model_validate(approved.action)
                except ValidationError as unreadable:
                    # It was valid when proposed, so only a schema change since can land here.
                    failure = f"act: the approved action could not be read ({unreadable.error_count()} validation errors)"
                    act.set_attribute(Attr.RESULT, "the approved action could not be read")
                else:
                    failure = pay(connection, claimed.run_id, steps, action)
                    act.set_attribute(Attr.RESULT, "refunded" if failure is None else "refused by the ledger")
                stored = (
                    Jsonb({"agent": {**before, "steps": steps, "failure": failure}}),
                    Decimal(0),
                    None,  # no model was asked, so the version the run was planned under stands
                    claimed.run_id,
                    claimed.worker,
                )
                if failure is None:
                    status = "done"
                    connection.execute(FINISH, stored)
                else:
                    status = "waiting_approval"
                    connection.execute(PARK, ("act", *stored))
                category = record_category(connection, claimed.run_id)
                if category is not None:
                    act.set_attribute(Attr.FAILURE_CATEGORY, category.value)
            close_tick(connection, claimed.run_id, tick_span, status)

    return RunOutcome(
        run_id=claimed.run_id,
        status=status,
        tool=tool,
        steps=len(steps),
        failure=failure,
        cost_usd=Decimal(0),  # no model is asked: paying what a person approved costs no tokens
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
    Claim one run and work it until it is done or waits for a person.

    A run a person has approved is paid as approved, without walking the graph.
    `after_step` is called once each continuing step has been committed, with the
    run id and how many steps it now has -- the moment a dying worker loses nothing.
    """
    configure_session(connection)
    claimed = claim_next(connection, worker or default_worker(), lock_timeout)
    if claimed is None:
        return None

    with connection.transaction():
        approved = approved_unexecuted(connection, claimed.run_id)
    if approved is not None:
        return act_on_approval(connection, claimed, approved)

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


def run_worker(graph: Any, *, limit: int, environ: Mapping[str, str]) -> int:  # pragma: no cover - runs in a worker process, see tests/test_worker_tracing.py
    """
    Work up to `limit` runs, traced: what the command line runs, with its model given.

    Spans are always recorded with the steps; OPSAGENT_OTLP_ENDPOINT adds a copy to
    Langfuse, sent before this returns. An endpoint off this machine is refused before
    any run is claimed, with exit status 2.
    """
    try:
        exporter = exporter_from_env(environ)
    except ValueError as refused:
        print(f"  not started: {refused}", file=sys.stderr)
        return 2
    tracing = Tracing(exporter).install()
    try:
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
    finally:
        tracing.shutdown()


def main(argv: list[str]) -> int:  # pragma: no cover - the interactive driver
    limit = int(argv[1]) if len(argv) > 1 else 10
    graph = build_graph(Ollama(), PolicyRetriever(connect, OllamaEmbedder()))
    return run_worker(graph, limit=limit, environ=os.environ)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
