"""
Approvals: asking a person, and recording what they decided.

The worker opens an approval inside its act transaction when the guardrail will
not let a refund run on its own (app/run_agent.py), and parks the run. A person
decides. Deciding is one statement that insists on what it expects -- the
approval still pending, its run still waiting -- so a double click, two people at
once, or a run that has moved on all change nothing the second time.

Approved, the run goes back to the queue and the worker pays exactly the stored
action, once. Rejected, the run is done and nothing is paid. Nothing here runs a
tool -- the executor is only ever the worker's -- and nothing here commits: the
caller owns the transaction.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from app.contracts import ProposedAction

MAX_NOTE = 1000

# How much of a customer's message is kept for the person deciding.
EXCERPT_CHARS = 2000

OPEN = """
    INSERT INTO approvals (run_id, action, evidence, confidence, reason)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id
"""

# Never the run's state or its lock holder: only what was put to the person.
LIST_PENDING = """
    SELECT id, run_id, action, evidence, confidence, reason, created_at
      FROM approvals
     WHERE status = 'pending'
     ORDER BY created_at, id
"""

# One statement. The run is locked first, so a second decision waits for the first
# and then finds the run no longer waiting; the approval is decided only while it
# is pending; the run moves only if its approval was decided.
DECIDE = """
    WITH waiting AS (
        SELECT runs.id
          FROM runs
          JOIN approvals ON approvals.run_id = runs.id
         WHERE approvals.id = %(id)s AND runs.status = 'waiting_approval'
           FOR UPDATE OF runs
    ), decided AS (
        UPDATE approvals
           SET status = CASE WHEN %(approved)s THEN 'approved' ELSE 'rejected' END,
               decided_by = %(by)s, decided_at = now(), decision_note = %(note)s
          FROM waiting
         WHERE approvals.id = %(id)s AND approvals.run_id = waiting.id AND approvals.status = 'pending'
        RETURNING approvals.run_id, approvals.status
    )
    UPDATE runs
       SET status = CASE WHEN decided.status = 'approved' THEN 'queued' ELSE 'done' END,
           failure_class = CASE WHEN decided.status = 'rejected' THEN 'rejected' ELSE runs.failure_class END,
           next_retry_at = NULL
      FROM decided
     WHERE runs.id = decided.run_id
    RETURNING runs.id
"""

# Runs waiting for a person with nothing to approve: escalations, and actions that
# were allowed but could not be completed. Without this list they wait where no
# screen shows them. Never the run's raw state or its lock holder.
HANDED_OVER = """
    SELECT runs.id,
           runs.current_node,
           coalesce(runs.state -> 'agent' ->> 'failure',
                    runs.state -> 'agent' -> 'proposal' -> 'args' ->> 'reason'),
           runs.state -> 'untrusted' ->> 'sender',
           runs.state -> 'untrusted' ->> 'subject',
           left(runs.state -> 'untrusted' ->> 'body', %s),
           runs.created_at
      FROM runs
     WHERE runs.status = 'waiting_approval'
       AND NOT EXISTS (
            SELECT 1 FROM approvals WHERE approvals.run_id = runs.id AND approvals.status = 'pending'
           )
     ORDER BY runs.created_at, runs.id
"""

APPROVED_UNEXECUTED = """
    SELECT id, action FROM approvals
     WHERE run_id = %s AND status = 'approved' AND executed_at IS NULL
"""

MARK_EXECUTED = """
    UPDATE approvals SET executed_at = now()
     WHERE id = %s AND status = 'approved' AND executed_at IS NULL
    RETURNING id
"""


@dataclass(frozen=True)
class PendingApproval:
    """What the approvals screen may show about one decision waiting to be made."""

    id: int
    run_id: UUID
    action: dict[str, Any]
    evidence: dict[str, Any]
    confidence: Decimal | None
    reason: str
    created_at: datetime


@dataclass(frozen=True)
class HandedOver:
    """A run waiting for a person with nothing to approve, and why it stopped."""

    run_id: UUID
    node: str
    why: str | None
    sender: str | None
    subject: str | None
    body: str | None
    created_at: datetime


@dataclass(frozen=True)
class ApprovedAction:
    """An approval a person granted, with the action exactly as stored -- validated by whoever executes it."""

    id: int
    action: dict[str, Any]


def open_approval(
    connection: psycopg.Connection,
    run_id: UUID,
    action: ProposedAction,
    evidence: Mapping[str, Any],
    reason: str,
) -> int:
    """Ask a person to decide on `action`, recording why they are being asked."""
    row = connection.execute(
        OPEN, (run_id, Jsonb(action.model_dump(mode="json")), Jsonb(dict(evidence)), action.confidence, reason)
    ).fetchone()
    if row is None:  # pragma: no cover - INSERT ... RETURNING always yields the row
        raise RuntimeError(f"approval for run {run_id} returned no row")
    return int(row[0])


def list_pending(connection: psycopg.Connection) -> list[PendingApproval]:
    """Decisions waiting to be made, oldest first."""
    return [PendingApproval(*row) for row in connection.execute(LIST_PENDING).fetchall()]


def list_handed_over(connection: psycopg.Connection) -> list[HandedOver]:
    """Runs waiting for a person that have nothing to approve, oldest first."""
    return [HandedOver(*row) for row in connection.execute(HANDED_OVER, (EXCERPT_CHARS,)).fetchall()]


def decide(
    connection: psycopg.Connection, approval_id: int, *, approved: bool, by: str, note: str | None = None
) -> bool:
    """Record a person's decision and move the run on. False if there was nothing to decide."""
    name = by.strip()
    if not name:
        raise ValueError("say who is deciding")
    if note is not None and len(note) > MAX_NOTE:
        raise ValueError(f"a note is at most {MAX_NOTE} characters")

    moved = connection.execute(
        DECIDE, {"id": approval_id, "approved": approved, "by": name, "note": note}
    ).fetchone()
    return moved is not None


def approved_unexecuted(connection: psycopg.Connection, run_id: UUID) -> ApprovedAction | None:
    """The action a person approved for this run that has not been executed yet, if any."""
    row = connection.execute(APPROVED_UNEXECUTED, (run_id,)).fetchone()
    if row is None:
        return None
    return ApprovedAction(id=row[0], action=dict(row[1]))


def mark_executed(connection: psycopg.Connection, approval_id: int) -> bool:
    """Stamp an approved action as executed. False if it already was, or was never approved."""
    return connection.execute(MARK_EXECUTED, (approval_id,)).fetchone() is not None
