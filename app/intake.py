"""
Turning a message into a run, exactly once.

The only interesting problem here is the second delivery. Pollers restart,
mailboxes re-serve messages, and a retry after a timeout looks identical to a new
request. Each of those must produce the run that already exists rather than
another one.

The check is left to the database. `INSERT ... ON CONFLICT DO NOTHING` is decided
inside Postgres while the row is being written, so two callers racing on the same
message cannot both pass it. The equivalent written in Python -- look for the
key, insert if absent -- has a gap between the two statements, and the gap is
where the duplicate is born.

DO NOTHING, not DO UPDATE. By the time a message is delivered again the agent may
be partway through the case, and an upsert would send it back to the start.
"""

from dataclasses import dataclass
from uuid import UUID

import psycopg

from app.contracts import IncomingMessage, RunRecord


@dataclass(frozen=True)
class IntakeResult:
    """
    What happened, said plainly.

    `created` is false for a message that was already known. Callers mostly do
    not care, but it is the number worth watching: duplicates suddenly rising
    means something upstream is re-delivering, and that is worth knowing before
    it turns into anything else.
    """

    run_id: UUID
    created: bool


def accept(connection: psycopg.Connection, message: IncomingMessage) -> IntakeResult:
    """
    Record `message` as a run, or report the run it already produced.

    Does not commit. The caller owns the transaction, because a poller will want
    to record the run and mark the source message as read together or not at all.
    """
    run = RunRecord.from_message(message)

    inserted = connection.execute(
        """
        INSERT INTO runs (id, channel, status, current_node, state, idempotency_key)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING id
        """,
        (
            run.id,
            run.channel,
            run.status,
            run.current_node,
            psycopg.types.json.Jsonb(run.state),
            run.idempotency_key,
        ),
    ).fetchone()

    if inserted is not None:
        return IntakeResult(run_id=inserted[0], created=True)

    # The insert was refused, so the row is someone else's. Read it back rather
    # than assuming anything about what it now contains.
    existing = connection.execute(
        "SELECT id FROM runs WHERE idempotency_key = %s", (run.idempotency_key,)
    ).fetchone()

    if existing is None:  # pragma: no cover - would mean the row vanished mid-statement
        raise RuntimeError(
            f"insert of {run.idempotency_key} conflicted with a row that is not there"
        )

    return IntakeResult(run_id=existing[0], created=False)
