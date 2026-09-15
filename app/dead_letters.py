"""
Dead letters: what could not be processed, why, and a way to send it back.

A run is dead-lettered by the driver in the same statement that marks it dead
(app/run_agent.py), so there is never a dead run without its letter, nor a
letter for a run that died and came back unnoticed. This module is the
operator's side: list what is open, and requeue a run with a fresh set of
attempts.

Requeueing happens once. The letter is stamped and the run reset in a single
statement that insists on what it expects -- an open letter, a dead run -- so
asking twice, or asking about a run that is not dead, changes nothing. Nothing
here commits; the caller owns the transaction.

Run:  .venv/bin/python -m app.dead_letters                 list open letters
      .venv/bin/python -m app.dead_letters requeue RUN_ID  send one run back
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import psycopg

from app.db import connect

LIST_OPEN = """
    SELECT id, kind, run_id, idempotency_key, reason, failure_class, created_at
      FROM dead_letters
     WHERE requeued_at IS NULL
     ORDER BY created_at, id
"""

# One statement: the letter is stamped only if the run is dead, and the run is
# reset only if its open letter was stamped. Neither can happen without the other.
REQUEUE = """
    WITH letter AS (
        UPDATE dead_letters
           SET requeued_at = now()
         WHERE kind = 'run' AND run_id = %s AND requeued_at IS NULL
           AND EXISTS (SELECT 1 FROM runs WHERE id = %s AND status = 'dead')
        RETURNING run_id
    )
    UPDATE runs
       SET status = 'queued', attempt = 0, next_retry_at = NULL, failure_class = NULL,
           current_node = 'intake', locked_by = NULL, locked_at = NULL
      FROM letter
     WHERE runs.id = letter.run_id AND runs.status = 'dead'
    RETURNING runs.id
"""


@dataclass(frozen=True)
class DeadLetter:
    id: int
    kind: str
    run_id: UUID | None
    idempotency_key: str
    reason: str
    failure_class: str | None
    created_at: datetime


def list_open(connection: psycopg.Connection) -> list[DeadLetter]:
    """Letters nobody has acted on yet, oldest first."""
    return [DeadLetter(*row) for row in connection.execute(LIST_OPEN).fetchall()]


def requeue(connection: psycopg.Connection, run_id: UUID | str) -> bool:
    """Send a dead run back to the queue with fresh attempts. False if there was nothing to send back."""
    return connection.execute(REQUEUE, (run_id, run_id)).fetchone() is not None


def main(argv: list[str]) -> int:  # pragma: no cover - the operator's command line
    with connect() as connection:
        if len(argv) == 3 and argv[1] == "requeue":
            if requeue(connection, argv[2]):
                print(f"  requeued {argv[2]}")
                return 0
            print(f"  nothing to requeue for {argv[2]}: no open letter for a dead run")
            return 1

        letters = list_open(connection)
        for letter in letters:
            subject = letter.run_id or letter.idempotency_key
            print(
                f"  {letter.created_at:%Y-%m-%d %H:%M}  {letter.kind:<7}  {subject}  "
                f"{letter.failure_class or '-'}  {letter.reason}"
            )
        if not letters:
            print("  no open dead letters")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
