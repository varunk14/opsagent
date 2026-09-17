"""
Replaying a run: ask whether an old case would go differently now.

A prompt changed, a policy was reworded, a line of code was fixed -- and the question is whether the
case that went wrong last week goes right this week. Replay answers it by making a fresh run that
carries the same customer message and letting the worker take it under whatever is true now. The
original is never touched, because the whole value is in comparing the two, and a comparison against
something that was itself edited proves nothing.

What a replay copies is only what the customer wrote: the `untrusted` block and when it arrived.
None of the original's own work comes across -- no steps, no proposal, no cost -- so the new run
starts from nothing the first one worked out, exactly as intake would have left it. It gets its own
id and its own idempotency key (`replay:<id>`, never the original's, which is unique and would look
like a re-delivery), and `replay_of` threads it back to where it came from.

Does not commit: the caller owns the transaction.

Run:  .venv/bin/python -m app.replay <run_id>
"""

import sys
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Jsonb

from app.db import connect

ORIGINAL = "SELECT channel, state FROM runs WHERE id = %s"

INSERT_REPLAY = """
    INSERT INTO runs (id, channel, status, current_node, state, idempotency_key, replay_of)
    VALUES (%s, %s, 'queued', 'intake', %s, %s, %s)
"""


def replay(connection: psycopg.Connection, original_id: UUID) -> UUID:
    """Queue a fresh run carrying `original_id`'s message, threaded back to it. Does not commit."""
    row = connection.execute(ORIGINAL, (original_id,)).fetchone()
    if row is None:
        raise ValueError(f"no run {original_id} to replay")
    channel, state = row

    untrusted = state.get("untrusted")
    if untrusted is None:  # pragma: no cover - every accepted run has one; a guard, not a path
        raise ValueError(f"run {original_id} has no message to replay")

    # Only what the customer wrote. Nothing the original worked out comes across.
    fresh = {"untrusted": untrusted}
    if "received_at" in state:
        fresh["received_at"] = state["received_at"]

    new_id = uuid4()
    connection.execute(INSERT_REPLAY, (new_id, channel, Jsonb(fresh), f"replay:{new_id}", original_id))
    return new_id


def main(argv: list[str]) -> int:  # pragma: no cover - the interactive driver
    if len(argv) != 2:
        print("usage: python -m app.replay <run_id>")
        return 2
    try:
        original = UUID(argv[1])
    except ValueError:
        print(f"  {argv[1]!r} is not a run id.")
        return 2

    with connect() as connection:
        try:
            new_id = replay(connection, original)
        except ValueError as exc:
            print(f"  {exc}")
            return 1

    print(f"  replayed  {original}")
    print(f"  new run   {new_id}")
    print("  Queued. `python -m app.run_agent` will work it under what is true now.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
