"""
One pass over an inbox.

Reads messages an adapter offers, records each as a run, and commits the pass.
Nothing here is specific to files: swap the adapter for Gmail and this code does
not change, which is the point of having an adapter at all.

Three decisions worth stating.

A pass owns its transaction outright. Called on a connection that already has
work open, it would commit that work too, as a side effect of polling something
unrelated. It refuses instead.

A pass is bounded, and the bound counts work done rather than messages read.
An inbox is as large as whoever fills it decides, and one unbounded pass is one
unbounded transaction holding locks for as long as it takes.

Counting reads instead would deadlock the poller outright, which is worth
spelling out because it is not obvious. An adapter has no memory: it offers the
whole inbox again on the next pass, starting from the beginning. If the first
`limit` messages were already handled, a pass that counted them would spend its
entire budget recognising duplicates and stop in exactly the same place, every
time, forever. Duplicates are cheap -- a lookup and nothing written -- so they
stream past, and the budget is spent only on rows that actually appear.

Within a pass it is all or nothing. Stopping halfway and keeping what it had
would make the number of messages handled depend on where the file broke, and
nothing downstream could tell that apart from a quiet inbox. The bound above is
what keeps that honest rather than expensive.

Run:  .venv/bin/python -m app.poll fixtures/inbox.jsonl
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.pq import TransactionStatus

from app.adapters.fixture import read_messages
from app.db import connect
from app.intake import accept

# Large enough that an ordinary backlog clears in one pass, small enough that the
# transaction behind it stays short.
DEFAULT_LIMIT = 500


@dataclass(frozen=True)
class PollSummary:
    accepted: int
    duplicates: int
    collisions: int
    more_waiting: bool

    @property
    def seen(self) -> int:
        return self.accepted + self.duplicates + self.collisions


def poll_once(
    connection: psycopg.Connection, inbox: Path, limit: int = DEFAULT_LIMIT
) -> PollSummary:
    """
    Take everything in `inbox`, stopping once `limit` runs have been written.

    Commits the pass. Messages already known do not count against the limit; see
    the module docstring for why counting them would stall the poller.
    """
    if connection.pgconn.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError(
            "poll_once commits, so it needs its own transaction: call it on a "
            "connection with no work already open"
        )

    accepted = duplicates = collisions = 0
    more_waiting = False
    messages = read_messages(inbox)

    with connection.transaction():
        for message in messages:
            result = accept(connection, message)
            if result.created:
                accepted += 1
            elif result.collided:
                collisions += 1
            else:
                duplicates += 1

            if accepted + collisions >= limit:
                more_waiting = next(messages, None) is not None
                break

    return PollSummary(
        accepted=accepted,
        duplicates=duplicates,
        collisions=collisions,
        more_waiting=more_waiting,
    )


def main(argv: list[str]) -> int:  # pragma: no cover - the interactive driver
    inbox = Path(argv[1] if len(argv) > 1 else "fixtures/inbox.jsonl")

    with connect() as connection:
        summary = poll_once(connection, inbox)

    print(f"  read      {summary.seen} message(s) from {inbox}")
    print(f"  accepted  {summary.accepted}")
    print(f"  duplicate {summary.duplicates}")

    if summary.collisions:
        print(f"\n  COLLIDED  {summary.collisions}")
        print("  A message arrived reusing a key that already exists, carrying")
        print("  different text. It was not a re-delivery, and its contents have")
        print("  been discarded. Someone should look at that.")
    if summary.more_waiting:
        print("\n  More waiting. Run it again.")
    elif summary.accepted == 0 and summary.seen:
        print("\n  Nothing new. Run it again as often as you like; that is the point.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
