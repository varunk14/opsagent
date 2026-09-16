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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb

from app.adapters.fixture import read_messages
from app.adapters.mailbox import (
    MAX_FETCHED,
    Fetched,
    Mailbox,
    mark_read,
    unread_messages,
)
from app.contracts import IncomingMessage
from app.db import connect
from app.intake import accept

# Large enough that an ordinary backlog clears in one pass, small enough that the
# transaction behind it stays short.
DEFAULT_LIMIT = 500

# The same unreadable message is kept once. The conflict target is migration 005's partial unique
# index, which is what makes a redelivery free rather than another row.
QUARANTINE_REFUSAL = """
    INSERT INTO dead_letters (kind, idempotency_key, payload, reason)
    VALUES ('message', %s, %s, %s)
    ON CONFLICT (idempotency_key, md5(payload::text)) WHERE kind = 'message' DO NOTHING
"""


@dataclass(frozen=True)
class PollSummary:
    accepted: int
    duplicates: int
    collisions: int
    more_waiting: bool
    refused: int = 0

    @property
    def seen(self) -> int:
        return self.accepted + self.duplicates + self.collisions + self.refused


def has_more(messages: Iterator[IncomingMessage]) -> bool:
    """
    Peek past the limit without letting that line decide this pass.

    A malformed line counts as waiting. The next pass reads it for real and
    fails loudly there, instead of rolling back runs accepted here.
    """
    try:
        return next(messages, None) is not None
    except ValueError:
        return True


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
                more_waiting = has_more(messages)
                break

    return PollSummary(
        accepted=accepted,
        duplicates=duplicates,
        collisions=collisions,
        more_waiting=more_waiting,
    )


def poll_mailbox(connection: psycopg.Connection, mailbox: Mailbox) -> PollSummary:
    """
    Take one pass over a mailbox: record what it offers, then tell it what was recorded.

    The ordering is the whole design, and it is the opposite way round from what is convenient.
    Nothing is marked read until the transaction has committed. A pass that dies in the middle
    therefore offers the same messages again, and the next pass recognises them and writes nothing
    -- where marking first would leave an email read with no run behind it, which is a customer
    dropped in silence and no record anywhere that it happened.

    What it costs: delivery is at-least-once, so a crash between the commit and the marking means
    a second look at work already done. Intake makes that cheap. It is the safe direction to be
    wrong in, and the other direction has no safe version.

    A message that cannot be read is written to dead_letters and then marked read like any other.
    Left unread it would be re-fetched and re-parsed on every pass forever, spending one of the
    pass's slots each time -- so one deliberately malformed email would degrade intake permanently.
    Recorded, it is on the screen where someone will see it, which is the thing that mattered about
    leaving it in the mailbox in the first place.

    How many messages a pass takes is MAX_FETCHED, in the adapter. Commits.
    """
    if connection.pgconn.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError(
            "poll_mailbox commits, so it needs its own transaction: call it on a "
            "connection with no work already open"
        )

    accepted = duplicates = collisions = refused = 0
    handled: list[bytes] = []

    with connection.transaction():
        for item in unread_messages(mailbox):
            if item.message is None:
                quarantine_refusal(connection, item)
                refused += 1
            else:
                result = accept(connection, item.message)
                if result.created:
                    accepted += 1
                elif result.collided:
                    collisions += 1
                else:
                    duplicates += 1

            handled.append(item.number)

    # Only now, and outside the transaction: a failure here costs a repeat, where a failure inside
    # it would roll back work the mailbox had already been told to forget.
    for number in handled:
        mark_read(mailbox, number)

    return PollSummary(
        accepted=accepted,
        duplicates=duplicates,
        collisions=collisions,
        refused=refused,
        more_waiting=len(handled) >= MAX_FETCHED,
    )


def quarantine_refusal(connection: psycopg.Connection, item: Fetched) -> None:
    """
    Record a message that could not be read, once, however many times it arrives.

    Keyed on the digest of the bytes rather than on anything inside the message. A message we
    refused has no Message-ID we are willing to trust -- often that is precisely why it was refused
    -- and a key taken from its contents would let one bad message stand in for another and hide it.

    Does not commit; it belongs to the pass's transaction, so the letter and the run counts land
    together or not at all.
    """
    connection.execute(
        QUARANTINE_REFUSAL,
        (
            f"email-sha256:{item.digest}",
            Jsonb({"preview": item.preview, "refusal": item.refusal}),
            f"the message could not be read: {item.refusal}",
        ),
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
        print("  different text. It was not a re-delivery, so it did not become a")
        print("  run; it is quarantined. `python -m app.dead_letters` lists it.")
    if summary.more_waiting:
        print("\n  More waiting. Run it again.")
    elif summary.accepted == 0 and summary.seen:
        print("\n  Nothing new. Run it again as often as you like; that is the point.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
