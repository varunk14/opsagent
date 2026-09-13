"""
One pass over an inbox.

Reads every message an adapter offers, records each as a run, and commits once.
Nothing here is specific to files: swap the adapter for Gmail and this code does
not change, which is the point of having an adapter at all.

Two decisions worth stating.

The pass is one transaction. A pass that broke halfway and kept what it had
would leave the number of messages handled depending on where the file happened
to break, and nothing downstream could tell that apart from a quiet inbox.

Duplicates are counted, not ignored. On a healthy second pass every message is a
duplicate and that is correct. What matters is the number moving when nobody
changed anything, which means something upstream has started re-delivering.

Run:  .venv/bin/python -m app.poll fixtures/inbox.jsonl
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

from app.adapters.fixture import read_messages
from app.db import connect
from app.intake import accept


@dataclass(frozen=True)
class PollSummary:
    accepted: int
    duplicates: int

    @property
    def seen(self) -> int:
        return self.accepted + self.duplicates


def poll_once(connection: psycopg.Connection, inbox: Path) -> PollSummary:
    """
    Take everything in `inbox` and record it. Commits the whole pass, or nothing.
    """
    accepted = duplicates = 0

    with connection.transaction():
        for message in read_messages(inbox):
            if accept(connection, message).created:
                accepted += 1
            else:
                duplicates += 1

    connection.commit()
    return PollSummary(accepted=accepted, duplicates=duplicates)


def main(argv: list[str]) -> int:  # pragma: no cover - the interactive driver
    inbox = Path(argv[1] if len(argv) > 1 else "fixtures/inbox.jsonl")

    with connect() as connection:
        summary = poll_once(connection, inbox)

    print(f"  read      {summary.seen} message(s) from {inbox}")
    print(f"  accepted  {summary.accepted}")
    print(f"  duplicate {summary.duplicates}")
    if summary.accepted == 0 and summary.seen:
        print("\n  Nothing new. Run it again as often as you like; that is the point.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
