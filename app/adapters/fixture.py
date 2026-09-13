"""
Reading messages out of a JSONL file.

This is the adapter the tests and the golden eval set use. It has the same shape
every other adapter will have: it produces IncomingMessage objects and knows
nothing whatsoever about runs, the database, or the agent.

It refuses malformed input rather than skipping it. A line quietly passed over
is a customer whose email was never answered, with nothing anywhere recording
that it happened -- which is the failure this project is about.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from app.contracts import IncomingMessage


def read_messages(path: Path) -> Iterator[IncomingMessage]:
    """
    Yield one IncomingMessage per non-blank line, in file order.

    Raises ValueError naming the line number if a line cannot be read. Lazy, so a
    large inbox is not held in memory at once -- which also means the error
    surfaces when the bad line is reached, not before.
    """
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                yield IncomingMessage(**json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} line {number} is not valid JSON: {exc}") from exc
            except ValidationError as exc:
                raise ValueError(f"{path.name} line {number} is not a valid message: {exc}") from exc
