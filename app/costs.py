"""
What the agent has been costing, for the screen to draw.

Two questions, both answered from rows the worker already writes. What did a run cost on each day it
ran -- `runs.cost_usd`, which `app/run_agent.py` sets when a run rests. And where did the tokens go
-- `runs.state -> 'agent' -> 'tokens_by_model'`, kept per model since each is priced at its own rate.

The day is measured by **cost per run**, not by the day's total. A busy day is not a dear one, and a
page that ranked days by total spend would show a queue backlog as a cost regression and hide a real
one on a quiet day.

The second question is answered out of a JSON column the agent fills from what a model said. So the
model names and the counts there are read the way app/failures.py reads the golden history: anything
that is not a whole number of tokens counts as none, and a name longer than a name is not drawn.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import psycopg

# The same guard the failure trend uses on counts out of a file we did not write. Shared rather than
# copied: two versions of "do not believe this number" drift, and only one of them gets fixed.
from app.failures import counted

# A model name is a name. Ollama's longest is well inside this; an essay here came from somewhere else.
MAX_MODEL_NAME_CHARS = 64
# Six places, as `runs.cost_usd` and the budgets are stored.
MONEY = Decimal("0.000001")

SPEND_BY_DAY = """
    SELECT created_at::date AS day, count(*), coalesce(sum(cost_usd), 0)
      FROM runs
     GROUP BY day
     ORDER BY day
"""

TOKENS_BY_MODEL = "SELECT state -> 'agent' -> 'tokens_by_model' FROM runs"


@dataclass(frozen=True)
class Day:
    """One day of the burn-down, as the page draws it."""

    on: date
    runs: int
    spent: Decimal
    # What one run cost that day, and that as a share of the dearest run any day -- the bar's width.
    each: Decimal
    width: int


def spend_by_day(connection: psycopg.Connection) -> list[tuple[date, int, Decimal]]:
    """What each day ran and what it spent. A run that has not rested has no cost yet, and counts as none."""
    return connection.execute(SPEND_BY_DAY).fetchall()


def spend_chart(rows: list[tuple[date, int, Decimal]]) -> list[Day]:
    """The burn-down as bars, scaled to the dearest run rather than to the busiest day."""
    each = [spent / runs if runs else Decimal(0) for _, runs, spent in rows]
    dearest = max(each, default=Decimal(0))
    return [
        Day(
            on=day,
            runs=runs,
            spent=spent,
            each=cost.quantize(MONEY),
            # Every day free makes the dearest run free, and a share of nothing is a division by it.
            width=round(100 * cost / dearest) if dearest else 0,
        )
        for (day, runs, spent), cost in zip(rows, each, strict=True)
    ]


def _pair(value: Any) -> tuple[int, int]:
    """A model's [prompt, completion] tokens. Anything that is not that pair is no tokens at all."""
    if not isinstance(value, list) or len(value) != 2:
        return (0, 0)
    halves = [counted(half) for half in value]
    # All or nothing. A pair with one impossible half did not come from the worker, so the half that
    # happens to look like a number is not evidence either -- counting it would draw a bar from it.
    return (halves[0], halves[1]) if halves == list(value) else (0, 0)


def tokens_by_model(connection: psycopg.Connection) -> list[tuple[str, tuple[int, int]]]:
    """Prompt and completion tokens per model, added up over every run that kept them."""
    total: dict[str, tuple[int, int]] = {}
    for (spent,) in connection.execute(TOKENS_BY_MODEL).fetchall():
        if not isinstance(spent, dict):  # a run from before the tokens were kept per model
            continue
        for model, pair in spent.items():
            if not isinstance(model, str) or len(model) > MAX_MODEL_NAME_CHARS:
                continue
            prompt, completion = _pair(pair)
            was = total.get(model, (0, 0))
            total[model] = (was[0] + prompt, was[1] + completion)
    return sorted(total.items())
