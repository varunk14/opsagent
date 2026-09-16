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
from app.failures import is_count

# A model name is a name. Ollama's longest is well inside this; an essay here came from somewhere else.
MAX_MODEL_NAME_CHARS = 64
# Six places, as `runs.cost_usd` and the budgets are stored.
MONEY = Decimal("0.000001")

# What one page view is willing to read. `runs` only grows, and this screen is where a person
# approves a refund while something is going wrong -- a burn-down that reads all of history would get
# slower every day and hold a worker thread the approvals page needs. The tokens are summed in Python
# rather than in SQL on purpose: the guards below are why that column is safe to read at all, and
# `sum()` inside Postgres would believe whatever it found there.
SHOWN_DAYS = 30
MAX_RUNS_READ = 10_000
# One row per model, and the names are keys in JSON we did not write. The biggest spenders are the
# ones worth drawing; a long tail of one-token models is noise at best.
MAX_MODELS = 20

SPEND_BY_DAY = """
    SELECT created_at::date AS day, count(*), sum(cost_usd)
      FROM runs
     WHERE created_at >= now() - make_interval(days => %s)
     GROUP BY day
     ORDER BY day
"""

TOKENS_BY_MODEL = """
    SELECT state -> 'agent' -> 'tokens_by_model'
      FROM runs
     WHERE created_at >= now() - make_interval(days => %s)
     ORDER BY created_at DESC
     LIMIT %s
"""


@dataclass(frozen=True)
class Day:
    """One day of the burn-down, as the page draws it."""

    on: date
    runs: int
    spent: Decimal
    # What one run cost that day, and that as a share of the dearest run any day -- the bar's width.
    each: Decimal
    width: int


def spend_by_day(connection: psycopg.Connection, *, days: int = SHOWN_DAYS) -> list[tuple[date, int, Decimal]]:
    """
    What each of the last `days` days ran, and what it spent.

    A day holds the runs that *started* on it. `cost_usd` is zero until a run rests, so a run still
    in flight is a free run in its day's average and pulls the figure down -- the mirror of the
    distortion cost-per-run exists to avoid. That is deliberate: the alternative, leaving unfinished
    runs out, would make a day of nothing but stuck runs look like a day of no work at all.
    """
    return connection.execute(SPEND_BY_DAY, (days,)).fetchall()


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
    # All or nothing. A pair with one impossible half did not come from the worker, so the half that
    # happens to look like a number is not evidence either -- counting it would draw a bar from it.
    # Asked of each half directly: a rejected half can be numerically equal to an accepted one
    # (0 == 0.0 == False), so deciding by what the guard *returned* would let its partner through.
    if not all(is_count(half) for half in value):
        return (0, 0)
    return (value[0], value[1])


def tokens_by_model(
    connection: psycopg.Connection,
    *,
    days: int = SHOWN_DAYS,
    limit: int = MAX_RUNS_READ,
    most: int = MAX_MODELS,
) -> list[tuple[str, tuple[int, int]]]:
    """
    Prompt and completion tokens per model, over the same window the chart covers.

    Newest runs first, so the cap loses the oldest rather than an arbitrary slice, and at most
    `most` models come back -- a row per name is a row whoever filled that column chose to ask for.
    """
    total: dict[str, tuple[int, int]] = {}
    for (spent,) in connection.execute(TOKENS_BY_MODEL, (days, limit)).fetchall():
        if not isinstance(spent, dict):  # a run from before the tokens were kept per model
            continue
        for model, pair in spent.items():
            if not isinstance(model, str) or len(model) > MAX_MODEL_NAME_CHARS:
                continue
            prompt, completion = _pair(pair)
            was = total.get(model, (0, 0))
            total[model] = (was[0] + prompt, was[1] + completion)
    biggest = sorted(total.items(), key=lambda kept: sum(kept[1]), reverse=True)[:most]
    return sorted(biggest)
