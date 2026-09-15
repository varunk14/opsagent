"""
The failure taxonomy: one named category for every run that failed.

"It broke" is not actionable. A run that went wrong is classified from what it left behind
-- its stop reason, its steps and what they returned, what was paid or put to a person --
into exactly one of six fixed categories, by a fixed priority, so the mix over time says
which kind of failure is growing and what to work on:

    context_overflow    the customer's text was too long for the prompt and was cut
    loop                the planner repeated a step, used the whole step budget, or was
                        deferred by the rate limit until a person had to take it
    tool_misuse         a tool that does not run here, arguments the ledger refused, a lookup
                        of an order the ledger does not know, a refund before any lookup
    hallucinated_field  an order id or amount in the extraction or the proposal that appears
                        in neither the customer's message nor any tool result
    wrong_escalation    a person rejected what the agent proposed
    drift               assigned only by the evaluation suite, from its history

The rules here use nothing but the run: no label, no model. The evaluation suite, which
knows what each golden case should have done, adds the label-aware cases on top (paid when a
person should decide, escalated something easy, drift). A run that did what it should has no
category, and neither does one that died of infrastructure -- an outage, an expired lock --
which failure_class and the dead-letter list already name.

The worker writes the category in the same transaction as the step that rests the run, so it
is as durable as the run's state; `python -m app.failures backfill` classifies runs that
rested before the column existed, once.
"""

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

import psycopg

from app.db import connect
from app.graph.prompts import MAX_CUSTOMER_TEXT


class FailureCategory(StrEnum):
    HALLUCINATED_FIELD = "hallucinated_field"
    TOOL_MISUSE = "tool_misuse"
    LOOP = "loop"
    CONTEXT_OVERFLOW = "context_overflow"
    WRONG_ESCALATION = "wrong_escalation"
    DRIFT = "drift"


# The stop reasons the driver and the graph write (app/run_agent.py hand_over, app/graph/nodes.py escalation).
LOOP_REASONS = ("repeated an earlier step", "step budget of", "by the rate limit")
TOOL_MISUSE_REASONS = ("is not a tool this worker runs", "the ledger refused", "could not be read")
CUT_REASON = "characters cut"

# A run has rested here; anywhere else it is still being worked or died of infrastructure.
RESTED = frozenset({"done", "waiting_approval"})

REST = """
    SELECT status, failure_class, state -> 'agent',
           coalesce(state -> 'untrusted' ->> 'subject', ''), coalesce(state -> 'untrusted' ->> 'body', '')
      FROM runs
     WHERE id = %s
"""
REFUNDS = "SELECT amount_paise FROM refunds WHERE run_id = %s ORDER BY id"
LAST_APPROVAL = """
    SELECT (action -> 'args' ->> 'amount_paise')::bigint, status
      FROM approvals
     WHERE run_id = %s
     ORDER BY id DESC
     LIMIT 1
"""
SET_CATEGORY = "UPDATE runs SET failure_category = %s WHERE id = %s"
UNCLASSIFIED = """
    SELECT id FROM runs
     WHERE status IN ('done', 'waiting_approval') AND failure_category IS NULL
     ORDER BY created_at
"""
MIX_BY_WEEK = """
    SELECT date_trunc('week', created_at)::date, failure_category, count(*)
      FROM runs
     WHERE failure_category IS NOT NULL
     GROUP BY 1, 2
     ORDER BY 1, 2
"""


@dataclass(frozen=True)
class RunRest:
    """What a run left behind, read from the database, with nothing a label would add."""

    status: str
    failure_class: str | None
    failure: str | None
    steps: Sequence[Mapping[str, Any]]
    proposal: Mapping[str, Any] | None
    extraction: Mapping[str, Any] | None
    refunds_paise: tuple[int, ...]
    approval_paise: int | None
    approval_status: str | None
    message_text: str


def numbers_in(text: str) -> set[str]:
    """Every number written in `text`, and each one taken as rupees and turned into paise."""
    plain = {found.replace(",", "") for found in re.findall(r"\d[\d,]*", text)}
    return plain | {f"{number}00" for number in plain}


def known_numbers(rest: RunRest) -> set[str]:
    """Every number the run had a source for: the customer's message and what its tools returned."""
    known = numbers_in(rest.message_text)
    for step in rest.steps:
        known |= numbers_in(json.dumps(step.get("result"), ensure_ascii=False))
    return known


def invented(rest: RunRest) -> bool:
    """An order id or amount the agent stated that appears nowhere it could have read it."""
    known = known_numbers(rest)
    stated = []
    if rest.extraction:
        stated.append(rest.extraction.get("order_id"))
    if rest.proposal:
        args = rest.proposal.get("args") or {}
        stated += [args.get("order_id"), args.get("amount_paise")]
    return any(value is not None and str(value) not in known for value in stated)


def classify(rest: RunRest) -> FailureCategory | None:
    """One category for a run that failed, by a fixed priority; None for a run that did not."""
    if rest.status not in RESTED:
        return None
    failure = rest.failure or ""
    if len(rest.message_text) > MAX_CUSTOMER_TEXT or CUT_REASON in failure:
        return FailureCategory.CONTEXT_OVERFLOW
    if any(reason in failure for reason in LOOP_REASONS):
        return FailureCategory.LOOP
    lookups = [step for step in rest.steps if step.get("tool") == "get_order"]
    if (
        any(reason in failure for reason in TOOL_MISUSE_REASONS)
        or any(isinstance(step.get("result"), Mapping) and "error" in step["result"] for step in lookups)
        or (rest.proposal is not None and rest.proposal.get("tool") == "issue_refund" and not lookups)
    ):
        return FailureCategory.TOOL_MISUSE
    if invented(rest):
        return FailureCategory.HALLUCINATED_FIELD
    if rest.approval_status == "rejected":
        return FailureCategory.WRONG_ESCALATION
    return None


def rest_of(connection: psycopg.Connection, run_id: UUID) -> RunRest:
    """What one run left behind. A run that is not in the database is refused by id."""
    row = connection.execute(REST, (run_id,)).fetchone()
    if row is None:
        raise LookupError(f"no run {run_id}")
    status, failure_class, stored, subject, body = row
    agent: dict[str, Any] = stored or {}
    approval = connection.execute(LAST_APPROVAL, (run_id,)).fetchone()
    return RunRest(
        status=status,
        failure_class=failure_class,
        failure=agent.get("failure"),
        steps=list(agent.get("steps") or []),
        proposal=agent.get("proposal"),
        extraction=agent.get("extraction"),
        refunds_paise=tuple(amount for (amount,) in connection.execute(REFUNDS, (run_id,)).fetchall()),
        approval_paise=approval[0] if approval else None,
        approval_status=approval[1] if approval else None,
        message_text=f"{subject}\n\n{body}",
    )


def record_category(connection: psycopg.Connection, run_id: UUID) -> FailureCategory | None:
    """Classify a run that has just rested and write its category, in the caller's transaction."""
    category = classify(rest_of(connection, run_id))
    connection.execute(SET_CATEGORY, (category.value if category else None, run_id))
    return category


def backfill(connection: psycopg.Connection) -> int:
    """Classify every rested run without a category. Returns how many got one; running it again writes nothing new."""
    written = 0
    for (run_id,) in connection.execute(UNCLASSIFIED).fetchall():
        written += record_category(connection, run_id) is not None
    return written


def mix_by_week(connection: psycopg.Connection) -> list[tuple[Any, str, int]]:
    """How many runs rested in each category, per week they were created."""
    return connection.execute(MIX_BY_WEEK).fetchall()


def main(argv: list[str]) -> int:  # pragma: no cover - the operator's command line
    parser = argparse.ArgumentParser(prog="python -m app.failures", description="The failure taxonomy over stored runs.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("backfill", help="classify every rested run that has no category yet")
    commands.add_parser("show", help="the failure mix per week")
    arguments = parser.parse_args(argv[1:])

    with connect() as connection:
        if arguments.command == "backfill":
            print(f"  classified {backfill(connection)} run(s)")
            return 0
        rows = mix_by_week(connection)
        if not rows:
            print("  no failed runs yet")
        for week, category, count in rows:
            print(f"  {week}  {category:<20} {count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
