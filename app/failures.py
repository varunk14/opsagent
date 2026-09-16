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
                        of an order the ledger does not know, a refund for an order that was
                        never looked up
    hallucinated_field  an order id or amount in the extraction or the proposal that appears
                        in neither the customer's message, nor the policy it was shown, nor
                        any tool result
    wrong_escalation    a person rejected what the agent proposed
    drift               assigned only by the evaluation suite, from its history

The rules here use nothing but the run: no label, no model. The evaluation suite, which
knows what each golden case should have done, adds the label-aware cases on top (paid when a
person should decide, escalated something easy, drift). A run that did what it should has no
category, and neither does one that died of infrastructure -- an outage, an expired lock --
which failure_class and the dead-letter list already name.

The worker writes the category in the same transaction as the step that rests the run, so it
is as durable as the run's state -- and inside a savepoint, so a bug in naming a failure can
never undo the refund or the decision that transaction holds. `python -m app.failures
backfill` classifies runs that rested before the column existed, once.

Known limit: the hallucination rule trusts every number in the customer's message as a
possible source, so a message padded with numbers can hide an invented amount from it. The
category is a diagnostic; the evaluation suite still scores the wrong outcome.
"""

import argparse
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg

from app.db import connect
from app.graph.prompts import MAX_CUSTOMER_TEXT

log = logging.getLogger(__name__)


class FailureCategory(StrEnum):
    HALLUCINATED_FIELD = "hallucinated_field"
    TOOL_MISUSE = "tool_misuse"
    LOOP = "loop"
    CONTEXT_OVERFLOW = "context_overflow"
    WRONG_ESCALATION = "wrong_escalation"
    DRIFT = "drift"


# What we would fix for each kind, specific to this system. Shown beside the chart.
FIXES: tuple[tuple[FailureCategory, str], ...] = (
    (
        FailureCategory.HALLUCINATED_FIELD,
        "Tighten extraction: an order id or amount must be quoted from the message or a lookup; refuse the proposal otherwise.",
    ),
    (
        FailureCategory.TOOL_MISUSE,
        "Check arguments before proposing: a refund must equal one of the order's ledger charges; the ledger already refuses more.",
    ),
    (
        FailureCategory.LOOP,
        "Give the planner a way to decide: when it repeats a lookup whose result is shown, ask once more with that result marked, then hand over.",
    ),
    (
        FailureCategory.CONTEXT_OVERFLOW,
        "Chunk or summarise a long message before the prompt; today anything over the cap is cut.",
    ),
    (
        FailureCategory.WRONG_ESCALATION,
        "Decide from policy conditions in code, not from confidence: a duplicate needs two ledger charges; change-of-mind and damaged items go to a person.",
    ),
    (
        FailureCategory.DRIFT,
        "Pin the model version and the prompt hashes; record again and compare each case against the last accepted baseline.",
    ),
)

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
    # The policy passages the run was planned with: a source of amounts like the message and the tools.
    policy: Sequence[str] = ()


def numbers_in(text: str) -> set[str]:
    """
    Every number written in `text`, as written and as paise.

    A whole number may be an order id or rupees, so both it and it times a hundred count;
    "99.50" is rupees and paise, so it counts as 9950.
    """
    known: set[str] = set()
    for found in re.findall(r"\d[\d,]*(?:\.\d{1,2})?", text):
        written = found.replace(",", "")
        if "." in written:
            known.add(str(int((Decimal(written) * 100).to_integral_value())))
        else:
            known.update((written, f"{written}00"))
    return known


def known_numbers(rest: RunRest) -> set[str]:
    """Every number the run had a source for: the customer's message, the policy it was shown, what its tools returned."""
    known = numbers_in(rest.message_text)
    for passage in rest.policy:
        known |= numbers_in(passage)
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


def misused_a_tool(rest: RunRest, failure: str) -> bool:
    """The wrong tool, arguments the ledger refused, a lookup that found nothing, or a refund for an order never looked up."""
    lookups = [step for step in rest.steps if step.get("tool") == "get_order"]
    looked_up = {str((step.get("args") or {}).get("order_id")) for step in lookups}
    proposal = rest.proposal or {}
    return (
        any(reason in failure for reason in TOOL_MISUSE_REASONS)
        or any(isinstance(step.get("result"), Mapping) and "error" in step["result"] for step in lookups)
        or (proposal.get("tool") == "issue_refund" and str((proposal.get("args") or {}).get("order_id")) not in looked_up)
    )


def classify(rest: RunRest) -> FailureCategory | None:
    """One category for a run that failed, by a fixed priority; None for a run that did not."""
    if rest.status not in RESTED:
        return None
    failure = rest.failure or ""
    if len(rest.message_text) > MAX_CUSTOMER_TEXT or CUT_REASON in failure:
        return FailureCategory.CONTEXT_OVERFLOW
    if any(reason in failure for reason in LOOP_REASONS):
        return FailureCategory.LOOP
    if misused_a_tool(rest, failure):
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
        policy=tuple(agent.get("policy") or []),
    )


def record_category(connection: psycopg.Connection, run_id: UUID) -> FailureCategory | None:
    """
    Classify a run that has just rested and write its category, in the caller's transaction.

    Inside a savepoint: naming a failure is a diagnostic, and a bug in it must never undo the
    refund or the decision the caller's transaction holds. What goes wrong is logged and the
    category is left empty.
    """
    try:
        with connection.transaction():
            category = classify(rest_of(connection, run_id))
            connection.execute(SET_CATEGORY, (category.value if category else None, run_id))
            return category
    except Exception:  # whatever it was, the run's own transaction must survive it
        log.warning("run %s could not be classified; its category is left empty", run_id, exc_info=True)
        return None


def backfill(connection: psycopg.Connection) -> int:
    """Classify every rested run without a category. Returns how many got one; running it again writes nothing new."""
    written = 0
    for (run_id,) in connection.execute(UNCLASSIFIED).fetchall():
        written += record_category(connection, run_id) is not None
    return written


def mix_by_week(connection: psycopg.Connection) -> list[tuple[date, str, int]]:
    """How many runs rested in each category, per week they were created."""
    return connection.execute(MIX_BY_WEEK).fetchall()


def failure_chart(rows: Sequence[tuple[date, str, int]]) -> list[tuple[date, list[tuple[str, int, int]]]]:
    """The mix per week as bars: each category's count and its width as a share of the largest count anywhere."""
    largest = max((count for _, _, count in rows), default=0)
    weeks: dict[date, list[tuple[str, int, int]]] = {}
    for week, category, count in rows:
        weeks.setdefault(week, []).append((category, count, round(100 * count / largest)))
    return list(weeks.items())


GOLDEN_HISTORY = Path(__file__).resolve().parent.parent / "evals" / "history.jsonl"
# One accepted baseline is well under a kilobyte, so this is thousands of them: past it the file is not a history.
HISTORY_LIMIT_BYTES = 1_000_000


@dataclass(frozen=True)
class Accepted:
    """One accepted baseline of the golden set, as the trend draws it."""

    on: str
    code: str
    completed: int
    cases: int
    mix: list[tuple[str, int, int]]


def golden_trend(path: Path = GOLDEN_HISTORY) -> list[Accepted]:
    """
    The failure mix of every accepted baseline, oldest first, as bars.

    Live runs say what is failing here and now; this says what the golden set does with each
    version of the agent, so a category that a change made worse is visible before a deploy.
    The file is the append-only history `python -m evals accept` writes. It is read as
    evidence, never trusted: a line that is not an accepted baseline is skipped, and a missing
    file is simply no trend -- the screen must not fall over because the history is absent,
    half-written, or not text at all.
    """
    try:
        if path.stat().st_size > HISTORY_LIMIT_BYTES:
            log.warning("the history at %s is larger than a history can be; the trend is left empty", path)
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):  # unreadable, or bytes that are not UTF-8: no trend, not a broken page
        return []
    accepted: list[tuple[dict[str, Any], list[tuple[str, int]]]] = []
    for line in lines:
        try:
            found = json.loads(line)
        except ValueError:
            continue
        mix = found.get("failure_mix") if isinstance(found, dict) else None
        if isinstance(mix, dict):
            accepted.append((found, [(category.value, counted(mix.get(category.value))) for category in FailureCategory]))
    # The widths are scaled to the largest count the trend actually shows, so every bar on the page means the same thing.
    largest = max((count for _, mix in accepted for _, count in mix), default=0)
    return [
        Accepted(
            on=str(found.get("accepted_on", "")),
            code=str(found.get("code", "")),
            completed=counted(found.get("completed")),
            cases=counted(found.get("cases")),
            mix=[(category, count, round(100 * count / largest) if largest else 0) for category, count in mix],
        )
        for found, mix in accepted
    ]


def counted(value: Any) -> int:
    """A count read from a history line: anything that is not a whole number of things counts as none."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


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
