"""
The guardrail: which refunds may run on their own, and which need a person.

Concept 2.10. The limits are numbers in Postgres compared in Python, and never
words in the prompt: a model told "refunds over Rs 5,000 need approval" can be
talked out of it by the email it is reading, and a comparison cannot. `judge` is
therefore pure -- no I/O, no clock, no model -- and the numbers it is given are
read fresh by the caller for every decision, so a change takes effect on the next
refund with no code change and no restart.

"Under Rs 5,000" is strict: exactly the limit needs a person, and a limit of zero
makes every refund manual. That is the kill switch.

Being small enough is not reason enough to pay. A refund that would run on its own
must also be owed, which `justified` decides from what the run established rather
than from what the model says about itself: the message read as a duplicate charge,
and the ledger showing two charges of the same amount on that order. A refund that
fails those conditions is refused outright and goes to a person with nothing to
approve, since there is no payment the agent can stand behind. Confidence is the
model's opinion of its own work, and an email it is reading can change it; the
ledger cannot be talked round either.

Nothing here commits; the caller owns the transaction.

Run:  .venv/bin/python -m app.guardrails show
      .venv/bin/python -m app.guardrails set --limit-rupees 10000 --by asha
      .venv/bin/python -m app.guardrails set --min-confidence 0.90 --by asha
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import psycopg

from app.contracts import Intent, ProposedAction
from app.db import connect

JUDGED = "issue_refund"
LOOKUP = "get_order"

# The reason a refusal gives is read by the person who picks the case up, so it says what the
# message was taken to be in their words, never the enum's.
READING_OF = {
    Intent.DUPLICATE_CHARGE: "a duplicate charge",
    Intent.REFUND_REQUEST: "a refund request",
    Intent.ORDER_STATUS: "a question about an order",
    Intent.OTHER: "something else",
}

LOAD = (
    "SELECT auto_refund_limit_paise, min_confidence, max_tokens_per_run, max_cost_usd_per_run, "
    "max_seconds_per_run FROM guardrails WHERE singleton"
)

SET_LIMITS = """
    UPDATE guardrails
       SET auto_refund_limit_paise = coalesce(%(limit)s::bigint, auto_refund_limit_paise),
           min_confidence = coalesce(%(confidence)s::numeric, min_confidence),
           max_tokens_per_run = coalesce(%(max_tokens)s::bigint, max_tokens_per_run),
           max_cost_usd_per_run = coalesce(%(max_cost)s::numeric, max_cost_usd_per_run),
           max_seconds_per_run = coalesce(%(max_seconds)s::integer, max_seconds_per_run),
           updated_by = %(by)s,
           updated_at = now()
     WHERE singleton
    RETURNING auto_refund_limit_paise, min_confidence, max_tokens_per_run, max_cost_usd_per_run,
              max_seconds_per_run
"""

TWO_PLACES = Decimal("0.01")


@dataclass(frozen=True)
class Budgets:
    """
    What one run may spend before a person takes it.

    Seconds are seconds the agent spent working -- the latency of its own model calls, added up --
    never wall-clock since the message arrived. A run waiting for somebody to approve a refund can
    sit for days, and charging that against a budget would hand over every refund anyone took a
    lunch break over.

    Zero means no ceiling, the opposite of `auto_refund_limit_paise` where zero is the kill switch.
    A budget that stopped every run the moment it was set to zero would make the safe way to switch
    a budget off indistinguishable from the harshest setting there is.
    """

    max_tokens_per_run: int
    max_cost_usd_per_run: Decimal
    max_seconds_per_run: int


def over_budget(budgets: Budgets, tokens: int, cost_usd: Decimal, seconds: int) -> str | None:
    """
    Which ceiling this run has passed, said so a person reading it knows what to do, or None.

    The ceiling is what a run may spend, not the first amount it may not: a run that lands exactly
    on it has stayed within what it was given.
    """
    if budgets.max_tokens_per_run and tokens > budgets.max_tokens_per_run:
        return f"{tokens:,} tokens spent, past the {budgets.max_tokens_per_run:,} this run was given"
    if budgets.max_cost_usd_per_run and cost_usd > budgets.max_cost_usd_per_run:
        return f"${cost_usd:f} spent, past the ${budgets.max_cost_usd_per_run:f} this run was given"
    if budgets.max_seconds_per_run and seconds > budgets.max_seconds_per_run:
        return f"{seconds} seconds of model time, past the {budgets.max_seconds_per_run} this run was given"
    return None


@dataclass(frozen=True)
class Guardrails:
    auto_refund_limit_paise: int
    min_confidence: Decimal
    budgets: Budgets


@dataclass(frozen=True)
class Verdict:
    """
    Whether an action may run without a person, and if not, what to tell them.

    `refused` means the guardrail refuses the proposal itself: a person takes the case and there
    is nothing to approve. A refund that is merely large is still one the agent stands behind, so
    it goes to the approval queue for someone to say yes or no to. A refund whose conditions were
    never established is different in kind -- there is no payment to approve, only a case to look
    at -- so it is handed over instead of queued.
    """

    runs: bool
    reason: str | None
    refused: bool = False


@dataclass(frozen=True)
class Evidence:
    """What the run established about the order before it proposed to pay: the reading, and the ledger."""

    intent: Intent | None
    charges_paise: tuple[int, ...] = ()


def evidence_of(intent: Intent | None, steps: Sequence[Mapping[str, Any]], order_id: str) -> Evidence:
    """The evidence a run's own steps give about one order. A lookup of any other order says nothing about it."""
    charges: list[int] = []
    for step in steps:
        if step.get("tool") != LOOKUP:
            continue
        result = step.get("result") or {}
        if str(result.get("order_id", "")) == order_id:
            charges += [amount for amount in result.get("charges_paise") or [] if isinstance(amount, int)]
    return Evidence(intent=intent, charges_paise=tuple(charges))


def justified(action: ProposedAction, evidence: Evidence) -> Verdict:
    """
    Whether the conditions for paying this refund without a person hold in the ledger.

    This is the check the guardrail was missing. Judging a refund on its amount and the model's
    own confidence means a confident model can have any small refund paid by asserting it is
    owed -- and a model reading a customer's email is exactly the thing an email can talk round.
    Confidence is not evidence. An automatic payment needs the duplicate to be real: the message
    read as a duplicate charge, and the ledger showing exactly one amount charged twice on that
    order, which is what the refund pays back. Being charged more than once is not being charged
    twice -- an order billed for the item and then for shipping has two charges and no duplicate,
    and taking that as one would leave the whole decision resting on the model's reading of an
    email again. Nor is an order with two different amounts each charged twice: the ledger does
    not say which of them the customer means, and only their message would, so a person reads it.

    The amount is deliberately not a condition. The ledger refuses a refund larger than the order
    was charged, the limit bounds what runs without a person, and refunds split into parts are
    judged as the total they add up to, so requiring the amount to equal one charge exactly would
    refuse legitimate partial refunds and add no safety.

    Nothing established here is forbidden -- it is a person's to decide.
    """
    order_id = action.args["order_id"]
    if evidence.intent != Intent.DUPLICATE_CHARGE:
        return Verdict(
            runs=False,
            reason=f"this reads as {READING_OF[evidence.intent] if evidence.intent else 'a message that was never read'}, "
            "not a duplicate charge, so a person decides whether it is owed",
            refused=True,
        )
    if not evidence.charges_paise:
        return Verdict(
            runs=False,
            reason=f"order {order_id} was never looked up, so nothing confirms a duplicate charge",
            refused=True,
        )
    if len(evidence.charges_paise) < 2:
        return Verdict(
            runs=False,
            reason=f"order {order_id} was charged once, so there is no duplicate to refund",
            refused=True,
        )
    duplicated = {amount for amount in evidence.charges_paise if evidence.charges_paise.count(amount) > 1}
    if not duplicated:
        return Verdict(
            runs=False,
            reason=f"order {order_id} was charged {len(evidence.charges_paise)} times but no two charges are "
            "the same amount, so none of them is a duplicate of another",
            refused=True,
        )
    if len(duplicated) > 1:
        return Verdict(
            runs=False,
            reason=f"order {order_id} has more than one amount charged twice "
            f"({', '.join(rupees(amount) for amount in sorted(duplicated))}), and nothing here says which of "
            "them the customer means, so a person decides",
            refused=True,
        )
    if action.args["amount_paise"] not in duplicated:
        return Verdict(
            runs=False,
            reason=f"{rupees(action.args['amount_paise'])} is not one of the charges duplicated on order "
            f"{order_id} ({', '.join(rupees(amount) for amount in sorted(duplicated))}), so it is not what "
            "this duplicate owes back",
            refused=True,
        )
    return Verdict(runs=True, reason=None)


def rupees(paise: int) -> str:
    """360000 -> 'Rs 3,600'; paise are shown only when there are some."""
    whole, fraction = divmod(paise, 100)
    return f"Rs {whole:,}" + (f".{fraction:02d}" if fraction else "")


def _shown(confidence: Decimal) -> str:
    """At least two places, and never rounded: 0.849 must not read as 0.85."""
    return str(confidence.quantize(TWO_PLACES)) if confidence == confidence.quantize(TWO_PLACES) else str(confidence)


def judge(action: ProposedAction, guardrails: Guardrails, already_refunded_paise: int = 0) -> Verdict:
    """
    Decide whether a refund runs on its own. Every reason that applies is given.

    The limit applies to what the order would have had back in total, so a refund
    split into parts under the limit is judged as the whole it adds up to.
    """
    if action.tool != JUDGED:
        raise ValueError(f"only {JUDGED} is judged by the guardrail, not {action.tool}")
    if already_refunded_paise < 0:
        raise ValueError("already refunded paise cannot be negative")

    amount = action.args["amount_paise"]
    total = already_refunded_paise + amount
    limit = rupees(guardrails.auto_refund_limit_paise)
    reasons = []
    if not total < guardrails.auto_refund_limit_paise:
        if already_refunded_paise:
            reasons.append(
                f"{rupees(amount)} would bring refunds on order {action.args['order_id']} to {rupees(total)}, "
                f"not under the {limit} limit for automatic refunds"
            )
        else:
            reasons.append(f"{rupees(amount)} is not under the {limit} limit for automatic refunds")
    if action.confidence < guardrails.min_confidence:
        reasons.append(
            f"confidence {_shown(action.confidence)} is below the "
            f"{_shown(guardrails.min_confidence)} needed for automatic refunds"
        )
    return Verdict(runs=not reasons, reason="; ".join(reasons) or None)


def load(connection: psycopg.Connection) -> Guardrails:
    """The limits in force now. With no row there is no limit, so that is an error, never a default."""
    row = connection.execute(LOAD).fetchone()
    if row is None:  # pragma: no cover - the row cannot be deleted
        raise LookupError("the guardrails row is missing; apply the migrations")
    return in_force(row)


def in_force(row: Sequence[Any]) -> Guardrails:
    """One row of the guardrails table as the limits it stands for."""
    return Guardrails(
        auto_refund_limit_paise=row[0],
        min_confidence=row[1],
        budgets=Budgets(max_tokens_per_run=row[2], max_cost_usd_per_run=row[3], max_seconds_per_run=row[4]),
    )


def set_limits(
    connection: psycopg.Connection,
    *,
    limit_paise: int | None = None,
    min_confidence: Decimal | None = None,
    max_tokens_per_run: int | None = None,
    max_cost_usd_per_run: Decimal | None = None,
    max_seconds_per_run: int | None = None,
    by: str,
) -> Guardrails:
    """
    Change one limit or both, recording who did it. Returns what is now in force.

    Bad values are refused here rather than left to the database, because the
    database would not refuse all of them: numeric(3,2) rounds 0.855 to 0.86.
    """
    if not by.strip():
        raise ValueError("say who is making the change")
    changes = (limit_paise, min_confidence, max_tokens_per_run, max_cost_usd_per_run, max_seconds_per_run)
    if all(change is None for change in changes):
        raise ValueError("nothing to change: give a limit, a confidence, or a budget")

    if limit_paise is not None:
        # bool is an int, and True is not an amount of paise.
        if isinstance(limit_paise, bool) or not isinstance(limit_paise, int):
            raise ValueError("the limit must be a whole number of paise")
        if limit_paise < 0:
            raise ValueError("the limit must be at least 0 paise")

    if min_confidence is not None:
        if not isinstance(min_confidence, Decimal):
            raise ValueError("min_confidence must be a Decimal such as Decimal('0.85'), so it stays exact")
        if not min_confidence.is_finite() or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if min_confidence != min_confidence.quantize(TWO_PLACES):
            raise ValueError("min_confidence has at most two decimal places")

    for name, ceiling in (("max_tokens_per_run", max_tokens_per_run), ("max_seconds_per_run", max_seconds_per_run)):
        if ceiling is None:
            continue
        # bool is an int, and True is not a number of tokens or seconds.
        if isinstance(ceiling, bool) or not isinstance(ceiling, int):
            raise ValueError(f"{name} must be a whole number")  # noqa: TRY004 - the CLI reports it as a bad value
        if ceiling < 0:
            raise ValueError(f"{name} must be at least 0, where 0 means no ceiling")

    if max_cost_usd_per_run is not None:
        if not isinstance(max_cost_usd_per_run, Decimal):
            raise ValueError("max_cost_usd_per_run must be a Decimal such as Decimal('0.002'), so it stays exact")
        if not max_cost_usd_per_run.is_finite() or max_cost_usd_per_run < 0:
            raise ValueError("max_cost_usd_per_run must be at least 0, where 0 means no ceiling")

    row = connection.execute(
        SET_LIMITS,
        {
            "limit": limit_paise,
            "confidence": min_confidence,
            "max_tokens": max_tokens_per_run,
            "max_cost": max_cost_usd_per_run,
            "max_seconds": max_seconds_per_run,
            "by": by.strip(),
        },
    ).fetchone()
    if row is None:  # pragma: no cover - the row cannot be deleted
        raise LookupError("the guardrails row is missing; apply the migrations")
    return in_force(row)


def main(argv: list[str]) -> int:  # pragma: no cover - the operator's command line
    parser = argparse.ArgumentParser(prog="python -m app.guardrails")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show", help="print the limits in force")
    change = commands.add_parser("set", help="change a limit")
    change.add_argument("--limit-rupees", type=Decimal, help="refunds strictly under this run on their own; 0 stops them")
    change.add_argument("--min-confidence", type=Decimal, help="refunds below this confidence need a person")
    change.add_argument("--by", required=True, help="who is making the change")
    arguments = parser.parse_args(argv[1:])

    with connect() as connection:
        if arguments.command == "set":
            limit = None
            if arguments.limit_rupees is not None:
                paise = arguments.limit_rupees * 100
                if paise != paise.to_integral_value():
                    print("  --limit-rupees has at most two decimal places")
                    return 2
                limit = int(paise)
            try:
                guardrails = set_limits(
                    connection, limit_paise=limit, min_confidence=arguments.min_confidence, by=arguments.by
                )
            except ValueError as refused:
                print(f"  refused: {refused}")
                return 2
            connection.commit()
        else:
            guardrails = load(connection)

    print(f"  refunds under {rupees(guardrails.auto_refund_limit_paise)} run on their own")
    print(f"  refunds below confidence {_shown(guardrails.min_confidence)} need a person")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
