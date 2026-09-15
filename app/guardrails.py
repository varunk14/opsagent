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

Nothing here commits; the caller owns the transaction.

Run:  .venv/bin/python -m app.guardrails show
      .venv/bin/python -m app.guardrails set --limit-rupees 10000 --by asha
      .venv/bin/python -m app.guardrails set --min-confidence 0.90 --by asha
"""

import argparse
import sys
from dataclasses import dataclass
from decimal import Decimal

import psycopg

from app.contracts import ProposedAction
from app.db import connect

JUDGED = "issue_refund"

LOAD = "SELECT auto_refund_limit_paise, min_confidence FROM guardrails WHERE singleton"

SET_LIMITS = """
    UPDATE guardrails
       SET auto_refund_limit_paise = coalesce(%(limit)s::bigint, auto_refund_limit_paise),
           min_confidence = coalesce(%(confidence)s::numeric, min_confidence),
           updated_by = %(by)s,
           updated_at = now()
     WHERE singleton
    RETURNING auto_refund_limit_paise, min_confidence
"""

TWO_PLACES = Decimal("0.01")


@dataclass(frozen=True)
class Guardrails:
    auto_refund_limit_paise: int
    min_confidence: Decimal


@dataclass(frozen=True)
class Verdict:
    """Whether an action may run without a person, and if not, what to tell them."""

    runs: bool
    reason: str | None


def rupees(paise: int) -> str:
    """360000 -> 'Rs 3,600'; paise are shown only when there are some."""
    whole, fraction = divmod(paise, 100)
    return f"Rs {whole:,}" + (f".{fraction:02d}" if fraction else "")


def _shown(confidence: Decimal) -> str:
    """At least two places, and never rounded: 0.849 must not read as 0.85."""
    return str(confidence.quantize(TWO_PLACES)) if confidence == confidence.quantize(TWO_PLACES) else str(confidence)


def judge(action: ProposedAction, guardrails: Guardrails) -> Verdict:
    """Decide whether a refund runs on its own. Every reason that applies is given."""
    if action.tool != JUDGED:
        raise ValueError(f"only {JUDGED} is judged by the guardrail, not {action.tool}")

    amount = action.args["amount_paise"]
    reasons = []
    if not amount < guardrails.auto_refund_limit_paise:
        reasons.append(
            f"{rupees(amount)} is not under the {rupees(guardrails.auto_refund_limit_paise)} "
            "limit for automatic refunds"
        )
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
    return Guardrails(auto_refund_limit_paise=row[0], min_confidence=row[1])


def set_limits(
    connection: psycopg.Connection,
    *,
    limit_paise: int | None = None,
    min_confidence: Decimal | None = None,
    by: str,
) -> Guardrails:
    """
    Change one limit or both, recording who did it. Returns what is now in force.

    Bad values are refused here rather than left to the database, because the
    database would not refuse all of them: numeric(3,2) rounds 0.855 to 0.86.
    """
    if not by.strip():
        raise ValueError("say who is making the change")
    if limit_paise is None and min_confidence is None:
        raise ValueError("nothing to change: give a limit, a confidence, or both")

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

    row = connection.execute(
        SET_LIMITS, {"limit": limit_paise, "confidence": min_confidence, "by": by.strip()}
    ).fetchone()
    if row is None:  # pragma: no cover - the row cannot be deleted
        raise LookupError("the guardrails row is missing; apply the migrations")
    return Guardrails(auto_refund_limit_paise=row[0], min_confidence=row[1])


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
