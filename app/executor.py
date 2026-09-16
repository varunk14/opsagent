"""
Running a proposed tool call for real, exactly once per operation.

Every call is keyed by the operation it performs -- run, step, tool -- never by
the attempt, for the reason experiments/prevent_duplicate_refunds.py shows: a key
made fresh on each attempt is one the ledger has never seen, so it stops nothing.

Claiming the key and doing the work share one transaction. The key is claimed
with INSERT ... ON CONFLICT DO NOTHING, so a second caller racing on the same key
waits on the unique index until the first commits, then finds the row and
replays its stored result. If the tool fails, the key goes with the rollback and
the operation can be tried again; there is no state in which the work happened
but the key was not recorded, or the reverse.

A repeat must be the same operation. A key that comes back with other arguments,
another tool or another run is refused rather than replayed, because replaying
would report a refund that was never made for what was asked.

An order is reachable only by the run whose sender placed it. Anyone can email
in quoting a guessed order number, so another customer's order is answered
exactly as a missing one is: nothing confirms it exists. The sender is the
address the message came from, which a forged From header can still claim. What
that could move without a person is bounded by the guardrail's limit on the
order's total refunds (app/guardrails.py), and by the ledger cap: refunds go back
against the order's own charges, never above them.

Refusals a person or the planner should see -- an unknown order, a refund above
what was charged -- are returned as data and stored like any result, so the same
operation gets the same answer every time it is asked.

Only the ledger tools run here. search_policy is the graph's retrieve step.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from app.contracts import ProposedAction

type JsonObject = dict[str, Any]
type Tool = Callable[[psycopg.Connection, UUID, str, JsonObject], JsonObject]

CLAIM_KEY = """
    INSERT INTO tool_calls (idempotency_key, run_id, tool, args)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (idempotency_key) DO NOTHING
    RETURNING idempotency_key
"""

# What the customer actually wrote, which is the only place an order number is allowed to come from.
WRITTEN = """
    SELECT coalesce(state -> 'untrusted' ->> 'subject', ''), coalesce(state -> 'untrusted' ->> 'body', '')
      FROM runs
     WHERE id = %s
"""

# Mail addresses are compared without regard to case; a run with no sender matches nothing.
OWNED_ORDER = """
    SELECT o.customer_email, o.amount_paise, o.status
      FROM orders o
      JOIN runs r ON r.id = %s
     WHERE o.id = %s
       AND lower(o.customer_email) = lower(r.state -> 'untrusted' ->> 'sender')
"""

# The constraint name the cap trigger raises with (migrations/004). Only this
# violation is a refusal; any other CHECK on refunds is a bug and propagates.
REFUND_CAP = "refunds_within_charges"

INSERT_REFUND = """
    INSERT INTO refunds (order_id, amount_paise, reason, run_id, idempotency_key)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id
"""


class KeyReused(RuntimeError):
    """An idempotency key that was already recorded for a different operation."""


@dataclass(frozen=True)
class ToolOutcome:
    key: str
    result: JsonObject
    replayed: bool


def operation_key(run_id: UUID, step: int, tool: str) -> str:
    """Names the operation, so every attempt at it carries the same key."""
    return f"{run_id}:step_{step}:{tool}"


def mentioned(connection: psycopg.Connection, run_id: UUID, order_id: str) -> bool:
    """
    Whether the customer wrote this order number, as a whole number, in the subject or the body.

    The model is not trusted to have got the number from the message. It was once seen looking up
    4821 for a message that named no order at all -- copied from an example in its own tool
    description -- and for a sender who owned a 4821 that lookup would have succeeded. Asking the
    model to behave better is a request; this is a check.

    Whole numbers only: 48210 contains 4821, and counting it would let a typo reach an order that
    belongs to someone else. A hyphen is not part of the number -- 'order-4821' is a way of writing
    4821, and every order id in both ledgers is plain digits -- so only letters and digits bound it.
    The text is NFKC-normalised first, so digits typed full-width by a CJK input method are the same
    digits the model read. Checked before ownership, and blind to the ledger, so the answer says
    nothing about whether the order exists or whose it is.
    """
    row = connection.execute(WRITTEN, (run_id,)).fetchone()
    if row is None:
        return False
    written = unicodedata.normalize("NFKC", f"{row[0]}\n{row[1]}")
    pattern = rf"(?<![A-Za-z0-9]){re.escape(order_id)}(?![A-Za-z0-9])"
    return re.search(pattern, written, re.IGNORECASE) is not None


def unmentioned(order_id: str) -> str:
    """Said plainly, because a person reads it on the screen when the run is handed over."""
    return f"the customer did not mention order {order_id}; it cannot come from anywhere else"


def get_order(connection: psycopg.Connection, run_id: UUID, key: str, args: JsonObject) -> JsonObject:
    """The sender's order, every charge taken for it, and what has already been paid back."""
    order_id = args["order_id"]
    if not mentioned(connection, run_id, order_id):
        return {"error": unmentioned(order_id)}
    order = connection.execute(OWNED_ORDER, (run_id, order_id)).fetchone()
    if order is None:
        return {"error": f"no order {order_id}"}

    customer_email, amount_paise, status = order
    charges = [
        amount
        for (amount,) in connection.execute(
            "SELECT amount_paise FROM charges WHERE order_id = %s ORDER BY id", (order_id,)
        ).fetchall()
    ]
    # sum() over bigint is numeric in Postgres, which psycopg reads as Decimal.
    refunded = connection.execute(
        "SELECT coalesce(sum(amount_paise), 0)::bigint FROM refunds WHERE order_id = %s", (order_id,)
    ).fetchone()
    return {
        "order_id": order_id,
        "customer_email": customer_email,
        "amount_paise": amount_paise,
        "status": status,
        "charges_paise": charges,
        "charged_paise": sum(charges),
        "refunded_paise": refunded[0] if refunded else 0,
    }


def issue_refund(connection: psycopg.Connection, run_id: UUID, key: str, args: JsonObject) -> JsonObject:
    """Pay money back. The cap is enforced by the database, under a lock on the order."""
    order_id, amount_paise = args["order_id"], args["amount_paise"]
    if not mentioned(connection, run_id, order_id):
        return {"refunded": False, "error": unmentioned(order_id)}
    if connection.execute(OWNED_ORDER, (run_id, order_id)).fetchone() is None:
        return {"refunded": False, "error": f"no order {order_id}"}

    try:
        # A savepoint, so a refused refund leaves the key and its result recordable.
        with connection.transaction():
            inserted = connection.execute(
                INSERT_REFUND, (order_id, amount_paise, args["reason"], run_id, key)
            ).fetchone()
    except psycopg.errors.CheckViolation as refused:
        if refused.diag.constraint_name != REFUND_CAP:
            raise
        return {"refunded": False, "error": refused.diag.message_primary or str(refused)}

    if inserted is None:  # pragma: no cover - INSERT ... RETURNING always yields the row
        raise RuntimeError(f"refund for {key} returned no row")
    return {"refunded": True, "refund_id": inserted[0], "order_id": order_id, "amount_paise": amount_paise}


def escalate_to_human(connection: psycopg.Connection, run_id: UUID, key: str, args: JsonObject) -> JsonObject:
    """Handing the case to a person is the one action that must never fail."""
    return {"escalated": True, "reason": args["reason"]}


TOOLS: dict[str, Tool] = {
    "get_order": get_order,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def replay(connection: psycopg.Connection, key: str, run_id: UUID, tool: str, args: JsonObject) -> JsonObject:
    """The stored result of an earlier call, provided it really was this operation."""
    stored = connection.execute(
        "SELECT run_id, tool, args, result FROM tool_calls WHERE idempotency_key = %s", (key,)
    ).fetchone()
    if stored is None:  # pragma: no cover - needs a concurrent DELETE of that row
        raise RuntimeError(f"tool call {key} conflicted with a row that is not there")

    stored_run, stored_tool, stored_args, result = stored
    if (stored_run, stored_tool, stored_args) != (run_id, tool, args):
        raise KeyReused(f"key {key} was already used for a different operation")
    if result is None:  # pragma: no cover - the result is written in the claiming transaction
        raise RuntimeError(f"tool call {key} was recorded without a result")
    return result


def execute(connection: psycopg.Connection, run_id: UUID, step: int, action: ProposedAction) -> ToolOutcome:
    """
    Perform `action` once for this run and step, or return what it did before.

    Commits when called with no transaction open; inside the caller's transaction
    it runs as a savepoint, so the caller decides when the effect becomes durable.
    """
    tool = TOOLS.get(action.tool)
    if tool is None:
        raise ValueError(f"{action.tool} is not executed by the ledger executor")

    key = operation_key(run_id, step, action.tool)
    args = dict(action.args)
    with connection.transaction():
        claimed = connection.execute(CLAIM_KEY, (key, run_id, action.tool, Jsonb(args))).fetchone()
        if claimed is None:
            return ToolOutcome(key=key, result=replay(connection, key, run_id, action.tool, args), replayed=True)

        result = tool(connection, run_id, key, args)
        connection.execute(
            "UPDATE tool_calls SET result = %s WHERE idempotency_key = %s", (Jsonb(result), key)
        )
    return ToolOutcome(key=key, result=result, replayed=False)
