"""
Turn a run's resting outcome into the one reply it owes, in the transaction that rested it.

Called from the driver's act step (app/run_agent.py) once a run's status is known, on the
same connection and inside the same transaction. That is the guarantee the outbox exists
to give: the reply and the outcome commit together or not at all, so a refund cannot leave
without a word to the customer and a word cannot go out for an outcome that rolled back.

Nothing here asks a model. The outcome picks one of four fixed messages, the message is
rendered and frozen into the row, and the only customer detail that reaches it is the order
number the agent acted on and the amount it paid.

The reply target is resolved here and stored, not re-derived by the sender later, because the
run does not keep enough to reply from. An email threads on its Message-ID, which survives
only inside the idempotency key; a Telegram reply goes to the chat, which survives only inside
the key as well -- never to the sender id the run kept for rate-limiting, which in a group is
someone else entirely.
"""

from collections.abc import Mapping
from typing import Any
from uuid import UUID

import psycopg

from app.replies import templates

# Read in the act transaction: the channel decided at intake, the key we can reply through,
# and the address the customer wrote from. All committed long before this run started resting.
RUN_FOR_REPLY = """
    SELECT channel, idempotency_key, state -> 'untrusted' ->> 'sender'
      FROM runs
     WHERE id = %s
"""

# What the lookups on this run established about the order: did any find one, did any come
# back empty. "Not found" is the second without the first -- a run that found an order and
# then stopped for some other reason is not a run that could not find the order.
LOOKUP_OUTCOME = """
    SELECT coalesce(bool_or(result ? 'charges_paise'), false)          AS any_found,
           coalesce(bool_or(result ->> 'error' LIKE 'no order%%'), false) AS any_missing
      FROM tool_calls
     WHERE run_id = %s AND tool = 'get_order'
"""

INSERT_REPLY = """
    INSERT INTO outbox (run_id, channel, reply_to, thread_ref, template, body)
    VALUES (%s, %s, %s, %s, %s, %s)
"""


def enqueue_reply(
    connection: psycopg.Connection,
    run_id: UUID,
    *,
    status: str,
    result: str,
    proposal: Any,
    failure: str | None,
) -> None:
    """Write the reply this outcome owes, or nothing when it owes none. Does not commit."""
    row = connection.execute(RUN_FOR_REPLY, (run_id,)).fetchone()
    if row is None:  # pragma: no cover - the run was just acted on, so it exists
        return
    channel, key, sender = row
    if channel == "form":
        # A web form has no channel to reply on; the screen shows the outcome instead.
        return

    chosen = _template_for(connection, run_id, status, result, proposal, channel)
    if chosen is None:
        return
    template, order_id, amount_paise = chosen

    body = templates.render(template, order_id=order_id, amount_paise=amount_paise)
    reply_to, thread_ref = _target(channel, key, sender)
    connection.execute(INSERT_REPLY, (run_id, channel, reply_to, thread_ref, template, body))


def _template_for(
    connection: psycopg.Connection,
    run_id: UUID,
    status: str,
    result: str,
    proposal: Any,
    channel: str,
) -> tuple[str, str | None, int | None] | None:
    """The template and its safe fields, or None when this outcome says nothing to the customer."""
    if status == "done" and result == "refunded":
        args: Mapping[str, Any] = proposal.args
        return templates.REFUND_ISSUED, args.get("order_id"), args.get("amount_paise")
    if status == "waiting_approval":
        if channel == "telegram":
            # Telegram is enquiries only by design: a sender there owns no order to refund.
            return templates.ENQUIRY, None, None
        if _order_not_found(connection, run_id):
            return templates.ORDER_NOT_FOUND, None, None
        return templates.HANDED_TO_PERSON, None, None
    # running, queued, failed, dead: still working, or an internal failure the customer is not told of.
    return None


def _order_not_found(connection: psycopg.Connection, run_id: UUID) -> bool:
    # The aggregate always returns its one row, coalesced, so a run with no lookups reads as false.
    row = connection.execute(LOOKUP_OUTCOME, (run_id,)).fetchone()
    if row is None:  # pragma: no cover - an aggregate over zero rows still returns one row
        return False
    any_found, any_missing = row
    return bool(any_missing) and not bool(any_found)


def _target(channel: str, key: str, sender: str | None) -> tuple[str, str | None]:
    """
    Where the reply goes, from what the run kept.

    Email threads on the Message-ID, which is the key with its channel prefix removed. Telegram
    replies to the chat, which is the first half of the key's `chat:message` identifier -- not the
    sender, which is the person and, in a group, not the conversation.
    """
    if channel == "telegram":
        identifier = key.removeprefix("telegram_msg_")
        chat_id = identifier.split(":", 1)[0]
        return chat_id, None
    # email (and any future threaded channel): reply to the sender, thread on the Message-ID.
    message_id = key.removeprefix("email_msg_")
    return sender or "", message_id
