"""
The boundary where untrusted input becomes a typed object.

Everything the agent later does is built on the assumption that a run row is
well formed. That assumption is only true if it is enforced exactly once, here,
at the edge. These tests are what makes it true.

The two that matter most are the idempotency-key tests. The key is what stops a
second poll of the same mailbox from creating a second run, and it must identify
THE MESSAGE, not the moment we happened to read it -- the same distinction that
made scenarios B and C differ in experiments/prevent_duplicate_refunds.py.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.contracts import Channel, IncomingMessage, RunRecord, RunStatus

RECEIVED_AT = datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc)


def a_message(**overrides) -> IncomingMessage:
    """Priya's email, which every part of this project is traced against."""
    fields = {
        "channel": Channel.EMAIL,
        "external_id": "9f2a",
        "sender": "priya@example.com",
        "subject": "Charged twice for order #4821",
        "body": "Hi, I think I was charged twice for order #4821 last Tuesday.",
        "received_at": RECEIVED_AT,
    }
    return IncomingMessage(**{**fields, **overrides})


# --- the idempotency key ----------------------------------------------------


def test_key_is_derived_from_the_channel_and_message_id():
    assert a_message().idempotency_key == "email_msg_9f2a"


def test_key_ignores_everything_except_the_message_identity():
    """
    Re-polling the same mailbox can return the same message with a different
    subject line after an edit, or a timestamp that reads differently. None of
    that makes it a new message, so none of it may change the key.
    """
    first = a_message()
    second = a_message(subject="Re: charged twice", received_at=datetime.now(timezone.utc))

    assert first.idempotency_key == second.idempotency_key


def test_different_messages_get_different_keys():
    assert a_message().idempotency_key != a_message(external_id="7c1b").idempotency_key


def test_the_same_id_on_a_different_channel_is_a_different_message():
    """Telegram message 9f2a and email 9f2a are unrelated; do not collapse them."""
    email = a_message()
    telegram = a_message(channel=Channel.TELEGRAM, sender="@priya")

    assert email.idempotency_key != telegram.idempotency_key


# --- rejecting what must never reach the database ---------------------------


def test_a_message_with_no_id_is_rejected():
    """Without an id there is no key, and without a key there is no idempotency."""
    with pytest.raises(ValidationError):
        a_message(external_id="")


def test_a_whitespace_only_id_is_rejected():
    with pytest.raises(ValidationError):
        a_message(external_id="   ")


def test_an_empty_body_is_rejected():
    with pytest.raises(ValidationError):
        a_message(body="  ")


def test_an_unknown_channel_is_rejected():
    with pytest.raises(ValidationError):
        a_message(channel="carrier_pigeon")


def test_the_sender_is_required():
    with pytest.raises(ValidationError):
        a_message(sender="")


def test_a_missing_subject_is_allowed():
    """Telegram and web forms have no subject. Absence is not malformed."""
    assert a_message(subject=None).subject is None


# --- the run record ---------------------------------------------------------


def test_a_new_run_starts_queued_with_no_attempts_spent():
    run = RunRecord(channel=Channel.EMAIL, idempotency_key="email_msg_9f2a")

    assert run.status is RunStatus.QUEUED
    assert run.attempt == 0
    assert run.max_attempts == 5


def test_every_run_gets_its_own_id():
    one = RunRecord(channel=Channel.EMAIL, idempotency_key="email_msg_1")
    two = RunRecord(channel=Channel.EMAIL, idempotency_key="email_msg_2")

    assert one.id != two.id


def test_an_unknown_status_is_rejected():
    """The six states are the whole state machine. A seventh is a bug."""
    with pytest.raises(ValidationError):
        RunRecord(channel=Channel.EMAIL, idempotency_key="k", status="mostly_done")


def test_cost_is_an_exact_decimal():
    run = RunRecord(channel=Channel.EMAIL, idempotency_key="k", cost_usd="0.000135")

    assert run.cost_usd == Decimal("0.000135")


def test_cost_given_as_a_float_is_rejected():
    """
    0.1 + 0.2 != 0.3 in binary floating point. A per-step cost accumulated as a
    float drifts, and week 9 compares these numbers against a baseline. Reject
    the float at the boundary rather than explaining the discrepancy later.
    """
    with pytest.raises(ValidationError):
        RunRecord(channel=Channel.EMAIL, idempotency_key="k", cost_usd=0.1)


def test_cost_cannot_be_negative():
    with pytest.raises(ValidationError):
        RunRecord(channel=Channel.EMAIL, idempotency_key="k", cost_usd=Decimal("-1"))


def test_a_run_is_built_from_a_message_without_restating_the_key():
    """
    The key lives in exactly one place. If intake had to recompute it, the two
    copies could drift apart, and the drift would only show up as a duplicate
    refund in production.
    """
    message = a_message()

    run = RunRecord.from_message(message)

    assert run.idempotency_key == message.idempotency_key
    assert run.channel is message.channel
    assert run.state["body"] == message.body
