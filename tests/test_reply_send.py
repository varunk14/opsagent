"""
Draining the outbox: the reply becomes a message, and only then is it marked sent.

This is the poll ordering seen from the other side. A poll records first and confirms the
channel afterwards, because confirming first could lose a message. A drain sends first and
marks afterwards, for the same reason: mark-then-send would let a crash in between leave a
row that says "sent" with nothing sent. So the send happens, and the mark is what commits
it -- at-least-once, where the cost of being wrong is a customer hearing twice, never not
at all.

The tests drive that ordering by breaking the send and by breaking the mark, and check that
a failure is written down and retried rather than dropped, and that one channel's outage does
not cost a reply waiting on another.
"""

import uuid
from dataclasses import dataclass, field

import psycopg
import pytest

from app.replies.send import MAX_SEND_ATTEMPTS, drain

pytestmark = pytest.mark.db


@dataclass
class RecordingSender:
    """Sends by remembering, or raises when told to, so a test can break one channel at will."""

    sent: list = field(default_factory=list)
    fail_times: int = 0

    def send(self, reply) -> None:
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("the channel would not take it")
        self.sent.append(reply)


def a_run(db, *, channel: str = "email", key: str = "email_msg_<one@example.com>") -> uuid.UUID:
    run_id = uuid.uuid4()
    db.execute(
        "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
        "VALUES (%s, %s, 'done', 'act', '{}'::jsonb, %s)",
        (run_id, channel, key),
    )
    return run_id


def enqueue(db, run_id, *, channel="email", reply_to="priya@example.com", body="Your refund has been processed.",
            template="refund_issued", thread_ref="<one@example.com>") -> int:
    return db.execute(
        "INSERT INTO outbox (run_id, channel, reply_to, thread_ref, template, body) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (run_id, channel, reply_to, thread_ref, template, body),
    ).fetchone()[0]


def outbox_row(dsn, row_id):
    with psycopg.connect(dsn) as db:
        return db.execute(
            "SELECT state, attempts, sent_at, last_error FROM outbox WHERE id = %s", (row_id,)
        ).fetchone()


# --- the ordinary drain -------------------------------------------------------


def test_a_pending_reply_is_sent_and_marked(fresh_database):
    with psycopg.connect(fresh_database) as db:
        row_id = enqueue(db, a_run(db))
    sender = RecordingSender()

    with psycopg.connect(fresh_database) as db:
        summary = drain(db, {"email": sender})

    assert summary.sent == 1
    assert len(sender.sent) == 1
    state, _attempts, sent_at, _ = outbox_row(fresh_database, row_id)
    assert state == "sent"
    assert sent_at is not None


def test_the_oldest_reply_goes_first(fresh_database):
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db)
        first = enqueue(db, run_id, body="first")
        second = enqueue(db, run_id, body="second")
        db.execute("UPDATE outbox SET created_at = now() - interval '1 minute' WHERE id = %s", (first,))
        db.execute("UPDATE outbox SET created_at = now() WHERE id = %s", (second,))
    sender = RecordingSender()

    with psycopg.connect(fresh_database) as db:
        drain(db, {"email": sender})

    assert [reply.body for reply in sender.sent] == ["first", "second"]


# --- a send that fails --------------------------------------------------------


def test_a_failed_send_stays_pending_and_writes_down_why(fresh_database):
    with psycopg.connect(fresh_database) as db:
        row_id = enqueue(db, a_run(db))
    sender = RecordingSender(fail_times=1)

    with psycopg.connect(fresh_database) as db:
        summary = drain(db, {"email": sender})

    assert summary.sent == 0
    assert sender.sent == []
    state, attempts, sent_at, last_error = outbox_row(fresh_database, row_id)
    assert state == "pending"
    assert attempts == 1
    assert sent_at is None
    assert last_error


def test_a_reply_that_failed_once_is_retried(fresh_database):
    with psycopg.connect(fresh_database) as db:
        row_id = enqueue(db, a_run(db))
    sender = RecordingSender(fail_times=1)

    with psycopg.connect(fresh_database) as db:
        drain(db, {"email": sender})   # fails
    with psycopg.connect(fresh_database) as db:
        drain(db, {"email": sender})   # succeeds

    assert len(sender.sent) == 1
    assert outbox_row(fresh_database, row_id)[0] == "sent"


def test_a_reply_is_not_retried_forever(fresh_database):
    with psycopg.connect(fresh_database) as db:
        row_id = enqueue(db, a_run(db))
    sender = RecordingSender(fail_times=MAX_SEND_ATTEMPTS)

    for _ in range(MAX_SEND_ATTEMPTS):
        with psycopg.connect(fresh_database) as db:
            drain(db, {"email": sender})

    state, attempts, _, _ = outbox_row(fresh_database, row_id)
    assert state == "failed"
    assert attempts == MAX_SEND_ATTEMPTS
    # A failed row is off the pending path: another drain does not touch it.
    with psycopg.connect(fresh_database) as db:
        assert drain(db, {"email": sender}).sent == 0


# --- the ordering that matters ------------------------------------------------


def test_a_crash_between_send_and_mark_sends_again(fresh_database, monkeypatch):
    """
    The decisive one. If the mark fails after the send, the row must stay pending so the next
    drain sends it again. The alternative -- marking first -- would drop a reply on any crash in
    the gap. At-least-once: the customer may hear twice; they never hear nothing.
    """
    with psycopg.connect(fresh_database) as db:
        enqueue(db, a_run(db))
    sender = RecordingSender()

    def refuse(*args, **kwargs):
        raise RuntimeError("the mark did not commit")

    monkeypatch.setattr("app.replies.send.mark_sent", refuse)
    with pytest.raises(RuntimeError), psycopg.connect(fresh_database) as db:
        drain(db, {"email": sender})
    assert len(sender.sent) == 1, "it was sent, just not marked"

    monkeypatch.undo()
    with psycopg.connect(fresh_database) as db:
        drain(db, {"email": sender})
    assert len(sender.sent) == 2, "so the next drain sends it again"


def test_one_channel_down_does_not_cost_a_reply_on_another(fresh_database):
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db)
        enqueue(db, run_id, body="by email")
        tg_run = a_run(db, channel="telegram", key="telegram_msg_9999:11")
        enqueue(db, tg_run, channel="telegram", reply_to="9999", thread_ref=None, template="enquiry", body="by chat")

    email = RecordingSender()
    telegram = RecordingSender(fail_times=1)
    with psycopg.connect(fresh_database) as db:
        summary = drain(db, {"email": email, "telegram": telegram})

    assert [reply.body for reply in email.sent] == ["by email"]
    assert summary.sent == 1 and summary.failed == 1


# --- the transaction ----------------------------------------------------------


def test_a_drain_needs_its_own_transaction(fresh_database):
    with psycopg.connect(fresh_database) as db:
        db.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="own transaction"):
            drain(db, {"email": RecordingSender()})
