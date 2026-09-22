"""
The outbox table, and the constraints that have to bite.

A reply the agent owes a customer is a row here before it is a message anywhere.
The row is written in the same transaction as the run's outcome, so a refund that
committed cannot leave without the reply that says so also committed. What is
worth testing is not that the columns exist -- the migration shows that -- but
that the database refuses the states a reply must never be in:

  a template outside the fixed set    -> an outcome nothing rendered
  a state outside pending/sent/failed -> a row no drain knows what to do with
  a channel the run never came from   -> a reply with nowhere to go
  a run_id that names no run          -> a reply to a customer we cannot find
"""

import uuid

import psycopg
import pytest

pytestmark = pytest.mark.db


def a_run(db, *, key: str = "email_msg_<one@example.com>", channel: str = "email") -> uuid.UUID:
    run_id = uuid.uuid4()
    db.execute(
        """
        INSERT INTO runs (id, channel, status, current_node, state, idempotency_key)
        VALUES (%s, %s, 'done', 'act', '{}'::jsonb, %s)
        """,
        (run_id, channel, key),
    )
    return run_id


def enqueue(db, run_id, **overrides) -> None:
    row = {
        "channel": "email",
        "reply_to": "priya@example.com",
        "thread_ref": "<one@example.com>",
        "template": "refund_issued",
        "body": "Your refund has been processed.",
        "state": "pending",
    }
    row.update(overrides)
    db.execute(
        """
        INSERT INTO outbox (run_id, channel, reply_to, thread_ref, template, body, state)
        VALUES (%(run_id)s, %(channel)s, %(reply_to)s, %(thread_ref)s, %(template)s, %(body)s, %(state)s)
        """,
        {"run_id": run_id, **row},
    )


def test_an_ordinary_reply_is_accepted(fresh_database):
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db)
        enqueue(db, run_id)  # does not raise


def test_a_reply_defaults_to_pending_and_unsent(fresh_database):
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db)
        db.execute(
            "INSERT INTO outbox (run_id, channel, reply_to, template, body) "
            "VALUES (%s, 'email', 'priya@example.com', 'refund_issued', 'done')",
            (run_id,),
        )
        state, attempts, sent_at = db.execute(
            "SELECT state, attempts, sent_at FROM outbox WHERE run_id = %s", (run_id,)
        ).fetchone()
    assert (state, attempts, sent_at) == ("pending", 0, None)


def test_a_template_outside_the_set_is_refused(fresh_database):
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db)
        with pytest.raises(psycopg.errors.CheckViolation):
            enqueue(db, run_id, template="apology_coupon")


def test_a_state_outside_the_machine_is_refused(fresh_database):
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db)
        with pytest.raises(psycopg.errors.CheckViolation):
            enqueue(db, run_id, state="halfway")


def test_a_channel_the_run_never_came_from_is_refused(fresh_database):
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db)
        with pytest.raises(psycopg.errors.CheckViolation):
            enqueue(db, run_id, channel="carrier_pigeon")


def test_a_reply_to_no_run_is_refused(fresh_database):
    with psycopg.connect(fresh_database) as db, pytest.raises(psycopg.errors.ForeignKeyViolation):
        enqueue(db, uuid.uuid4())


def test_two_replies_for_one_run_are_allowed(fresh_database):
    """A run handed to a person and later refunded owes two honest replies, not one."""
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db)
        enqueue(db, run_id, template="handed_to_person", body="A colleague is looking into this.")
        enqueue(db, run_id, template="refund_issued", body="Your refund has been processed.")
        count = db.execute("SELECT count(*) FROM outbox WHERE run_id = %s", (run_id,)).fetchone()[0]
    assert count == 2


def test_a_voice_run_may_owe_a_voice_reply(fresh_database):
    """Voice is a first-class channel: `runs` and `outbox` both accept it, so a voice run's
    outcome can be written in the same transaction that rested it, exactly like email or telegram."""
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db, key="voice_clip_abc123", channel="voice")
        enqueue(db, run_id, channel="voice", template="enquiry", body="A person will be in touch.")
        count = db.execute(
            "SELECT count(*) FROM outbox WHERE run_id = %s AND channel = 'voice'", (run_id,)
        ).fetchone()[0]
    assert count == 1
