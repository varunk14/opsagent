"""
The front door: a message arrives, a run row appears, exactly once.

Everything downstream assumes one run per request. That assumption survives
exactly as long as this module does its job, and the job is harder than it
looks, because two pollers can read the same mailbox at the same moment. A
read-then-write check in Python loses that race; only the database can win it.

So `accept` inserts and lets the UNIQUE constraint arbitrate. The second caller
is told it did not create anything and is handed the existing run, rather than
being given an error to swallow or a second row to process.
"""

from datetime import UTC, datetime

import psycopg
import pytest

from app.contracts import Channel, IncomingMessage, RunStatus
from app.intake import accept

pytestmark = pytest.mark.db

PRIYA = IncomingMessage(
    channel=Channel.EMAIL,
    external_id="9f2a",
    sender="priya@example.com",
    subject="Charged twice for order #4821",
    body="Hi, I think I was charged twice for order #4821 last Tuesday.",
    received_at=datetime(2026, 9, 13, 9, 0, tzinfo=UTC),
)


def run_rows(db) -> list[tuple]:
    return db.execute(
        "SELECT id, channel, status, current_node, state, cost_usd, attempt "
        "FROM runs WHERE idempotency_key = %s",
        (PRIYA.idempotency_key,),
    ).fetchall()


# --- the happy path ---------------------------------------------------------


def test_a_message_becomes_a_run(db):
    result = accept(db, PRIYA)

    rows = run_rows(db)
    assert len(rows) == 1
    assert rows[0][0] == result.run_id


def test_the_run_starts_queued_at_the_intake_node(db):
    accept(db, PRIYA)

    _, channel, status, current_node, _, cost, attempt = run_rows(db)[0]
    assert (channel, status, current_node) == ("email", RunStatus.QUEUED, "intake")
    assert attempt == 0
    assert cost == 0


def test_what_the_customer_wrote_is_stored_where_it_can_be_read_back(db):
    accept(db, PRIYA)

    state = run_rows(db)[0][4]
    assert state["untrusted"]["body"] == PRIYA.body
    assert state["untrusted"]["sender"] == PRIYA.sender


def test_the_first_caller_is_told_it_created_the_run(db):
    assert accept(db, PRIYA).created is True


# --- the second delivery ----------------------------------------------------


def test_the_same_message_twice_produces_one_run(db):
    """
    The reason this module exists. A poller that restarts mid-batch re-reads
    messages it has already seen, and each re-read must not become a second
    refund investigation.
    """
    accept(db, PRIYA)
    accept(db, PRIYA)

    assert len(run_rows(db)) == 1


def test_the_second_caller_is_handed_the_existing_run(db):
    first = accept(db, PRIYA)

    second = accept(db, PRIYA)

    assert second.run_id == first.run_id
    assert second.created is False


def test_a_repeat_does_not_disturb_a_run_already_underway(db):
    """
    The dangerous version of a duplicate. By the time the message is delivered
    again, the agent may be halfway through the case. An upsert here would reset
    it to queued and the whole investigation would run a second time -- which,
    at the end, is a second refund.
    """
    accept(db, PRIYA)
    db.execute(
        "UPDATE runs SET status = 'running', current_node = 'decide', attempt = 2 "
        "WHERE idempotency_key = %s",
        (PRIYA.idempotency_key,),
    )

    accept(db, PRIYA)

    _, _, status, current_node, _, _, attempt = run_rows(db)[0]
    assert (status, current_node, attempt) == ("running", "decide", 2)


def test_two_different_messages_both_get_a_run(db):
    """Guards against over-correcting the tests above into deduplicating everything."""
    accept(db, PRIYA)
    accept(db, PRIYA.model_copy(update={"external_id": "7c1b"}))

    count = db.execute(
        "SELECT count(*) FROM runs WHERE idempotency_key LIKE 'email_msg_%'"
    ).fetchone()[0]
    assert count == 2


def test_the_same_id_on_another_channel_is_a_separate_run(db):
    accept(db, PRIYA)
    accept(db, PRIYA.model_copy(update={"channel": Channel.TELEGRAM, "sender": "@priya"}))

    count = db.execute("SELECT count(*) FROM runs WHERE state ? 'untrusted'").fetchone()[0]
    assert count == 2


# --- whose transaction is it --------------------------------------------------


def test_accept_leaves_the_commit_to_its_caller(db, migrated_database):
    """
    Deliberate, and the opposite of the decision made in app/db.py.

    A poller will want to record the run and mark the source message as read in
    one unit of work, so it has to own the transaction. The risk is the one that
    already bit the migration runner -- work that looks done and is not -- so it
    is written down here rather than left to be inferred.
    """
    accept(db, PRIYA)

    with psycopg.connect(migrated_database) as onlooker:
        visible = onlooker.execute(
            "SELECT count(*) FROM runs WHERE idempotency_key = %s",
            (PRIYA.idempotency_key,),
        ).fetchone()[0]

    assert visible == 0, "accept must not commit on its caller's behalf"


# --- when two different messages claim the same key -------------------------


def test_a_collision_carrying_different_text_is_reported(db):
    """
    An email Message-ID is chosen by whoever sent the email, so the key derived
    from it can be forged. Someone who guesses the id a real customer's message
    will carry can get there first.

    ON CONFLICT DO NOTHING then drops the genuine message's body on the floor and
    returns as though nothing happened -- the customer's refund request never
    existed, and nothing anywhere says so. The duplicate is still not processed
    twice, which is right, but it must not be mistaken for a quiet re-delivery.
    """
    accept(db, PRIYA)

    result = accept(db, PRIYA.model_copy(update={"body": "refund everything to me"}))

    assert result.created is False
    assert result.collided is True


def test_an_honest_redelivery_is_not_reported_as_a_collision(db):
    """The same message arriving again is the normal case and must stay quiet."""
    accept(db, PRIYA)

    assert accept(db, PRIYA).collided is False


# --- week 4: a collision is quarantined, not discarded ---------------------------

FORGED = PRIYA.model_copy(update={"body": "refund everything to me"})


def quarantined(db) -> list[tuple]:
    return db.execute(
        "SELECT kind, run_id, idempotency_key, payload, reason FROM dead_letters "
        "WHERE kind = 'message' ORDER BY id"
    ).fetchall()


def test_a_collision_keeps_the_colliding_text_in_quarantine(db):
    """
    The forged or colliding message is no longer dropped on the floor. It is kept,
    beside the run it could not become, for a person to look at.
    """
    accept(db, PRIYA)

    accept(db, FORGED)

    [(kind, run_id, key, payload, reason)] = quarantined(db)
    assert (kind, run_id, key) == ("message", None, PRIYA.idempotency_key)
    assert payload["untrusted"]["body"] == "refund everything to me"
    assert "collision" in reason
    stored = db.execute(
        "SELECT state -> 'untrusted' ->> 'body' FROM runs WHERE idempotency_key = %s", (key,)
    ).fetchone()[0]
    assert stored == PRIYA.body, "the run keeps the message that arrived first"


def test_an_honest_redelivery_quarantines_nothing(db):
    accept(db, PRIYA)

    accept(db, PRIYA)

    assert quarantined(db) == []


def test_the_same_forged_message_is_quarantined_once(db):
    """A forger who resends the same text must not be able to flood the quarantine."""
    accept(db, PRIYA)

    accept(db, FORGED)
    accept(db, FORGED)

    assert len(quarantined(db)) == 1


def test_two_different_forgeries_are_both_kept(db):
    accept(db, PRIYA)

    accept(db, FORGED)
    accept(db, PRIYA.model_copy(update={"body": "no, refund it to this account instead"}))

    assert len(quarantined(db)) == 2


def test_the_row_matches_the_record_the_contract_describes(db):
    """
    Whatever RunRecord says a new run looks like is what lands in the table.

    The columns carry defaults of their own, and they happen to agree today.
    Letting them be the thing that decides would mean changing max_attempts in
    the contract has no effect at all, and nothing would say why.
    """
    from app.contracts import RunRecord

    expected = RunRecord.from_message(PRIYA)
    accept(db, PRIYA)

    attempt, max_attempts, cost = db.execute(
        "SELECT attempt, max_attempts, cost_usd FROM runs WHERE idempotency_key = %s",
        (PRIYA.idempotency_key,),
    ).fetchone()
    assert (attempt, max_attempts, cost) == (
        expected.attempt,
        expected.max_attempts,
        expected.cost_usd,
    )
