"""
A rested run owes the customer a word, written in the transaction that earned it.

The reply is not a message yet -- it is a row in the outbox, chosen by how the run
turned out and frozen there beside the outcome. The binding is the whole point: a
refund that committed cannot have left without the reply that says so committing
with it, and an outcome that rolled back takes its reply down too. So the tests
here drive a run to each resting outcome and read the row it left, and one of them
breaks the write on purpose to prove the outcome cannot commit without it.

No model writes any of this. The wording is fixed per outcome; the only customer
detail that reaches a reply is the order number the agent acted on and the amount
it paid.
"""

from datetime import UTC, datetime

import psycopg
import pytest

from app.contracts import Channel, IncomingMessage
from app.intake import accept
from app.run_agent import work_next
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    CLASSIFIED_STATUS,
    EXTRACTED_4821,
    PROPOSED_ESCALATE,
    PROPOSED_LOOKUP,
    ScriptedModel,
    proposed_refund,
)
from tests.test_run_agent import graph_of, happy_graph, ledger, queue, row

pytestmark = pytest.mark.db


def replies_for(dsn: str, run_id: str) -> list[dict]:
    with psycopg.connect(dsn) as connection:
        cursor = connection.execute(
            "SELECT channel, reply_to, thread_ref, template, body, state "
            "FROM outbox WHERE run_id = %s ORDER BY id",
            (run_id,),
        )
        names = [column.name for column in cursor.description]
        return [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]


def queue_telegram(dsn: str, *, chat: str = "9999", message: str = "11") -> str:
    """A Telegram enquiry: no order to own, replied to on the chat, not the sender id."""
    incoming = IncomingMessage(
        channel=Channel.TELEGRAM,
        external_id=f"{chat}:{message}",
        sender=chat,
        subject=None,
        body="Hi, when will my order arrive?",
        received_at=datetime(2026, 9, 13, 9, 0, tzinfo=UTC),
    )
    with psycopg.connect(dsn) as connection:
        return str(accept(connection, incoming).run_id)


def escalating(intent: str = CLASSIFIED_DUPLICATE) -> ScriptedModel:
    return ScriptedModel(classify=intent, extract=EXTRACTED_4821, plan=PROPOSED_ESCALATE)


# --- a refund the customer is told about --------------------------------------


def test_a_paid_refund_writes_one_reply_that_names_it(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    replies = replies_for(fresh_database, run_id)
    assert len(replies) == 1
    reply = replies[0]
    assert reply["channel"] == "email"
    assert reply["template"] == "refund_issued"
    assert reply["reply_to"] == "priya@example.com"
    assert reply["thread_ref"] == "9f2a"
    assert reply["state"] == "pending"
    assert "3,600.00" in reply["body"]
    assert "4821" in reply["body"]


# --- a handover the customer is told about ------------------------------------


def test_an_escalation_hands_the_customer_to_a_person(fresh_database):
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(escalating(CLASSIFIED_STATUS)))

    replies = replies_for(fresh_database, run_id)
    assert len(replies) == 1
    assert replies[0]["template"] == "handed_to_person"
    assert replies[0]["reply_to"] == "priya@example.com"


def test_a_refund_that_needs_approval_reads_as_a_handover(fresh_database):
    """The customer is not promised a refund a person has not approved yet."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    model = ScriptedModel(
        classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, proposed_refund(720_000)]
    )
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(model))

    replies = replies_for(fresh_database, run_id)
    assert len(replies) == 1
    assert replies[0]["template"] == "handed_to_person"


# --- a Telegram enquiry -------------------------------------------------------


def test_a_telegram_enquiry_is_answered_on_the_chat(fresh_database):
    run_id = queue_telegram(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(escalating(CLASSIFIED_STATUS)))

    replies = replies_for(fresh_database, run_id)
    assert len(replies) == 1
    assert replies[0]["channel"] == "telegram"
    assert replies[0]["template"] == "enquiry"
    assert replies[0]["reply_to"] == "9999"
    assert replies[0]["thread_ref"] is None


# --- an order we could not find -----------------------------------------------


def test_an_order_the_ledger_does_not_have_says_so(fresh_database):
    """No ledger loaded, so the lookup finds nothing; the reply says we could not match it."""
    run_id = queue(fresh_database)
    model = ScriptedModel(
        classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, PROPOSED_ESCALATE]
    )
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(model))

    replies = replies_for(fresh_database, run_id)
    assert len(replies) == 1
    assert replies[0]["template"] == "order_not_found"


# --- what does not owe a reply ------------------------------------------------


def test_a_lookup_along_the_way_is_not_a_reply(fresh_database):
    """
    The happy run takes two ticks -- a lookup, then the refund -- and rests once. A reply is owed
    by the resting, not by every tick: the mid-flight get_order must leave no row, so the refunded
    run ends with exactly one reply and not one per step.
    """
    ledger(fresh_database)
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    with psycopg.connect(fresh_database) as connection:
        tool_calls = connection.execute(
            "SELECT count(*) FROM tool_calls WHERE run_id = %s", (run_id,)
        ).fetchone()[0]
    assert tool_calls == 2, "the lookup tick and the refund tick both ran"
    assert len(replies_for(fresh_database, run_id)) == 1, "but only the resting owed a reply"


# --- the binding --------------------------------------------------------------


def test_the_outcome_cannot_commit_without_its_reply(fresh_database, monkeypatch):
    """
    The decisive one. If writing the reply fails, the refund it was about must fail too:
    a customer's money moving with no word to them is exactly the silence this avoids.
    """
    ledger(fresh_database)
    run_id = queue(fresh_database)

    def refuse(*args, **kwargs):
        raise RuntimeError("the outbox went away mid-commit")

    monkeypatch.setattr("app.run_agent.enqueue_reply", refuse)

    with pytest.raises(RuntimeError), psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    assert row(fresh_database, run_id)["status"] != "done"
    with psycopg.connect(fresh_database) as connection:
        refunds = connection.execute("SELECT count(*) FROM refunds").fetchone()[0]
    assert refunds == 0
