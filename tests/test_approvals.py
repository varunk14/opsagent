"""
Approvals: a person's decision, recorded once, that moves the run on.

A refund the guardrail will not run on its own becomes a pending approval and
its run waits. Deciding is one statement that insists on what it expects -- a
pending approval, a run that is waiting -- so a double click, two people at once,
or a decision about a run that has moved on all change nothing the second time.

Approved, the run goes back to the queue and the worker executes exactly the
stored action (tests/test_approval_path.py). Rejected, the run is finished and
nothing is paid. Deciding never executes anything and never commits: the screen
that calls it owns the transaction, and the executor is only ever the worker's.
"""

import uuid
from dataclasses import fields
from decimal import Decimal

import psycopg
import pytest

from psycopg.types.json import Jsonb

from app.approvals import (
    HandedOver,
    PendingApproval,
    decide,
    list_handed_over,
    list_pending,
    mark_executed,
    open_approval,
)
from app.contracts import ProposedAction

pytestmark = pytest.mark.db

REFUND = ProposedAction(
    tool="issue_refund",
    args={"order_id": "4821", "amount_paise": 720_000, "reason": "charged twice"},
    confidence=Decimal("0.9"),
    reasoning="the ledger shows a duplicate charge",
)
OVER_THE_LIMIT = "Rs 7,200 is not under the Rs 5,000 limit for automatic refunds"


def waiting_run(db, key: str = "email_msg_wait", status: str = "waiting_approval", locked_by: str | None = None):
    run_id = uuid.uuid4()
    db.execute(
        "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key, locked_by) "
        "VALUES (%s, 'email', %s, 'approval', '{}'::jsonb, %s, %s)",
        (run_id, status, key, locked_by),
    )
    return run_id


def ask(db, run_id) -> int:
    return open_approval(db, run_id, REFUND, {"sender": "priya@example.com"}, OVER_THE_LIMIT)


def run_of(db, run_id) -> tuple:
    return db.execute("SELECT status, failure_class FROM runs WHERE id = %s", (run_id,)).fetchone()


def approval(db, approval_id: int) -> tuple:
    return db.execute(
        "SELECT status, decided_by, decision_note, decided_at IS NOT NULL FROM approvals WHERE id = %s",
        (approval_id,),
    ).fetchone()


# --- opening -------------------------------------------------------------------------


def test_an_approval_records_the_action_the_evidence_and_why(db):
    run_id = waiting_run(db)

    approval_id = ask(db, run_id)

    action, evidence, confidence, reason, status = db.execute(
        "SELECT action, evidence, confidence, reason, status FROM approvals WHERE id = %s", (approval_id,)
    ).fetchone()
    assert action == REFUND.model_dump(mode="json")
    assert evidence == {"sender": "priya@example.com"}
    assert confidence == Decimal("0.90")
    assert (reason, status) == (OVER_THE_LIMIT, "pending")


# --- deciding ------------------------------------------------------------------------


def test_approving_sends_the_run_back_to_the_queue(db):
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)

    assert decide(db, approval_id, approved=True, by="asha", note="ledger shows both charges") is True

    assert approval(db, approval_id) == ("approved", "asha", "ledger shows both charges", True)
    assert run_of(db, run_id) == ("queued", None)


def test_rejecting_finishes_the_run_and_says_so(db):
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)

    assert decide(db, approval_id, approved=False, by="asha") is True

    assert approval(db, approval_id) == ("rejected", "asha", None, True)
    assert run_of(db, run_id) == ("done", "rejected")


def test_a_decision_is_taken_once(db):
    """A double click, or two people at once: the first decision stands."""
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)
    decide(db, approval_id, approved=True, by="asha")

    assert decide(db, approval_id, approved=False, by="ravi") is False

    assert approval(db, approval_id)[:2] == ("approved", "asha")
    assert run_of(db, run_id) == ("queued", None)


def test_an_approval_already_decided_stays_decided_when_its_run_waits_again(db):
    """
    An approved refund the ledger refuses sends its run back to waiting, with the
    approval still approved. Deciding that approval again must change nothing.
    """
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)
    decide(db, approval_id, approved=True, by="asha")
    db.execute("UPDATE runs SET status = 'waiting_approval' WHERE id = %s", (run_id,))

    assert decide(db, approval_id, approved=False, by="ravi") is False

    assert approval(db, approval_id)[:2] == ("approved", "asha")


@pytest.mark.parametrize("decision", [None, False], ids=["pending", "rejected"])
def test_only_an_approved_action_can_be_marked_executed(db, decision):
    """Database review: anything else is refused as False, not raised as a CHECK violation."""
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)
    if decision is not None:
        decide(db, approval_id, approved=decision, by="asha")

    assert mark_executed(db, approval_id) is False


def test_a_run_that_is_no_longer_waiting_is_not_moved(db):
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)
    db.execute("UPDATE runs SET status = 'running' WHERE id = %s", (run_id,))

    assert decide(db, approval_id, approved=True, by="asha") is False

    assert approval(db, approval_id)[0] == "pending"
    assert run_of(db, run_id) == ("running", None)


def test_deciding_an_approval_that_does_not_exist_changes_nothing(db):
    assert decide(db, 987_654_321, approved=True, by="asha") is False


@pytest.mark.parametrize("by", ["", "   "])
def test_a_decision_needs_the_name_of_who_made_it(db, by):
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)

    with pytest.raises(ValueError, match="who"):
        decide(db, approval_id, approved=True, by=by)

    assert approval(db, approval_id)[0] == "pending"


def test_a_note_is_kept_to_a_readable_length(db):
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)

    with pytest.raises(ValueError, match="1000"):
        decide(db, approval_id, approved=True, by="asha", note="x" * 1001)


def test_the_name_is_stored_without_surrounding_space(db):
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)

    decide(db, approval_id, approved=True, by="  asha ")

    assert approval(db, approval_id)[1] == "asha"


def test_deciding_does_not_commit(fresh_database):
    """The screen owns the transaction, as every caller does in this codebase."""
    with psycopg.connect(fresh_database) as setup:
        run_id = waiting_run(setup)
        approval_id = ask(setup, run_id)

    with psycopg.connect(fresh_database) as deciding:
        decide(deciding, approval_id, approved=True, by="asha")
        with psycopg.connect(fresh_database) as other:
            assert approval(other, approval_id)[0] == "pending"
        deciding.rollback()


# --- listing ---------------------------------------------------------------------------


def test_pending_approvals_are_listed_oldest_first(db):
    """
    By when a person was asked, not by id. The oldest is inserted last, with its
    time set directly -- the record trigger forbids moving created_at afterwards.
    """
    newer = ask(db, waiting_run(db, key="email_msg_newer"))
    decided = ask(db, waiting_run(db, key="email_msg_decided"))
    decide(db, decided, approved=False, by="asha")
    oldest_run = waiting_run(db, key="email_msg_oldest")
    oldest = db.execute(
        "INSERT INTO approvals (run_id, action, evidence, reason, created_at) "
        "VALUES (%s, '{}', '{}', 'test', now() - interval '1 day') RETURNING id",
        (oldest_run,),
    ).fetchone()[0]

    listed = [pending.id for pending in list_pending(db)]

    assert decided not in listed
    assert listed.index(oldest) < listed.index(newer)


def test_a_pending_approval_carries_what_the_screen_shows(db):
    run_id = waiting_run(db)
    approval_id = ask(db, run_id)

    (pending,) = [item for item in list_pending(db) if item.id == approval_id]

    assert pending.run_id == run_id
    assert pending.action["args"]["amount_paise"] == 720_000
    assert pending.evidence == {"sender": "priya@example.com"}
    assert pending.confidence == Decimal("0.90")
    assert pending.reason == OVER_THE_LIMIT
    assert pending.created_at is not None


def handed_over_run(db, *, key: str, status: str = "waiting_approval", agent: dict) -> uuid.UUID:
    run_id = uuid.uuid4()
    db.execute(
        "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key, locked_by) "
        "VALUES (%s, 'email', %s, 'act', %s, %s, 'build-host-4242')",
        (run_id, status, Jsonb({"untrusted": {"sender": "priya@example.com", "subject": "Refund", "body": "Hi"}, "agent": agent}), key),
    )
    return run_id


def test_a_run_handed_to_a_person_without_an_approval_is_listed_with_why(db):
    """Security review of Unit B: otherwise a refused refund waits where no screen shows it."""
    run_id = handed_over_run(db, key="email_msg_refused", agent={"failure": "act: the ledger refused the refund: over"})

    (handed,) = [item for item in list_handed_over(db) if item.run_id == run_id]

    assert handed.why == "act: the ledger refused the refund: over"
    assert (handed.sender, handed.subject, handed.body) == ("priya@example.com", "Refund", "Hi")


def test_an_escalation_the_planner_chose_is_listed_with_its_reason(db):
    agent = {"failure": None, "proposal": {"tool": "escalate_to_human", "args": {"reason": "asks about delivery"}}}
    run_id = handed_over_run(db, key="email_msg_escalated", agent=agent)

    (handed,) = [item for item in list_handed_over(db) if item.run_id == run_id]

    assert handed.why == "asks about delivery"


def test_a_run_with_a_pending_approval_is_not_listed_as_handed_over(db):
    run_id = waiting_run(db)
    ask(db, run_id)

    assert run_id not in {item.run_id for item in list_handed_over(db)}


def test_a_run_no_longer_waiting_is_not_listed_as_handed_over(db):
    run_id = handed_over_run(db, key="email_msg_finished", status="done", agent={"failure": "old"})

    assert run_id not in {item.run_id for item in list_handed_over(db)}


def test_the_handed_over_list_never_carries_the_worker_either(db):
    handed_over_run(db, key="email_msg_locked", agent={"failure": "act: refused"})

    assert "locked_by" not in {field.name for field in fields(HandedOver)}
    assert "build-host-4242" not in repr(list_handed_over(db))


def test_the_list_never_carries_the_worker_that_held_the_run(db):
    """Carried from week 4: locked_by is a hostname and a process id, and has no business on a screen."""
    run_id = waiting_run(db, locked_by="build-host-4242")
    ask(db, run_id)

    assert "locked_by" not in {field.name for field in fields(PendingApproval)}
    assert "build-host-4242" not in repr(list_pending(db))
