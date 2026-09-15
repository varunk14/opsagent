"""
The approval path in the worker: the guardrail decides at the act step, a person decides the rest.

Week 5. A refund the guardrail allows is paid through the keyed executor and
the run is done. One it does not allow becomes a pending approval, and the run
waits with nothing paid. Approved, the worker pays exactly the stored refund --
without asking the model again, and without judging it again, since a person has
already overruled the guardrail. Rejected, nothing is ever paid.

The limit applies to what the order would have had back in total, so a refund
split into parts under the limit is judged as the whole it adds up to. The ledger
cap and sender-scoped orders still apply to every refund however it was allowed;
one the ledger refuses goes to a person rather than being tried again.

These tests commit, so each one gets its own scratch database.
"""

import threading
import time
from datetime import UTC, datetime

import psycopg
import pytest

from app.approvals import approved_unexecuted, decide, list_pending
from app.contracts import Channel, IncomingMessage
from app.guardrails import set_limits
from app.intake import accept
from app.run_agent import (
    EVIDENCE_BODY_CHARS,
    LostClaim,
    act_on_approval,
    claim_next,
    work_next,
)
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    EXTRACTED_4821,
    PROPOSED_ESCALATE,
    PROPOSED_LOOKUP,
    ScriptedModel,
    proposed_refund,
)
from tests.test_run_agent import (
    ReclaimedMidRun,
    count,
    graph_of,
    keys,
    ledger,
    queue,
    row,
)

pytestmark = pytest.mark.db


class MustNotBeAsked:
    """An approved action is executed as it was approved; the model has no say in it."""

    def generate(self, prompt: str):
        raise AssertionError("the model was asked again about an action a person already approved")


def refund_model(amount_paise: int, confidence: str = "0.9") -> ScriptedModel:
    """Look order 4821 up, then propose refunding `amount_paise` at `confidence`."""
    return ScriptedModel(
        classify=CLASSIFIED_DUPLICATE,
        extract=EXTRACTED_4821,
        plan=[PROPOSED_LOOKUP, proposed_refund(amount_paise, confidence)],
    )


def work(dsn: str, model, **options):
    with psycopg.connect(dsn) as connection:
        return work_next(connection, graph_of(model), **options)


def refunds(dsn: str) -> list[tuple]:
    with psycopg.connect(dsn) as connection:
        return [
            (order_id, amount, str(run_id))
            for order_id, amount, run_id in connection.execute(
                "SELECT order_id, amount_paise, run_id FROM refunds ORDER BY id"
            ).fetchall()
        ]


def approvals_of(dsn: str, run_id: str) -> list[dict]:
    with psycopg.connect(dsn) as connection:
        cursor = connection.execute(
            "SELECT id, status, reason, action, evidence, confidence, executed_at "
            "FROM approvals WHERE run_id = %s ORDER BY id",
            (run_id,),
        )
        names = [column.name for column in cursor.description]
        return [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]


def decide_on(dsn: str, run_id: str, *, approved: bool) -> None:
    with psycopg.connect(dsn) as connection:
        (pending,) = [item for item in list_pending(connection) if str(item.run_id) == run_id]
        assert decide(connection, pending.id, approved=approved, by="asha")


def limit(dsn: str, paise: int) -> None:
    with psycopg.connect(dsn) as connection:
        set_limits(connection, limit_paise=paise, by="asha")


def wait_for_lock_waiters(dsn: str, expected: int, timeout: float = 20.0) -> None:
    """Block until `expected` sessions in this database are waiting on a lock."""
    deadline = time.monotonic() + timeout
    with psycopg.connect(dsn, autocommit=True) as watcher:
        while time.monotonic() < deadline:
            waiting = watcher.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'"
            ).fetchone()[0]
            if waiting >= expected:
                return
            time.sleep(0.05)
    raise AssertionError(f"expected {expected} sessions waiting on a lock")


# --- the guardrail at the act step -------------------------------------------------------


def test_a_refund_under_the_limit_is_paid_once_and_the_run_is_done(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    outcome = work(fresh_database, refund_model(90_000))

    assert (outcome.status, outcome.tool, outcome.steps, outcome.failure) == ("done", "issue_refund", 2, None)
    assert refunds(fresh_database) == [("4821", 90_000, run_id)]
    assert approvals_of(fresh_database, run_id) == []
    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:get_order", f"{run_id}:step_2:issue_refund"]
    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["current_node"], stored["locked_by"]) == ("done", "act", None)
    assert stored["state"]["agent"]["steps"][1]["result"]["refunded"] is True


def test_a_refund_over_the_limit_waits_for_a_person_and_pays_nothing(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    outcome = work(fresh_database, refund_model(720_000))

    assert (outcome.status, outcome.tool, outcome.steps, outcome.failure) == (
        "waiting_approval", "issue_refund", 1, None,
    )
    assert refunds(fresh_database) == []
    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:get_order"]
    (asked,) = approvals_of(fresh_database, run_id)
    assert asked["status"] == "pending"
    assert asked["reason"] == "Rs 7,200 is not under the Rs 5,000 limit for automatic refunds"
    assert asked["action"]["args"]["amount_paise"] == 720_000
    assert str(asked["confidence"]) == "0.90"
    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["current_node"], stored["locked_by"]) == ("waiting_approval", "approval", None)


def test_an_unsure_refund_waits_however_small(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    work(fresh_database, refund_model(90_000, confidence="0.5"))

    assert refunds(fresh_database) == []
    (asked,) = approvals_of(fresh_database, run_id)
    assert asked["reason"] == "confidence 0.50 is below the 0.85 needed for automatic refunds"


def test_the_evidence_holds_what_the_person_needs_to_decide(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    work(fresh_database, refund_model(720_000))

    (asked,) = approvals_of(fresh_database, run_id)
    evidence = asked["evidence"]
    assert evidence["sender"] == "priya@example.com"
    assert evidence["subject"] == "Charged twice for order #4821"
    assert evidence["body"].startswith("Hi, I think I was charged twice")
    assert evidence["classification"]["intent"] == "duplicate_charge"
    assert evidence["policy_sources"] == ["duplicate-payments#1"]
    assert evidence["steps"][0]["result"]["charges_paise"] == [360_000, 360_000]


def test_a_worker_that_lost_its_claim_asks_no_one_anything(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    graph = ReclaimedMidRun(fresh_database, inner=graph_of(refund_model(720_000)))

    with psycopg.connect(fresh_database) as connection, pytest.raises(LostClaim):
        work_next(connection, graph, worker="worker-a")

    assert approvals_of(fresh_database, run_id) == []


def test_a_refund_split_in_two_is_judged_as_the_whole_it_adds_up_to(fresh_database):
    """Found in the Unit A security review: two Rs 3,600 refunds must not pass a Rs 5,000 limit."""
    ledger(fresh_database)
    first = queue(fresh_database, "first", minute=1)
    second = queue(fresh_database, "second", minute=2)

    assert work(fresh_database, refund_model(360_000)).status == "done"
    outcome = work(fresh_database, refund_model(360_000))

    assert (str(outcome.run_id), outcome.status) == (second, "waiting_approval")
    assert refunds(fresh_database) == [("4821", 360_000, first)]
    (asked,) = approvals_of(fresh_database, second)
    assert asked["reason"] == (
        "Rs 3,600 would bring refunds on order 4821 to Rs 7,200, not under the Rs 5,000 limit for automatic refunds"
    )


def test_two_runs_refunding_one_order_at_once_are_judged_one_after_the_other(fresh_database):
    """
    The race half of the splitting finding. Both runs reach the act step while
    something else holds order 4821; released together, the second must judge
    against the first's refund, not against the nothing both would read unlocked.
    """
    ledger(fresh_database)
    queue(fresh_database, "first", minute=1)
    queue(fresh_database, "second", minute=2)
    outcomes = []
    errors: list[BaseException] = []

    def worker(name: str) -> None:
        try:
            outcomes.append(work(fresh_database, refund_model(360_000), worker=name))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("worker-a", "worker-b")]
    with psycopg.connect(fresh_database) as holder:
        holder.execute("SELECT 1 FROM orders WHERE id = '4821' FOR UPDATE")
        for thread in threads:
            thread.start()
        wait_for_lock_waiters(fresh_database, 2)
        holder.rollback()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    assert sorted(outcome.status for outcome in outcomes) == ["done", "waiting_approval"]
    assert len(refunds(fresh_database)) == 1


def test_the_evidence_keeps_only_the_start_of_a_long_message(fresh_database):
    """The whole thread stays on the run; the approval holds what a person will read."""
    ledger(fresh_database)
    message = IncomingMessage(
        channel=Channel.EMAIL,
        external_id="long-thread",
        sender="priya@example.com",
        subject="Charged twice",
        body="Charged twice for 4821. " + "quoted thread " * 1000,
        received_at=datetime(2026, 9, 13, 9, tzinfo=UTC),
    )
    with psycopg.connect(fresh_database) as connection:
        run_id = str(accept(connection, message).run_id)

    work(fresh_database, refund_model(720_000))

    (asked,) = approvals_of(fresh_database, run_id)
    assert len(asked["evidence"]["body"]) == EVIDENCE_BODY_CHARS
    assert asked["evidence"]["body"].startswith("Charged twice for 4821.")


def test_the_limit_in_force_when_the_worker_acts_is_the_one_applied(fresh_database):
    """A limit of zero is the kill switch, and it needs no restart."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    limit(fresh_database, 0)

    work(fresh_database, refund_model(90_000))

    assert refunds(fresh_database) == []
    (asked,) = approvals_of(fresh_database, run_id)
    assert asked["reason"] == "Rs 900 is not under the Rs 0 limit for automatic refunds"


def test_a_refund_the_ledger_refuses_goes_to_a_person(fresh_database):
    """Rs 8,000 against Rs 7,200 charged: allowed by a raised limit, refused by the ledger cap."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    limit(fresh_database, 1_000_000)

    outcome = work(fresh_database, refund_model(800_000))

    assert (outcome.status, outcome.tool) == ("waiting_approval", "issue_refund")
    assert "the ledger refused the refund" in outcome.failure
    assert refunds(fresh_database) == []
    assert row(fresh_database, run_id)["state"]["agent"]["steps"][1]["result"]["refunded"] is False
    assert work(fresh_database, MustNotBeAsked()) is None, "a refused refund is not retried"


def test_a_refund_on_someone_elses_order_is_refused_not_paid(fresh_database):
    """Order 3310 is Dev's. Priya's run counts nothing against it, and the executor answers as if it were missing."""
    ledger(fresh_database)
    queue(fresh_database)
    model = ScriptedModel(
        classify=CLASSIFIED_DUPLICATE,
        extract=EXTRACTED_4821,
        plan=[PROPOSED_LOOKUP, proposed_refund(90_000, order_id="3310")],
    )

    outcome = work(fresh_database, model)

    assert outcome.status == "waiting_approval"
    assert outcome.failure == "act: the ledger refused the refund: no order 3310"
    assert refunds(fresh_database) == []


# --- after a person decides -----------------------------------------------------------------


def test_an_approved_refund_is_paid_exactly_as_approved_without_asking_the_model(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(720_000))
    decide_on(fresh_database, run_id, approved=True)

    outcome = work(fresh_database, MustNotBeAsked())

    assert (outcome.status, outcome.tool, outcome.steps, outcome.failure) == ("done", "issue_refund", 2, None)
    assert refunds(fresh_database) == [("4821", 720_000, run_id)]
    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:get_order", f"{run_id}:step_2:issue_refund"]
    (approved,) = approvals_of(fresh_database, run_id)
    assert (approved["status"], approved["executed_at"] is not None) == ("approved", True)
    assert row(fresh_database, run_id)["state"]["agent"]["steps"][1]["result"]["refunded"] is True


def test_an_approved_refund_is_not_judged_again(fresh_database):
    """A person overruled the guardrail; switching automatic refunds off afterwards does not undo that."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(720_000))
    decide_on(fresh_database, run_id, approved=True)
    limit(fresh_database, 0)

    assert work(fresh_database, MustNotBeAsked()).status == "done"
    assert refunds(fresh_database) == [("4821", 720_000, run_id)]


def test_a_rejected_refund_is_never_paid(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(720_000))

    decide_on(fresh_database, run_id, approved=False)

    assert work(fresh_database, MustNotBeAsked()) is None
    assert refunds(fresh_database) == []
    stored = row(fresh_database, run_id)
    assert stored["status"] == "done"
    assert count(fresh_database, "tool_calls") == 1


def test_an_approved_refund_survives_the_worker_that_took_it_dying(fresh_database):
    """Claimed, then the worker vanished before acting: the next worker pays it, once."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(720_000))
    decide_on(fresh_database, run_id, approved=True)
    with psycopg.connect(fresh_database) as connection:
        assert claim_next(connection, worker="worker-a") is not None
        connection.execute("UPDATE runs SET locked_at = now() - interval '10 minutes' WHERE id = %s", (run_id,))

    outcome = work(fresh_database, MustNotBeAsked(), worker="worker-b")

    assert outcome.status == "done"
    assert refunds(fresh_database) == [("4821", 720_000, run_id)]
    assert work(fresh_database, MustNotBeAsked()) is None


def test_a_run_sent_back_to_the_queue_by_hand_never_pays_an_approval_twice(fresh_database):
    """Once executed, an approval is spent. An operator's UPDATE, or any future requeue, must not pay it again."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(360_000, confidence="0.5"))
    decide_on(fresh_database, run_id, approved=True)
    assert work(fresh_database, MustNotBeAsked()).status == "done"
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET status = 'queued' WHERE id = %s", (run_id,))

    work(fresh_database, ScriptedModel(plan=PROPOSED_ESCALATE))

    assert refunds(fresh_database) == [("4821", 360_000, run_id)]


def test_an_approved_action_that_can_no_longer_be_read_goes_to_a_person(fresh_database):
    """
    Python review: a stored action that no longer validates -- a schema change, an
    older worker's record -- must not crash every worker that claims the run.
    """
    ledger(fresh_database)
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        connection.execute(
            "INSERT INTO approvals (run_id, action, evidence, reason, status, decided_by, decided_at) "
            "VALUES (%s, '{\"tool\": \"issue_refund\", \"args\": {\"order_id\": \"4821\"}}', '{}', 'test', "
            "'approved', 'asha', now())",
            (run_id,),
        )

    outcome = work(fresh_database, MustNotBeAsked())

    assert outcome.status == "waiting_approval"
    assert "the approved action could not be read" in outcome.failure
    assert refunds(fresh_database) == []
    assert work(fresh_database, MustNotBeAsked()) is None


def test_an_approval_already_executed_is_not_paid_by_a_worker_holding_an_old_copy(fresh_database):
    """Python review: the act transaction checks for itself that the approval is still unexecuted."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(360_000, confidence="0.5"))
    decide_on(fresh_database, run_id, approved=True)
    with psycopg.connect(fresh_database) as connection, connection.transaction():
        old_copy = approved_unexecuted(connection, run_id)
    assert work(fresh_database, MustNotBeAsked()).status == "done"
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET status = 'queued' WHERE id = %s", (run_id,))
    with psycopg.connect(fresh_database) as connection:
        claimed = claim_next(connection, worker="worker-late")

        with pytest.raises(RuntimeError, match="already executed"):
            act_on_approval(connection, claimed, old_copy)

    assert refunds(fresh_database) == [("4821", 360_000, run_id)]


def test_a_worker_that_lost_its_claim_pays_nothing_that_was_approved(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(720_000))
    decide_on(fresh_database, run_id, approved=True)
    with psycopg.connect(fresh_database) as connection:
        stale = claim_next(connection, worker="worker-a")
        connection.execute("UPDATE runs SET locked_at = now() - interval '10 minutes' WHERE id = %s", (run_id,))
    with psycopg.connect(fresh_database) as connection:
        assert claim_next(connection, worker="worker-b") is not None

    with psycopg.connect(fresh_database) as connection:
        with connection.transaction():
            approved = approved_unexecuted(connection, stale.run_id)
        with pytest.raises(LostClaim):
            act_on_approval(connection, stale, approved)

    assert refunds(fresh_database) == []
    assert approvals_of(fresh_database, run_id)[0]["executed_at"] is None


def test_an_approved_refund_the_ledger_refuses_goes_back_to_a_person(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(800_000))
    decide_on(fresh_database, run_id, approved=True)

    outcome = work(fresh_database, MustNotBeAsked())

    assert outcome.status == "waiting_approval"
    assert "the ledger refused the refund" in outcome.failure
    assert refunds(fresh_database) == []
    (approved,) = approvals_of(fresh_database, run_id)
    assert approved["executed_at"] is not None, "attempted once; its refusal is the recorded result"
    assert work(fresh_database, MustNotBeAsked()) is None
