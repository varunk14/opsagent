"""
The driver: take a queued run, walk it through the graph, record the proposal.

Proposing only. The run ends waiting for a decision, the proposal and its exact
cost are stored on the run row, and nothing is executed: no tool call recorded,
no approval created, no order touched.

These tests commit, so each one gets its own scratch database.
"""

from datetime import UTC, datetime
from decimal import Decimal

import psycopg
import pytest

from app.baseline import REFERENCE_RATE, token_cost
from app.contracts import Channel, IncomingMessage
from app.graph.build import build_graph
from app.intake import accept
from app.llm import ModelUnavailable, Reply
from app.run_agent import LostClaim, claim_next, propose_next
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    EXTRACTED_4821,
    OUTAGE,
    PROPOSED_LOOKUP,
    ScriptedModel,
)

pytestmark = pytest.mark.db


def happy_graph():
    return build_graph(
        ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_LOOKUP)
    )


def queue(dsn: str, external_id: str = "9f2a", minute: int = 0) -> str:
    message = IncomingMessage(
        channel=Channel.EMAIL,
        external_id=external_id,
        sender="priya@example.com",
        subject="Charged twice for order #4821",
        body="Hi, I think I was charged twice for order #4821 last Tuesday.",
        received_at=datetime(2026, 9, 13, 9, minute, tzinfo=UTC),
    )
    with psycopg.connect(dsn) as connection:
        run_id = accept(connection, message).run_id
        connection.execute(
            "UPDATE runs SET created_at = %s WHERE id = %s",
            (datetime(2026, 9, 13, 9, minute, tzinfo=UTC), run_id),
        )
    return str(run_id)


def row(dsn: str, run_id: str) -> dict:
    with psycopg.connect(dsn) as connection:
        cursor = connection.execute(
            "SELECT status, current_node, state, cost_usd, attempt, locked_by, locked_at "
            "FROM runs WHERE id = %s",
            (run_id,),
        )
        names = [column.name for column in cursor.description]
        return dict(zip(names, cursor.fetchone(), strict=True))


# --- the ordinary case --------------------------------------------------------


def test_a_queued_run_ends_with_a_proposal_waiting_for_a_decision(fresh_database):
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        outcome = propose_next(connection, happy_graph())

    assert str(outcome.run_id) == run_id
    stored = row(fresh_database, run_id)
    assert stored["status"] == "waiting_approval"
    assert stored["current_node"] == "plan"
    assert stored["state"]["agent"]["proposal"]["tool"] == "get_order"
    assert stored["state"]["agent"]["proposal"]["args"] == {"order_id": "4821"}


def test_what_the_steps_found_is_stored_beside_the_untouched_message(fresh_database):
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        propose_next(connection, happy_graph())

    state = row(fresh_database, run_id)["state"]
    assert state["agent"]["classification"]["intent"] == "duplicate_charge"
    assert state["agent"]["extraction"]["order_id"] == "4821"
    assert state["agent"]["policy"]
    assert state["untrusted"]["body"].startswith("Hi, I think I was charged twice")


def test_the_run_is_charged_exactly_for_every_model_call(fresh_database):
    """Three calls of 10 prompt and 5 completion tokens, priced at the baseline rate."""
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        propose_next(connection, happy_graph())

    stored = row(fresh_database, run_id)
    expected = token_cost(prompt_tokens=30, completion_tokens=15, rate=REFERENCE_RATE)
    assert stored["cost_usd"] == expected.quantize(Decimal("0.000001"))
    assert stored["state"]["agent"]["model_calls"] == 3
    assert stored["state"]["agent"]["prompt_tokens"] == 30
    assert stored["state"]["agent"]["completion_tokens"] == 15


def test_nothing_is_executed(fresh_database):
    """Week 2 proposes. No tool call, no approval row, no order."""
    queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        propose_next(connection, happy_graph())
        counts = connection.execute(
            "SELECT (SELECT count(*) FROM tool_calls), (SELECT count(*) FROM approvals),"
            " (SELECT count(*) FROM orders)"
        ).fetchone()

    assert counts == (0, 0, 0)


def test_the_lock_is_released_once_the_proposal_is_recorded(fresh_database):
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        propose_next(connection, happy_graph())

    stored = row(fresh_database, run_id)
    assert (stored["locked_by"], stored["locked_at"], stored["attempt"]) == (None, None, 1)


# --- which run, and only once ---------------------------------------------------


def test_an_empty_queue_asks_the_model_nothing(fresh_database):
    model = ScriptedModel()

    with psycopg.connect(fresh_database) as connection:
        assert propose_next(connection, build_graph(model)) is None

    assert model.prompts == []


def test_the_oldest_queued_run_goes_first(fresh_database):
    later = queue(fresh_database, "late", minute=30)
    earlier = queue(fresh_database, "early", minute=5)

    with psycopg.connect(fresh_database) as connection:
        outcome = propose_next(connection, happy_graph())

    assert str(outcome.run_id) == earlier
    assert row(fresh_database, later)["status"] == "queued"


def test_runs_that_are_not_queued_are_left_alone(fresh_database):
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET status = 'waiting_approval' WHERE id = %s", (run_id,))

    with psycopg.connect(fresh_database) as connection:
        assert propose_next(connection, happy_graph()) is None


def test_a_claimed_run_is_marked_as_running_with_its_worker(fresh_database):
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        claimed = claim_next(connection, worker="worker-7")

    stored = row(fresh_database, run_id)
    assert str(claimed.run_id) == run_id
    assert (stored["status"], stored["locked_by"], stored["attempt"]) == ("running", "worker-7", 1)
    assert stored["locked_at"] is not None


def test_two_workers_never_claim_the_same_run(fresh_database):
    """
    Worker A holds its claim open. Worker B must skip the locked row rather
    than wait for it or take it too -- which is what SKIP LOCKED is for.
    """
    first = queue(fresh_database, "one", minute=1)
    second = queue(fresh_database, "two", minute=2)

    with psycopg.connect(fresh_database) as holder:
        holder.execute("SELECT id FROM runs WHERE id = %s FOR UPDATE", (first,))

        with psycopg.connect(fresh_database) as other:
            claimed = claim_next(other, worker="worker-b")

        holder.rollback()

    assert str(claimed.run_id) == second


def test_driving_on_a_connection_already_in_a_transaction_is_refused(fresh_database):
    queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        connection.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="own transaction"):
            propose_next(connection, happy_graph())


# --- when the model does not cooperate ------------------------------------------


def test_an_escalation_is_recorded_and_still_waits_for_a_person(fresh_database):
    run_id = queue(fresh_database)
    graph = build_graph(ScriptedModel(classify="no idea at all"))

    with psycopg.connect(fresh_database) as connection:
        propose_next(connection, graph)

    stored = row(fresh_database, run_id)
    assert stored["status"] == "waiting_approval"
    assert stored["current_node"] == "classify"
    assert stored["state"]["agent"]["proposal"]["tool"] == "escalate_to_human"
    assert "classify" in stored["state"]["agent"]["failure"]
    assert stored["cost_usd"] > 0, "the failed attempts were still paid for"


def test_an_unreachable_model_puts_the_run_back_in_the_queue(fresh_database):
    """An outage is not a verdict on the case. The run waits for the model to return."""

    class Down:
        def generate(self, prompt: str) -> Reply:
            raise ModelUnavailable("connection refused")

    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        propose_next(connection, build_graph(Down()))

    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["locked_by"], stored["locked_at"]) == ("queued", None, None)
    assert "agent" not in stored["state"]
    assert stored["attempt"] == 1, "the attempt still counts toward max_attempts"


# --- review findings: attempts run out, claims can be lost ------------------------


class Down:
    def generate(self, prompt: str) -> Reply:
        raise ModelUnavailable("connection refused")


def test_a_run_out_of_attempts_is_dead_not_requeued(fresh_database):
    """
    Requeueing forever lets one message that always fails sit at the head of the
    queue for good, starving everything behind it.
    """
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET attempt = max_attempts - 1 WHERE id = %s", (run_id,))

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        propose_next(connection, build_graph(Down()))

    with psycopg.connect(fresh_database) as connection:
        status, failure_class, locked_by = connection.execute(
            "SELECT status, failure_class, locked_by FROM runs WHERE id = %s", (run_id,)
        ).fetchone()
    assert (status, failure_class, locked_by) == ("dead", "model_unavailable", None)


class ReclaimedMidRun:
    """Stands in for week 4's lock expiry: another worker takes the run while the model thinks."""

    def __init__(self, dsn: str, inner=None, then_raise: Exception | None = None):
        self.dsn, self.inner, self.then_raise = dsn, inner, then_raise

    def invoke(self, initial):
        with psycopg.connect(self.dsn) as other:
            other.execute("UPDATE runs SET locked_by = 'worker-b', locked_at = now()")
        if self.then_raise is not None:
            raise self.then_raise
        return self.inner.invoke(initial)


def test_a_proposal_is_not_recorded_over_a_claim_that_was_lost(fresh_database):
    run_id = queue(fresh_database)
    graph = ReclaimedMidRun(fresh_database, inner=happy_graph())

    with psycopg.connect(fresh_database) as connection, pytest.raises(LostClaim):
        propose_next(connection, graph, worker="worker-a")

    stored = row(fresh_database, run_id)
    assert stored["locked_by"] == "worker-b"
    assert "agent" not in stored["state"]


def test_an_outage_does_not_release_a_claim_that_was_lost(fresh_database):
    run_id = queue(fresh_database)
    graph = ReclaimedMidRun(fresh_database, then_raise=ModelUnavailable("down"))

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        propose_next(connection, graph, worker="worker-a")

    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["locked_by"]) == ("running", "worker-b")


def test_an_outage_partway_through_still_charges_for_finished_steps(fresh_database):
    """
    Found in milestone review. classify answered, then the model went down. The
    run goes back to the queue, but the classify call was paid for and is charged.
    """
    run_id = queue(fresh_database)
    graph = build_graph(ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=OUTAGE))

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        propose_next(connection, graph)

    stored = row(fresh_database, run_id)
    assert stored["status"] == "queued"
    assert stored["cost_usd"] == token_cost(10, 5, REFERENCE_RATE).quantize(Decimal("0.000001"))
