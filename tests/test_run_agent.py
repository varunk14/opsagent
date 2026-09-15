"""
The driver: take a queued run and work it one committed step at a time.

The tools run. get_order and escalate_to_human execute through the
keyed executor; issue_refund is judged by the guardrail, and paid or
handed to a person (tests/test_approval_path.py covers that path). Each
executed step is committed with the run's record before the next one starts, so
a worker that dies loses at most the step in flight.

These tests commit, so each one gets its own scratch database.
"""

from datetime import UTC, datetime
from decimal import Decimal

import psycopg
import pytest
from psycopg import sql

from app.baseline import REFERENCE_RATE, token_cost
from app.contracts import Channel, IncomingMessage
from app.graph.build import build_graph
from app.intake import accept
from app.llm import ModelUnavailable, Reply
from app.retrieval import PolicySearchUnavailable
from app.run_agent import LostClaim, claim_next, work_next
from app.seed import load_ledger
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    EXTRACTED_4821,
    OUTAGE,
    PROPOSED_ESCALATE,
    PROPOSED_LOOKUP,
    PROPOSED_REFUND,
    FakeRetriever,
    ScriptedModel,
    proposed_refund,
)

pytestmark = pytest.mark.db

LOOKUP_3310 = (
    '{"tool": "get_order", "args": {"order_id": "3310"}, "confidence": 0.6,'
    ' "reasoning": "maybe the other order"}'
)
PROPOSED_SEARCH = (
    '{"tool": "search_policy", "args": {"question": "charged twice"}, "confidence": 0.5,'
    ' "reasoning": "no policy was found"}'
)


def graph_of(model, retriever=None):
    """Every driver test gets the same fixed policy passages unless it says otherwise."""
    return build_graph(model, retriever if retriever is not None else FakeRetriever())


def happy_model() -> ScriptedModel:
    """Look the order up, then propose the refund the ledger supports."""
    return ScriptedModel(
        classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, PROPOSED_REFUND]
    )


def happy_graph():
    return graph_of(happy_model())


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


def ledger(dsn: str) -> None:
    with psycopg.connect(dsn) as connection:
        load_ledger(connection)


def row(dsn: str, run_id: str) -> dict:
    with psycopg.connect(dsn) as connection:
        cursor = connection.execute(
            "SELECT status, current_node, state, cost_usd, attempt, locked_by, locked_at "
            "FROM runs WHERE id = %s",
            (run_id,),
        )
        names = [column.name for column in cursor.description]
        return dict(zip(names, cursor.fetchone(), strict=True))


def keys(dsn: str, run_id: str) -> list[str]:
    """Every tool call recorded for the run, in step order."""
    with psycopg.connect(dsn) as connection:
        return [
            key
            for (key,) in connection.execute(
                # By step number, not as text: 'step_10' sorts before 'step_2' as a string.
                "SELECT idempotency_key FROM tool_calls WHERE run_id = %s "
                "ORDER BY substring(idempotency_key from ':step_([0-9]+):')::int",
                (run_id,),
            ).fetchall()
        ]


def count(dsn: str, table: str) -> int:
    with psycopg.connect(dsn) as connection:
        query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
        return connection.execute(query).fetchone()[0]


def plan_prompts(model: ScriptedModel) -> list[str]:
    return [prompt for prompt in model.prompts if prompt.startswith("TASK: plan")]


# --- the ordinary case --------------------------------------------------------


def test_a_run_looks_the_order_up_then_refunds_what_the_ledger_supports(fresh_database):
    """Rs 3,600 at 0.9 confidence is under the default guardrail, so it is paid without waiting."""
    ledger(fresh_database)
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, happy_graph())

    assert str(outcome.run_id) == run_id
    assert (outcome.status, outcome.tool, outcome.steps) == ("done", "issue_refund", 2)
    stored = row(fresh_database, run_id)
    assert stored["status"] == "done"
    assert stored["current_node"] == "act"
    assert stored["state"]["agent"]["proposal"]["tool"] == "issue_refund"
    assert stored["state"]["agent"]["proposal"]["args"]["amount_paise"] == 360_000


def test_the_lookup_really_ran_against_the_ledger(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    steps = row(fresh_database, run_id)["state"]["agent"]["steps"]
    assert [step["tool"] for step in steps] == ["get_order", "issue_refund"]
    assert steps[0]["step"] == 1
    assert steps[0]["tool"] == "get_order"
    assert steps[0]["args"] == {"order_id": "4821"}
    assert steps[0]["result"]["charges_paise"] == [360_000, 360_000]
    assert steps[0]["replayed"] is False


def test_a_refund_over_the_limit_is_proposed_but_never_executed(fresh_database):
    """The lookup runs; Rs 7,200 waits for a person with an approval to decide."""
    ledger(fresh_database)
    queue(fresh_database)
    model = ScriptedModel(
        classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, proposed_refund(720_000)]
    )

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(model))

    assert count(fresh_database, "tool_calls") == 1
    assert count(fresh_database, "refunds") == 0
    assert count(fresh_database, "approvals") == 1


def test_each_executed_step_is_keyed_by_its_step_number(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:get_order", f"{run_id}:step_2:issue_refund"]


def test_what_the_lookup_returned_reaches_the_next_plan(fresh_database):
    ledger(fresh_database)
    queue(fresh_database)
    model = happy_model()

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(model))

    first, second = plan_prompts(model)
    assert "none yet" in first
    assert "360000" in second


def test_a_continuing_run_is_not_classified_again(fresh_database):
    ledger(fresh_database)
    queue(fresh_database)
    model = happy_model()

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(model))

    assert model.tasks() == ["classify", "extract", "plan", "plan"]


def test_what_the_steps_found_is_stored_beside_the_untouched_message(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    state = row(fresh_database, run_id)["state"]
    assert state["agent"]["classification"]["intent"] == "duplicate_charge"
    assert state["agent"]["extraction"]["order_id"] == "4821"
    assert state["agent"]["policy"]
    assert state["agent"]["policy_sources"] == ["duplicate-payments#1"]
    assert state["untrusted"]["body"].startswith("Hi, I think I was charged twice")


def test_the_run_is_charged_exactly_for_every_model_call(fresh_database):
    """classify, extract and two plans, each 10 prompt and 5 completion tokens."""
    ledger(fresh_database)
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    stored = row(fresh_database, run_id)
    expected = token_cost(prompt_tokens=40, completion_tokens=20, rate=REFERENCE_RATE)
    assert stored["cost_usd"] == expected.quantize(Decimal("0.000001"))
    assert stored["state"]["agent"]["model_calls"] == 4
    assert stored["state"]["agent"]["prompt_tokens"] == 40
    assert stored["state"]["agent"]["completion_tokens"] == 20


def test_each_step_is_committed_before_the_next_begins(fresh_database):
    """What a worker that dies right now would leave behind, seen from another connection."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    seen = []

    def look(step_run_id, step: int) -> None:
        stored = row(fresh_database, str(step_run_id))
        seen.append(
            (step, stored["status"], stored["locked_by"], len(stored["state"]["agent"]["steps"]),
             len(keys(fresh_database, run_id)))
        )

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph(), worker="worker-1", after_step=look)

    assert seen == [(1, "running", "worker-1", 1, 1)]


def test_the_lock_is_released_once_the_run_waits(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    stored = row(fresh_database, run_id)
    assert (stored["locked_by"], stored["locked_at"], stored["attempt"]) == (None, None, 1)


# --- which run, and only once ---------------------------------------------------


def test_an_empty_queue_asks_the_model_nothing(fresh_database):
    model = ScriptedModel()

    with psycopg.connect(fresh_database) as connection:
        assert work_next(connection, graph_of(model)) is None

    assert model.prompts == []


def test_the_oldest_queued_run_goes_first(fresh_database):
    ledger(fresh_database)
    later = queue(fresh_database, "late", minute=30)
    earlier = queue(fresh_database, "early", minute=5)

    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, happy_graph())

    assert str(outcome.run_id) == earlier
    assert row(fresh_database, later)["status"] == "queued"


def test_runs_that_are_not_queued_are_left_alone(fresh_database):
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET status = 'waiting_approval' WHERE id = %s", (run_id,))

    with psycopg.connect(fresh_database) as connection:
        assert work_next(connection, happy_graph()) is None


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
            work_next(connection, happy_graph())


# --- when the model does not cooperate ------------------------------------------


def test_an_escalation_is_recorded_executed_and_waits_for_a_person(fresh_database):
    run_id = queue(fresh_database)
    graph = graph_of(ScriptedModel(classify="no idea at all"))

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph)

    stored = row(fresh_database, run_id)
    assert stored["status"] == "waiting_approval"
    assert stored["current_node"] == "classify"
    assert stored["state"]["agent"]["proposal"]["tool"] == "escalate_to_human"
    assert "classify" in stored["state"]["agent"]["failure"]
    assert stored["cost_usd"] > 0, "the failed attempts were still paid for"
    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:escalate_to_human"]


def test_an_escalation_the_planner_proposes_runs_and_waits(fresh_database):
    run_id = queue(fresh_database)
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_ESCALATE)

    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(model))

    stored = row(fresh_database, run_id)
    assert (outcome.status, outcome.tool, outcome.failure) == ("waiting_approval", "escalate_to_human", None)
    assert stored["state"]["agent"]["steps"][0]["result"] == {"escalated": True, "reason": "status question"}


def test_a_repeated_proposal_is_escalated_not_run_again(fresh_database):
    """Seen as a risk with small models: asking for the same lookup forever."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_LOOKUP)

    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(model))

    stored = row(fresh_database, run_id)
    assert outcome.tool == "escalate_to_human"
    assert "repeated" in outcome.failure
    assert stored["status"] == "waiting_approval"
    assert keys(fresh_database, run_id) == [
        f"{run_id}:step_1:get_order",
        f"{run_id}:step_2:escalate_to_human",
    ]


def test_the_step_budget_hands_the_case_to_a_person(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    model = ScriptedModel(
        classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, LOOKUP_3310]
    )

    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(model), max_steps=1)

    assert outcome.tool == "escalate_to_human"
    assert "step budget" in outcome.failure
    assert keys(fresh_database, run_id) == [
        f"{run_id}:step_1:get_order",
        f"{run_id}:step_2:escalate_to_human",
    ]


def test_every_tool_the_model_can_propose_has_exactly_one_decision():
    """
    Found in security review. Whether a tool runs is decided only here, so a tool
    added to app/tools.py must be placed deliberately -- never run, or not run, by default.
    """
    from app.executor import TOOLS as IMPLEMENTED
    from app.guardrails import JUDGED
    from app.run_agent import GUARDED, NOT_RUN_HERE, RUNS_NOW
    from app.tools import TOOLS

    decisions = [RUNS_NOW, GUARDED, NOT_RUN_HERE]
    proposable = {tool.name for tool in TOOLS}

    assert set().union(*decisions) == proposable
    assert sum(len(decision) for decision in decisions) == len(proposable), "a tool is in two groups"
    assert RUNS_NOW | GUARDED <= set(IMPLEMENTED), "a decided tool has no implementation"
    assert GUARDED == {JUDGED}, "a guarded tool the guardrail does not judge would never be decided"


def test_a_proposal_the_executor_cannot_run_is_escalated(fresh_database):
    """search_policy is offered only when retrieval found nothing; searching again would find nothing too."""
    run_id = queue(fresh_database)
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_SEARCH)

    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(model, FakeRetriever(passages=[])))

    assert outcome.tool == "escalate_to_human"
    assert "search_policy" in outcome.failure
    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:escalate_to_human"]


def test_an_unreachable_model_puts_the_run_back_in_the_queue(fresh_database):
    """An outage is not a verdict on the case. The run waits for the model to return."""

    class Down:
        def generate(self, prompt: str) -> Reply:
            raise ModelUnavailable("connection refused")

    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph_of(Down()))

    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["locked_by"], stored["locked_at"]) == ("failed", None, None)
    assert "agent" not in stored["state"]
    assert stored["attempt"] == 1, "the attempt still counts toward max_attempts"


def test_an_outage_after_a_committed_step_keeps_it_and_resumes_there(fresh_database):
    """The lookup happened and was paid for. The next worker plans from it, without repeating it."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    first = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, OUTAGE])

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph_of(first))

    stored = row(fresh_database, run_id)
    assert stored["status"] == "failed"
    assert len(stored["state"]["agent"]["steps"]) == 1
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET next_retry_at = now() WHERE id = %s", (run_id,))

    second = ScriptedModel(plan=PROPOSED_REFUND)
    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(second))

    assert second.tasks() == ["plan"]
    assert (outcome.status, outcome.tool) == ("done", "issue_refund")
    assert keys(fresh_database, run_id) == [f"{run_id}:step_1:get_order", f"{run_id}:step_2:issue_refund"]


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
        work_next(connection, graph_of(Down()))

    with psycopg.connect(fresh_database) as connection:
        status, failure_class, locked_by = connection.execute(
            "SELECT status, failure_class, locked_by FROM runs WHERE id = %s", (run_id,)
        ).fetchone()
    assert (status, failure_class, locked_by) == ("dead", "model_unavailable", None)


class ReclaimedMidRun:
    """Another worker takes the run while the model thinks, as lock expiry will allow."""

    def __init__(self, dsn: str, inner=None, then_raise: Exception | None = None):
        self.dsn, self.inner, self.then_raise = dsn, inner, then_raise

    def stream(self, initial, stream_mode="values"):
        with psycopg.connect(self.dsn) as other:
            other.execute("UPDATE runs SET locked_by = 'worker-b', locked_at = now()")
        if self.then_raise is not None:
            raise self.then_raise
        yield from self.inner.stream(initial, stream_mode=stream_mode)


def test_a_step_is_not_executed_or_recorded_over_a_claim_that_was_lost(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    graph = ReclaimedMidRun(fresh_database, inner=happy_graph())

    with psycopg.connect(fresh_database) as connection, pytest.raises(LostClaim):
        work_next(connection, graph, worker="worker-a")

    stored = row(fresh_database, run_id)
    assert stored["locked_by"] == "worker-b"
    assert "agent" not in stored["state"]
    assert keys(fresh_database, run_id) == [], "a worker that lost its claim must execute nothing"


def test_an_outage_does_not_release_a_claim_that_was_lost(fresh_database):
    run_id = queue(fresh_database)
    graph = ReclaimedMidRun(fresh_database, then_raise=ModelUnavailable("down"))

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph, worker="worker-a")

    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["locked_by"]) == ("running", "worker-b")


def test_an_outage_partway_through_still_charges_for_finished_steps(fresh_database):
    """
    Found in milestone review. classify answered, then the model went down. The
    run goes back to the queue, but the classify call was paid for and is charged.
    """
    run_id = queue(fresh_database)
    graph = graph_of(ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=OUTAGE))

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph)

    stored = row(fresh_database, run_id)
    assert stored["status"] == "failed"
    assert stored["cost_usd"] == token_cost(10, 5, REFERENCE_RATE).quantize(Decimal("0.000001"))


class PolicyStoreDown:
    def search(self, question: str):
        raise PolicySearchUnavailable("policy search could not reach the database")


def test_a_policy_store_outage_puts_the_run_back_in_the_queue(fresh_database):
    """classify and extract were paid for; the database then dropped during retrieval."""
    run_id = queue(fresh_database)
    graph = build_graph(ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821), PolicyStoreDown())

    with psycopg.connect(fresh_database) as connection, pytest.raises(PolicySearchUnavailable):
        work_next(connection, graph)

    stored = row(fresh_database, run_id)
    assert (stored["status"], stored["locked_by"]) == ("failed", None)
    assert stored["cost_usd"] == token_cost(20, 10, REFERENCE_RATE).quantize(Decimal("0.000001"))


def test_a_policy_store_outage_on_the_last_attempt_says_what_failed(fresh_database):
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET attempt = max_attempts - 1 WHERE id = %s", (run_id,))
    graph = build_graph(ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821), PolicyStoreDown())

    with psycopg.connect(fresh_database) as connection, pytest.raises(PolicySearchUnavailable):
        work_next(connection, graph)

    with psycopg.connect(fresh_database) as connection:
        status, failure_class = connection.execute(
            "SELECT status, failure_class FROM runs WHERE id = %s", (run_id,)
        ).fetchone()
    assert (status, failure_class) == ("dead", "policy_search_unavailable")


def test_the_driver_brings_an_unmigrated_database_up_to_date(empty_database):
    """app.db promises migrations are safe on every start-up; the driver now relies on it."""
    from app.run_agent import prepare_database

    with psycopg.connect(empty_database) as connection:
        passages = prepare_database(connection)
        columns = {
            row[0]
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'policy_chunks'"
            ).fetchall()
        }

    assert passages == 0
    assert "embedding_model" in columns


def test_the_driver_reports_how_many_policy_passages_are_loaded(fresh_database):
    """Zero means app.policies was never run, which the driver warns about."""
    from app.policies import ingest
    from app.run_agent import prepare_database
    from tests.fakes import FakeEmbedder

    with psycopg.connect(fresh_database) as connection:
        ingest(connection, FakeEmbedder(), {"d": "# D\n\n## One\n\nfirst\n\n## Two\n\nsecond"})

    with psycopg.connect(fresh_database) as connection:
        assert prepare_database(connection) == 2
