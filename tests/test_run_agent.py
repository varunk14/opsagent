"""
The driver: take a queued run and work it one committed step at a time.

The tools run. get_order and escalate_to_human execute through the
keyed executor; issue_refund is judged by the guardrail, and paid or
handed to a person (tests/test_approval_path.py covers that path). Each
executed step is committed with the run's record before the next one starts, so
a worker that dies loses at most the step in flight.

These tests commit, so each one gets its own scratch database.
"""

import json
from datetime import UTC, datetime
from decimal import Decimal

import psycopg
import pytest
from psycopg import sql

from app.baseline import REFERENCE_RATE, token_cost
from app.contracts import Channel, IncomingMessage, ProposedAction
from app.graph.build import build_graph
from app.graph.prompts import run_prompt_version
from app.intake import accept
from app.llm import DEFAULT_MODEL, ModelUnavailable, Reply
from app.retrieval import PolicySearchUnavailable
from app.run_agent import ALREADY_SHOWN, MICRO_DOLLAR, LostClaim, claim_next, work_next
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


# --- the prompt version a run was worked under ------------------------------------


def prompt_version_of(dsn: str, run_id: str) -> str | None:
    with psycopg.connect(dsn) as connection:
        return connection.execute("SELECT prompt_version FROM runs WHERE id = %s", (run_id,)).fetchone()[0]


def test_a_worked_run_records_the_prompt_version_it_was_planned_under(fresh_database):
    """Which prompts produced an outcome is on the run, so a later change in behaviour is attributable."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    assert prompt_version_of(fresh_database, run_id) is None

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    assert prompt_version_of(fresh_database, run_id) == run_prompt_version()


def test_a_run_handed_to_a_person_records_the_prompt_version(fresh_database):
    run_id = queue(fresh_database)
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_ESCALATE)

    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(model))

    assert outcome.status == "waiting_approval"
    assert prompt_version_of(fresh_database, run_id) == run_prompt_version()


def test_a_run_that_hit_an_outage_records_the_prompt_version_it_was_tried_under(fresh_database):
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph_of(ScriptedModel(classify=OUTAGE)))

    assert row(fresh_database, run_id)["status"] == "failed"
    assert prompt_version_of(fresh_database, run_id) == run_prompt_version()


# --- what a run has been charged for, read back ---------------------------------------


def test_the_charged_totals_are_the_larger_of_a_ticks_and_an_outages(fresh_database):
    """An outage records what it charged under billing; the next tick must count on from it."""
    from uuid import uuid4

    from psycopg.types.json import Jsonb

    from app.run_agent import agent_of

    run_id = uuid4()
    state = {
        "agent": {"prompt_tokens": 5, "completion_tokens": 7, "steps": []},
        "billing": {"prompt_tokens": 9, "model_calls": 2},
    }
    with psycopg.connect(fresh_database) as connection:
        connection.execute(
            "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
            "VALUES (%s, 'email', 'queued', 'intake', %s, %s)",
            (run_id, Jsonb(state), f"email_msg_{run_id.hex}"),
        )
        agent = agent_of(connection, run_id)

    assert (agent["prompt_tokens"], agent["completion_tokens"], agent["model_calls"]) == (9, 7, 2)
    assert agent["steps"] == []


def test_a_run_that_is_not_there_has_recorded_nothing(fresh_database):
    from uuid import uuid4

    from app.run_agent import agent_of

    with psycopg.connect(fresh_database) as connection:
        assert agent_of(connection, uuid4()) == {}


def test_outage_totals_are_folded_in_and_cleared_once_a_tick_records_its_own(fresh_database):
    """Kept only until the run's own totals include them; left behind, they would read as current."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph_of(ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=OUTAGE)))
    assert row(fresh_database, run_id)["state"]["billing"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "model_calls": 1,
        "tokens_by_model": {DEFAULT_MODEL: [10, 5]},
    }
    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET next_retry_at = now() WHERE id = %s", (run_id,))

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph())

    state = row(fresh_database, run_id)["state"]
    assert "billing" not in state
    agent = state["agent"]
    assert (agent["prompt_tokens"], agent["completion_tokens"], agent["model_calls"]) == (50, 25, 5)


# --- a repeated lookup is asked once more before anyone is woken ------------------------------


def repeating_model(*after_the_repeat: str) -> ScriptedModel:
    """
    Look 4821 up, ask for the same lookup again, then whatever `after_the_repeat` says.

    The planner asking for a result it has already been shown is the single largest cause of
    incomplete runs. It is not a reason to wake a person yet: the run is asked once more with
    that result pointed at, and only a second repeat hands the case over.
    """
    return ScriptedModel(
        classify=CLASSIFIED_DUPLICATE,
        extract=EXTRACTED_4821,
        plan=[PROPOSED_LOOKUP, PROPOSED_LOOKUP, *after_the_repeat],
    )


def test_a_repeated_lookup_is_asked_again_with_its_result_pointed_at(fresh_database):
    ledger(fresh_database)
    queue(fresh_database)
    model = repeating_model(proposed_refund(360_000))

    with psycopg.connect(fresh_database) as connection:
        # One call works the run to rest: the lookup, then the repeat and the second ask.
        outcome = work_next(connection, graph_of(model))

    assert outcome.status == "done"
    assert outcome.failure is None
    asked = plan_prompts(model)
    assert ALREADY_SHOWN in asked[-1], "the repeated result is pointed at in the second ask"
    assert ALREADY_SHOWN not in asked[0], "and not in an ask that repeated nothing"


def test_the_second_ask_sees_the_result_it_asked_for_again(fresh_database):
    ledger(fresh_database)
    queue(fresh_database)
    model = repeating_model(proposed_refund(360_000))

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(model))

    second = plan_prompts(model)[-1]
    assert '"charges_paise"' in second and "360000" in second


def test_repeating_a_second_time_hands_the_case_over(fresh_database):
    ledger(fresh_database)
    queue(fresh_database)
    model = repeating_model(PROPOSED_LOOKUP)

    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(model))

    assert outcome.status == "waiting_approval"
    assert outcome.failure == "plan: repeated an earlier step"
    assert count(fresh_database, "refunds") == 0


def test_both_asks_are_charged_for(fresh_database):
    """The run is asked twice, so it pays for two plan calls: nothing is charged for free."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    model = repeating_model(proposed_refund(360_000))

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(model))

    agent = row(fresh_database, run_id)["state"]["agent"]
    assert agent["model_calls"] == 5, "classify, extract and plan, then plan and plan again"
    assert (agent["prompt_tokens"], agent["completion_tokens"]) == (50, 25), "the tokens of both asks"


def test_the_marker_is_never_written_into_what_the_run_stores(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    model = repeating_model(proposed_refund(360_000))

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(model))

    steps = row(fresh_database, run_id)["state"]["agent"]["steps"]
    assert all(ALREADY_SHOWN not in json.dumps(step) for step in steps)


def test_an_outage_during_the_second_ask_charges_for_both_and_returns_the_run(fresh_database):
    """
    The second ask is a model call like any other, so an outage in it is an outage in the tick.

    The run goes back to the queue with the calls that did complete charged -- including the ask
    that surfaced the repeat, which finished before the outage and would otherwise be spent and
    never recorded.
    """
    ledger(fresh_database)
    run_id = queue(fresh_database)
    model = repeating_model(OUTAGE)

    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, graph_of(model))

    run = row(fresh_database, run_id)
    assert run["status"] == "failed", "returned to the queue, not left claimed"
    assert run["locked_by"] is None
    billing = run["state"]["billing"]
    assert billing["model_calls"] == 4, "classify, extract, the plan that repeated, and the retry"
    assert (billing["prompt_tokens"], billing["completion_tokens"]) == (40, 20)


def test_a_hand_over_for_any_other_reason_is_not_asked_again(fresh_database):
    """Only a repeat earns a second ask. A tool this worker does not run is settled, not confused."""
    ledger(fresh_database)
    queue(fresh_database)
    model = ScriptedModel(
        classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, PROPOSED_SEARCH]
    )

    with psycopg.connect(fresh_database) as connection:
        outcome = work_next(connection, graph_of(model))

    assert outcome.failure == "plan: search_policy is not a tool this worker runs"
    assert len(plan_prompts(model)) == 2, "the planner was asked once per tick and no more"


def test_only_the_result_that_was_repeated_is_pointed_at(fresh_database):
    """The mark says which answer the planner already has. Marking them all would say nothing."""
    ledger(fresh_database)
    queue(fresh_database)
    model = ScriptedModel(
        classify=CLASSIFIED_DUPLICATE,
        extract=EXTRACTED_4821,
        plan=[PROPOSED_LOOKUP, LOOKUP_3310, PROPOSED_LOOKUP, proposed_refund(360_000)],
    )

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph_of(model))

    second = plan_prompts(model)[-1]
    assert second.count(ALREADY_SHOWN) == 1, "one of the two lookups, not both"
    assert '"order_id": "4821"' in second and '"order_id": "3310"' in second


def test_a_step_is_the_same_step_only_when_the_tool_is_the_same_too():
    """
    A step is identified by what was called as well as how. Two tools given the same arguments
    are two different things asked, and only the arguments matching is not a repeat -- otherwise
    a tool added later that happens to take an order id would be mistaken for one already run.
    """
    from app.run_agent import repeats

    arguments = {"order_id": "4821"}
    lookup = ProposedAction(
        tool="get_order", args=arguments, confidence=Decimal("0.9"), reasoning="confirm the charges"
    )

    assert repeats(lookup, {"tool": "get_order", "args": arguments, "result": {}}) is True
    assert repeats(lookup, {"tool": "issue_refund", "args": arguments, "result": {}}) is False
    assert repeats(lookup, {"tool": "get_order", "args": {"order_id": "3310"}, "result": {}}) is False


# --- a run is charged at the rate of each model that worked on it -----------------------------


def test_a_reply_says_which_model_produced_it():
    """Without it, the run cannot be priced: tokens alone do not say what they cost."""
    from app.llm import DEFAULT_MODEL, Reply

    assert Reply(text="{}", prompt_tokens=1, completion_tokens=1, latency_ms=1).model == DEFAULT_MODEL


def test_a_run_worked_by_two_models_is_charged_at_both_rates():
    from app.baseline import cost_of
    from app.llm import DEFAULT_MODEL, SMALL_MODEL, Reply
    from app.run_agent import charge

    replies = [
        Reply(text="{}", prompt_tokens=1_000, completion_tokens=100, latency_ms=1, model=SMALL_MODEL),
        Reply(text="{}", prompt_tokens=2_000, completion_tokens=200, latency_ms=1, model=DEFAULT_MODEL),
    ]

    cost, _, _ = charge({}, replies)

    assert cost == cost_of({SMALL_MODEL: (1_000, 100), DEFAULT_MODEL: (2_000, 200)}).quantize(MICRO_DOLLAR)


def test_the_same_tokens_cost_less_on_the_small_model():
    from app.llm import DEFAULT_MODEL, SMALL_MODEL, Reply
    from app.run_agent import charge

    def cost_on(model: str):
        return charge({}, [Reply(text="{}", prompt_tokens=100_000, completion_tokens=0, latency_ms=1, model=model)])[0]

    assert cost_on(SMALL_MODEL) < cost_on(DEFAULT_MODEL)


def test_what_each_model_spent_is_kept_on_the_run_so_a_later_tick_counts_on_from_it():
    from app.llm import DEFAULT_MODEL, SMALL_MODEL, Reply
    from app.run_agent import charge, tokens_by_model

    first = tokens_by_model({}, [Reply(text="{}", prompt_tokens=10, completion_tokens=1, latency_ms=1, model=SMALL_MODEL)])
    second = tokens_by_model(
        {"tokens_by_model": first},
        [Reply(text="{}", prompt_tokens=20, completion_tokens=2, latency_ms=1, model=DEFAULT_MODEL)],
    )

    assert first == {SMALL_MODEL: [10, 1]}
    assert second == {SMALL_MODEL: [10, 1], DEFAULT_MODEL: [20, 2]}
    assert charge({"tokens_by_model": first}, []) [0] == Decimal("0.000000"), "no new replies, no new charge"


def test_the_larger_count_per_model_wins_whichever_side_holds_it():
    """
    Billing counts on from what the last successful tick recorded, so it should never be behind --
    but the fold says so rather than assuming it, exactly as the other charged totals do, and a
    model only one side has seen is kept either way.
    """
    from app.run_agent import most_spent

    assert most_spent({"a": [10, 2]}, {"a": [4, 5]}) == {"a": [10, 5]}
    assert most_spent({"a": [1, 1]}, {"b": [2, 2]}) == {"a": [1, 1], "b": [2, 2]}
    assert most_spent({}, {"a": [3, 4]}) == {"a": [3, 4]}
