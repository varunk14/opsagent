"""
The worker's spans: one trace per run, written with the step they describe.

Every tick is a root span holding the graph's steps and an `act` span for what the
tick did about the proposal: the tool that ran, what the guardrail decided, the
approval it opened. The spans go into the spans table in the transaction that
commits the step, so the table holds exactly what committed -- a tick that lost its
claim leaves nothing behind -- and paying an approval days later is one more tick in
the same trace.

Whatever path a run takes, the costs of its model calls, added up exactly and
rounded once, are the cost recorded on the run: the trace view shows per-step costs,
and they must agree with the total beside them.
"""

import json
from collections.abc import Callable
from decimal import Decimal
from uuid import UUID

import psycopg
import pytest

import app.run_agent as agent
from app import tracing
from app.graph.build import build_graph
from app.graph.prompts import PROMPT_VERSIONS, run_prompt_version
from app.llm import ModelUnavailable, Reply
from app.run_agent import MICRO_DOLLAR, LostClaim, work_next
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    CLASSIFIED_STATUS,
    EXTRACTED_4821,
    OUTAGE,
    PROPOSED_ESCALATE,
    PROPOSED_LOOKUP,
    PROPOSED_REFUND,
    FakeRetriever,
    ScriptedModel,
)
from tests.test_approval_path import MustNotBeAsked, decide_on, refund_model, work
from tests.test_rate_limit import PRIYA, past_lookups
from tests.test_run_agent import (
    ReclaimedMidRun,
    happy_graph,
    happy_model,
    ledger,
    queue,
    row,
)

pytestmark = pytest.mark.db


def spans_of(dsn: str, run_id: str) -> list[dict]:
    with psycopg.connect(dsn) as connection:
        cursor = connection.execute(
            "SELECT * FROM spans WHERE trace_id = %s ORDER BY started_at, parent_span_id IS NOT NULL, span_id",
            (run_id,),
        )
        names = [column.name for column in cursor.description]
        return [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]


def ticks(spans: list[dict]) -> list[dict]:
    return [span for span in spans if span["parent_span_id"] is None]


def children(spans: list[dict], parent: dict) -> list[dict]:
    return [span for span in spans if span["parent_span_id"] == parent["span_id"]]


def names(spans: list[dict]) -> list[str]:
    return [span["name"] for span in spans]


def only(spans: list[dict], name: str) -> dict:
    (found,) = [span for span in spans if span["name"] == name]
    return found


def calls_cost(spans: list[dict]) -> Decimal:
    """What the trace says the run's model calls cost: added up exactly, rounded once."""
    return sum((span["cost_usd"] for span in spans if span["kind"] == "generation"), Decimal(0)).quantize(MICRO_DOLLAR)


def retry_now(dsn: str, run_id: str) -> None:
    with psycopg.connect(dsn) as connection:
        connection.execute("UPDATE runs SET next_retry_at = now() WHERE id = %s", (run_id,))


def fail_with_an_outage(dsn: str, model: ScriptedModel) -> None:
    with psycopg.connect(dsn) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, build_graph(model, FakeRetriever()))


# --- the shape of a run's trace -----------------------------------------------------


def test_a_run_is_one_trace_of_ticks_each_holding_its_steps(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    work(fresh_database, happy_model())

    spans = spans_of(fresh_database, run_id)
    first, second = ticks(spans)
    assert names(children(spans, first)) == ["classify", "extract", "retrieve", "plan", "act"]
    assert names(children(spans, second)) == ["plan", "act"]
    assert {span["trace_id"] for span in spans} == {UUID(run_id)}
    assert (first["attributes"]["opsagent.attempt"], first["attributes"]["opsagent.outcome"]) == (1, "running")
    assert second["attributes"]["opsagent.outcome"] == "done"


def test_each_act_says_which_tool_ran_and_what_came_of_it(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    work(fresh_database, happy_model())

    lookup, refund = [span for span in spans_of(fresh_database, run_id) if span["name"] == "act"]
    assert (lookup["kind"], lookup["attributes"]["opsagent.tool"], lookup["attributes"]["opsagent.result"]) == (
        "tool",
        "get_order",
        "looked up",
    )
    assert (refund["attributes"]["opsagent.tool"], refund["attributes"]["opsagent.result"]) == ("issue_refund", "refunded")


def test_a_refund_act_records_the_guardrail_verdict_and_the_limits_in_force(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    work(fresh_database, happy_model())

    spans = spans_of(fresh_database, run_id)
    refund = [span for span in spans if span["name"] == "act"][-1]
    (guardrail,) = children(spans, refund)
    assert guardrail["kind"] == "guardrail"
    found = guardrail["attributes"]
    assert (found["opsagent.verdict"], found["opsagent.limit_paise"], found["opsagent.min_confidence"]) == (
        "runs",
        500_000,
        "0.85",
    )


def test_a_refund_over_the_limit_records_the_approval_it_opened(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    work(fresh_database, refund_model(720_000))

    spans = spans_of(fresh_database, run_id)
    refund = [span for span in spans if span["name"] == "act"][-1]
    with psycopg.connect(fresh_database) as connection:
        (approval_id,) = connection.execute("SELECT id FROM approvals WHERE run_id = %s", (run_id,)).fetchone()
    assert (refund["attributes"]["opsagent.result"], refund["attributes"]["opsagent.approval_id"]) == (
        "approval opened",
        approval_id,
    )
    guardrail = only(children(spans, refund), "guardrail")["attributes"]
    assert guardrail["opsagent.verdict"] == "needs a person"
    assert guardrail["opsagent.reason"].startswith("Rs 7,200 is not under the Rs 5,000 limit")
    assert ticks(spans)[-1]["attributes"]["opsagent.outcome"] == "waiting_approval"


def test_paying_an_approved_refund_is_one_more_tick_in_the_same_trace(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(720_000))
    decide_on(fresh_database, run_id, approved=True)

    work(fresh_database, MustNotBeAsked())

    spans = spans_of(fresh_database, run_id)
    payment = ticks(spans)[-1]
    assert len(ticks(spans)) == 3
    assert payment["attributes"]["opsagent.approval_id"] > 0
    assert payment["attributes"]["opsagent.outcome"] == "done"
    (act,) = children(spans, payment)
    assert (act["attributes"]["opsagent.tool"], act["attributes"]["opsagent.result"]) == ("issue_refund", "refunded")


def test_a_run_handed_to_a_person_says_so_on_its_act(fresh_database, exported):
    run_id = queue(fresh_database)

    work(fresh_database, ScriptedModel(classify=CLASSIFIED_STATUS, plan=PROPOSED_ESCALATE))

    act = only(spans_of(fresh_database, run_id), "act")["attributes"]
    assert (act["opsagent.tool"], act["opsagent.result"]) == ("escalate_to_human", "handed to a person")


def test_every_generation_carries_the_version_of_its_prompt(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    work(fresh_database, happy_model())

    calls = [span for span in spans_of(fresh_database, run_id) if span["kind"] == "generation"]
    assert [span["prompt_version"] for span in calls] == [
        PROMPT_VERSIONS["classify"],
        PROMPT_VERSIONS["extract"],
        PROMPT_VERSIONS["plan"],
        PROMPT_VERSIONS["plan"],
    ]
    with psycopg.connect(fresh_database) as connection:
        (version,) = connection.execute("SELECT prompt_version FROM runs WHERE id = %s", (run_id,)).fetchone()
    assert version == run_prompt_version()


# --- written with the step, or not at all ---------------------------------------------


def test_a_tick_that_lost_its_claim_leaves_no_spans(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection, pytest.raises(LostClaim):
        work_next(connection, ReclaimedMidRun(fresh_database, inner=happy_graph()), worker="worker-a")

    assert spans_of(fresh_database, run_id) == []
    assert tracing._installed.recorder.take(UUID(run_id)) == [], "what it held was dropped, not left for later"


def test_an_outage_writes_the_tick_that_failed_and_why(fresh_database, exported):
    run_id = queue(fresh_database)

    fail_with_an_outage(fresh_database, ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=OUTAGE))

    spans = spans_of(fresh_database, run_id)
    (tick,) = ticks(spans)
    assert (tick["status"], tick["status_message"], tick["attributes"]["opsagent.outcome"]) == (
        "error",
        "model_unavailable",
        "failed",
    )
    assert only(spans, "extract")["status"] == "error"
    assert names(children(spans, tick)) == ["classify", "extract"]


def test_a_deferred_tick_writes_its_spans_and_says_it_was_deferred(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    past_lookups(fresh_database, PRIYA, agent.RATE_LIMIT)

    work(fresh_database, happy_model())

    spans = spans_of(fresh_database, run_id)
    (tick,) = ticks(spans)
    assert tick["attributes"]["opsagent.outcome"] == "queued"
    act = only(children(spans, tick), "act")["attributes"]
    assert (act["opsagent.tool"], act["opsagent.result"]) == ("get_order", "deferred by the rate limit")


def test_no_span_names_the_worker(fresh_database, exported):
    """locked_by holds a host name and pid; it stays off every screen and out of every trace."""
    ledger(fresh_database)
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        work_next(connection, happy_graph(), worker="worker-on-host-4242")

    for span in spans_of(fresh_database, run_id):
        written = json.dumps(span["attributes"]) + (span["status_message"] or "") + span["name"]
        assert "worker-on-host" not in written, span["name"]
        assert agent.default_worker() not in written, span["name"]


# --- the costs in the trace add up to the cost on the run -----------------------------


def plain(dsn: str) -> str:
    ledger(dsn)
    run_id = queue(dsn)
    work(dsn, happy_model())
    return run_id


def retried_answer(dsn: str) -> str:
    ledger(dsn)
    run_id = queue(dsn)
    work(
        dsn,
        ScriptedModel(
            classify=["no idea", CLASSIFIED_DUPLICATE], extract=EXTRACTED_4821, plan=[PROPOSED_LOOKUP, PROPOSED_REFUND]
        ),
    )
    return run_id


def deferred_then_served(dsn: str) -> str:
    ledger(dsn)
    run_id = queue(dsn)
    past_lookups(dsn, PRIYA, agent.RATE_LIMIT)
    work(dsn, happy_model())
    with psycopg.connect(dsn) as connection:
        connection.execute("UPDATE tool_calls SET created_at = now() - interval '2 hours' WHERE run_id <> %s", (run_id,))
    retry_now(dsn, run_id)
    work(dsn, ScriptedModel(plan=[PROPOSED_LOOKUP, PROPOSED_REFUND]))
    return run_id


def outage_then_resumed(dsn: str) -> str:
    ledger(dsn)
    run_id = queue(dsn)
    fail_with_an_outage(dsn, ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=OUTAGE))
    retry_now(dsn, run_id)
    work(dsn, happy_model())
    return run_id


def approved_and_paid(dsn: str) -> str:
    ledger(dsn)
    run_id = queue(dsn)
    work(dsn, refund_model(720_000))
    decide_on(dsn, run_id, approved=True)
    work(dsn, MustNotBeAsked())
    return run_id


@pytest.mark.parametrize("path", [plain, retried_answer, deferred_then_served, outage_then_resumed, approved_and_paid])
def test_the_costs_in_the_trace_add_up_to_the_cost_on_the_run(fresh_database, exported, path: Callable[[str], str]):
    run_id = path(fresh_database)

    spans = spans_of(fresh_database, run_id)
    assert calls_cost(spans) > 0
    assert calls_cost(spans) == row(fresh_database, run_id)["cost_usd"]


class Cheap(ScriptedModel):
    """Replies costing a fraction of a micro-dollar each, so a charge rounded on its own shows."""

    def __init__(self, prompt_tokens: int, **replies):
        super().__init__(**replies)
        self.prompt_tokens = prompt_tokens

    def generate(self, prompt: str) -> Reply:
        reply = super().generate(prompt)
        return Reply(text=reply.text, prompt_tokens=self.prompt_tokens, completion_tokens=0, latency_ms=1)


def test_an_outage_charge_is_not_rounded_apart_from_the_rest_of_the_run(fresh_database, exported):
    """
    3 prompt tokens cost $0.00000045 and 2 cost $0.0000003: each rounds to nothing on
    its own, while together they round to a micro-dollar. A run charged the first
    at an outage and the second on its next tick must still be charged $0.000001.
    """
    run_id = queue(fresh_database)
    fail_with_an_outage(fresh_database, Cheap(3, classify=CLASSIFIED_DUPLICATE, extract=OUTAGE))
    retry_now(fresh_database, run_id)

    work(fresh_database, Cheap(1, classify=CLASSIFIED_STATUS, plan=PROPOSED_ESCALATE))

    spans = spans_of(fresh_database, run_id)
    assert calls_cost(spans) == Decimal("0.000001")
    assert row(fresh_database, run_id)["cost_usd"] == Decimal("0.000001")
