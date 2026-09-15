"""
Spans for the graph: one per step, one per model call inside it, all under the tick
that walked the graph.

A model call is a `generation` carrying the model, tokens in and out, the reference
cost, the prompt version and the latency. Every call is one, including the retries of
an unusable answer, because every one of them was paid for. Cost lives on those calls
and nowhere else: a step's cost is the sum of its calls, worked out by whoever reads
the trace, so nothing can be counted twice.

A call that never answered is recorded as an error, not as a generation: a
generation with no tokens would read as a free model call.

What the steps decided is recorded -- the intent, the proposed tool -- but not what
the customer wrote. The run row already holds that, and a span is a copy that leaves
for Langfuse.
"""

import json
from decimal import Decimal
from uuid import UUID

import pytest

from app.baseline import REFERENCE_RATE, token_cost
from app.graph.build import build_graph, run_graph
from app.graph.prompts import prompt_version
from app.llm import ModelUnavailable
from app.tracing import Attr, SpanRow, rows_for, run_context, tracer
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    CLASSIFIED_STATUS,
    EXTRACTED_4821,
    OUTAGE,
    PROPOSED_ESCALATE,
    PROPOSED_LOOKUP,
    FakeRetriever,
    ScriptedModel,
)

RUN = UUID("a2a97b37-62fb-4a4d-ab07-60172aa05ef9")
SUBJECT = "Charged twice for order #4821"
BODY = "Hi, I think I was charged twice for order #4821 last Tuesday. My card ends 4242."
# ScriptedModel answers every call with these counts.
IN, OUT = 10, 5


class NamedModel(ScriptedModel):
    """A model that says which model it is, as the real Ollama client does."""

    model = "llama3.1:8b"


def walk(exported, model, retriever=None) -> list[SpanRow]:
    with run_context(RUN), tracer().start_as_current_span("tick"):
        run_graph(build_graph(model, retriever or FakeRetriever()), SUBJECT, BODY)
    return rows_for(exported.get_finished_spans())


def named(rows: list[SpanRow], name: str) -> SpanRow:
    (row,) = [row for row in rows if row.name == name]
    return row


def children(rows: list[SpanRow], name: str) -> list[str]:
    parent = named(rows, name)
    return [row.name for row in rows if row.parent_span_id == parent.span_id]


def happy() -> ScriptedModel:
    return NamedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_LOOKUP)


def test_each_step_is_a_span_under_the_tick(exported):
    rows = walk(exported, happy())

    assert children(rows, "tick") == ["classify", "extract", "retrieve", "plan"]
    assert {named(rows, step).kind for step in ("classify", "extract", "retrieve", "plan")} == {"chain"}
    assert {row.trace_id for row in rows} == {RUN}


def test_each_model_call_is_a_generation_under_its_step(exported):
    rows = walk(exported, happy())

    assert children(rows, "classify") == ["classify.generate"]
    assert children(rows, "extract") == ["extract.generate"]
    assert children(rows, "plan") == ["plan.generate"]
    assert children(rows, "retrieve") == []
    assert [row.name for row in rows if row.kind == "generation"] == [
        "classify.generate",
        "extract.generate",
        "plan.generate",
    ]


def test_a_generation_carries_model_tokens_cost_version_and_latency(exported):
    rows = walk(exported, happy())

    call = named(rows, "plan.generate")
    assert (call.model, call.input_tokens, call.output_tokens) == ("llama3.1:8b", IN, OUT)
    assert call.cost_usd == token_cost(IN, OUT, REFERENCE_RATE)
    assert call.prompt_version == prompt_version("plan")
    assert call.attributes[Attr.PROMPT_VERSION_METADATA] == prompt_version("plan")
    assert call.attributes[Attr.LATENCY_MS] == 1
    assert Decimal(str(json.loads(call.attributes[Attr.COST_DETAILS])["total"])) == call.cost_usd


def test_a_model_that_does_not_name_itself_is_recorded_by_its_class(exported):
    rows = walk(exported, ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_LOOKUP))

    assert named(rows, "classify.generate").model == "ScriptedModel"


def test_every_retry_is_its_own_costed_generation(exported):
    """Three goes at classify cost three calls; counting only the one that worked would understate cost."""
    model = NamedModel(classify=["no idea", "still no idea", CLASSIFIED_DUPLICATE], extract=EXTRACTED_4821, plan=PROPOSED_LOOKUP)

    rows = walk(exported, model)

    assert children(rows, "classify") == ["classify.generate"] * 3
    calls = [row for row in rows if row.kind == "generation"]
    assert len(calls) == 5
    assert sum(call.cost_usd for call in calls) == token_cost(5 * IN, 5 * OUT, REFERENCE_RATE)


def test_steps_carry_no_cost_of_their_own(exported):
    """A step's cost is the sum of its calls; stored twice, a page adding up the trace would count it twice."""
    rows = walk(exported, happy())

    assert [row.cost_usd for row in rows if row.kind != "generation"] == [None] * 5


def test_a_status_question_runs_the_extract_step_without_a_model_call(exported):
    rows = walk(exported, NamedModel(classify=CLASSIFIED_STATUS, plan=PROPOSED_ESCALATE))

    assert children(rows, "tick") == ["classify", "extract", "retrieve", "plan"]
    assert children(rows, "extract") == []


def test_what_each_step_decided_is_recorded(exported):
    rows = walk(exported, happy())

    classify = named(rows, "classify").attributes
    assert (classify["opsagent.intent"], classify["opsagent.confidence"]) == ("duplicate_charge", "0.9")
    assert named(rows, "extract").attributes["opsagent.order_id_found"] is True
    assert named(rows, "retrieve").attributes["opsagent.passages"] == 1
    plan = named(rows, "plan").attributes
    assert (plan["opsagent.tool"], plan["opsagent.confidence"]) == ("get_order", "0.8")


def test_no_customer_text_goes_into_any_span(exported):
    rows = walk(exported, happy())

    for row in rows:
        written = json.dumps(row.attributes) + (row.status_message or "")
        assert "4242" not in written, row.name
        assert "last Tuesday" not in written, row.name
        assert SUBJECT not in written, row.name


def test_an_escalating_step_is_marked_but_is_not_an_error(exported):
    """An unusable answer is a verdict the step reached; the model worked, so nothing failed."""
    rows = walk(exported, NamedModel(classify="no idea"))

    assert children(rows, "tick") == ["classify"]
    classify = named(rows, "classify")
    assert classify.status == "ok"
    assert classify.attributes["opsagent.failure"] == "classify: model output unusable"
    assert classify.attributes["opsagent.tool"] == "escalate_to_human"


def test_a_model_that_never_answered_is_an_error_and_not_a_free_generation(exported):
    model = NamedModel(classify=CLASSIFIED_DUPLICATE, extract=OUTAGE)

    with pytest.raises(ModelUnavailable):
        walk(exported, model)

    rows = rows_for(exported.get_finished_spans())
    call = named(rows, "extract.generate")
    assert (call.kind, call.status, call.status_message) == ("span", "error", "model_unavailable")
    assert (call.input_tokens, call.cost_usd) == (None, None)
    step = named(rows, "extract")
    assert (step.status, step.status_message) == ("error", "model_unavailable")
    assert named(rows, "classify.generate").kind == "generation"


def test_an_outage_records_no_exception_text(exported):
    """An exception message can quote whatever the model or the customer wrote; the failure class is enough."""
    with pytest.raises(ModelUnavailable):
        walk(exported, NamedModel(classify=OUTAGE))

    for span in exported.get_finished_spans():
        if span.name in ("classify", "classify.generate"):
            assert [event.name for event in span.events] == []


def test_an_unexpected_error_is_recorded_by_its_type_and_nothing_it_said(exported):
    """A bug is not an outage, but its message is no safer to copy into a trace than an outage's."""

    class Broken(NamedModel):
        def generate(self, prompt: str):
            raise RuntimeError("the customer wrote: my card ends 4242")

    with pytest.raises(RuntimeError):
        walk(exported, Broken())

    spans = {span.name: span for span in exported.get_finished_spans()}
    rows = rows_for(exported.get_finished_spans())
    for name in ("classify", "classify.generate"):
        row = named(rows, name)
        assert (row.status, row.status_message) == ("error", "RuntimeError")
        assert [event.name for event in spans[name].events] == []


def test_tracing_changes_nothing_the_graph_returns(exported):
    with run_context(RUN):
        traced = run_graph(build_graph(happy(), FakeRetriever()), SUBJECT, BODY)
    untraced = run_graph(build_graph(happy(), FakeRetriever()), SUBJECT, BODY)

    assert traced["proposal"] == untraced["proposal"]
    assert traced["replies"] == untraced["replies"]
