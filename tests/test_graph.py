"""
The whole path, classify -> extract -> retrieve -> plan, proposing only.

LangGraph orchestrates; it does not persist. The run row stays the source of
truth, so the compiled graph must carry no checkpointer of its own.
"""

from pathlib import Path

import pytest

from app.contracts import Intent
from app.graph.build import build_graph, run_graph
from app.llm import ModelUnavailable
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

SUBJECT = "Charged twice for order #4821"
BODY = "Hi, I think I was charged twice for order #4821 last Tuesday."


def test_priyas_email_ends_in_a_proposal_to_look_up_the_order():
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_LOOKUP)

    state = run_graph(build_graph(model, FakeRetriever()), SUBJECT, BODY)

    assert state["classification"].intent is Intent.DUPLICATE_CHARGE
    assert state["extraction"].order_id == "4821"
    assert state["proposal"].tool == "get_order"
    assert model.tasks() == ["classify", "extract", "plan"]


def test_every_model_call_on_the_way_is_counted():
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_LOOKUP)

    state = run_graph(build_graph(model, FakeRetriever()), SUBJECT, BODY)

    assert len(state["replies"]) == 3


def test_a_status_question_skips_extraction():
    model = ScriptedModel(classify=CLASSIFIED_STATUS, plan=PROPOSED_ESCALATE)

    state = run_graph(build_graph(model, FakeRetriever()), "Where is #5102?", "When does it arrive?")

    assert state["extraction"] is None
    assert model.tasks() == ["classify", "plan"]


def test_a_failed_classification_stops_before_planning():
    """Planning on top of a classification that never happened would be a guess."""
    model = ScriptedModel(classify="no idea")

    state = run_graph(build_graph(model, FakeRetriever()), SUBJECT, BODY)

    assert state["proposal"].tool == "escalate_to_human"
    assert model.tasks() == ["classify"] * 3


def test_the_graph_has_the_four_steps():
    graph = build_graph(ScriptedModel(), FakeRetriever())

    assert {"classify", "extract", "retrieve", "plan"} <= set(graph.get_graph().nodes)


def test_the_graph_keeps_no_state_of_its_own():
    """Persistence belongs to the runs table; a checkpointer would become a second truth."""
    assert build_graph(ScriptedModel(), FakeRetriever()).checkpointer is None


def test_the_graph_cannot_touch_the_database():
    """Proposing only: nothing under app/graph may reach the database or run a tool."""
    for source in Path("app/graph").glob("*.py"):
        text = source.read_text()
        assert "psycopg" not in text, source
        assert "app.db" not in text, source
        assert "app.intake" not in text, source


def test_an_outage_partway_through_keeps_the_cost_of_finished_steps():
    """classify answered and was paid for; extract then found the model down."""
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=OUTAGE)

    with pytest.raises(ModelUnavailable) as raised:
        run_graph(build_graph(model, FakeRetriever()), SUBJECT, BODY)

    assert len(raised.value.replies) == 1


def test_the_retrieved_policy_reaches_the_planner():
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_LOOKUP)

    state = run_graph(build_graph(model, FakeRetriever()), SUBJECT, BODY)

    plan_prompt = next(p for p in model.prompts if p.startswith("TASK: plan"))
    assert "The duplicate amount is returned in full." in plan_prompt
    assert state["policy_sources"] == ["duplicate-payments#1"]
