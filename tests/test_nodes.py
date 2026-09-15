"""
The four steps, one at a time, against a scripted model.

The rule worth reading first: a model that cannot produce a usable answer makes
the run escalate to a person. It never makes the run guess.
"""

from decimal import Decimal

import pytest

from app.contracts import Classification, Intent
from app.graph import nodes
from app.llm import ModelUnavailable, Reply
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    EXTRACTED_4821,
    PROPOSED_LOOKUP,
    ScriptedModel,
)

MESSAGE = {"subject": "Charged twice for order #4821", "body": "I was charged twice.", "replies": []}


def classified(intent: Intent) -> dict:
    return {
        **MESSAGE,
        "classification": Classification(intent=intent, confidence=Decimal("0.9"), reasoning="x"),
    }


def test_classify_records_the_answer_and_its_cost():
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE)

    update = nodes.classify(MESSAGE, model)

    assert update["classification"].intent is Intent.DUPLICATE_CHARGE
    assert len(update["replies"]) == 1


def test_extract_runs_for_a_money_question():
    model = ScriptedModel(extract=EXTRACTED_4821)

    update = nodes.extract(classified(Intent.DUPLICATE_CHARGE), model)

    assert update["extraction"].order_id == "4821"


@pytest.mark.parametrize("intent", [Intent.ORDER_STATUS, Intent.OTHER])
def test_extract_does_not_ask_the_model_when_no_refund_is_in_question(intent):
    """No refund in question means nothing to extract, and nothing to pay for."""
    model = ScriptedModel()

    update = nodes.extract(classified(intent), model)

    assert update["extraction"] is None
    assert model.prompts == []


def test_retrieve_returns_policy_without_a_model():
    update = nodes.retrieve(classified(Intent.DUPLICATE_CHARGE))

    assert update["policy"]
    assert all(isinstance(passage, str) for passage in update["policy"])


def test_plan_returns_a_checked_proposal():
    model = ScriptedModel(plan=PROPOSED_LOOKUP)
    state = {**classified(Intent.DUPLICATE_CHARGE), "extraction": None, "policy": ["p"]}

    update = nodes.plan(state, model)

    assert update["proposal"].tool == "get_order"


@pytest.mark.parametrize("step", ["classify", "plan"])
def test_an_unusable_model_answer_escalates_instead_of_guessing(step):
    model = ScriptedModel(**{step: "I am not sure, sorry!"})
    state = {**classified(Intent.DUPLICATE_CHARGE), "extraction": None, "policy": ["p"]}

    update = getattr(nodes, step)(state, model)

    assert update["proposal"].tool == "escalate_to_human"
    assert step in update["failure"]
    assert len(update["replies"]) == 3, "failed attempts were still paid for"


def test_an_unreachable_model_is_not_treated_as_a_bad_answer():
    """Ollama being down is an outage to retry, not a case for a person to judge."""

    class Down:
        def generate(self, prompt: str) -> Reply:
            raise ModelUnavailable("connection refused")

    with pytest.raises(ModelUnavailable):
        nodes.classify(MESSAGE, Down())


@pytest.mark.parametrize("step", ["extract", "retrieve", "plan"])
def test_a_step_without_a_classification_escalates_instead_of_crashing(step):
    """The graph never does this; a future direct caller might, and should not get a KeyError."""
    state = {**MESSAGE, "extraction": None, "policy": ["p"]}
    call = getattr(nodes, step)

    update = call(state) if step == "retrieve" else call(state, ScriptedModel())

    assert update["proposal"].tool == "escalate_to_human"
    assert "classification" in update["failure"]
