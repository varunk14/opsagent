"""
The four steps.

A step whose model answer stays unusable after retries does not guess: it
proposes handing the case to a person, records why, and keeps the cost of the
failed attempts. A model that cannot be reached at all is different -- that is
an outage to retry later, so it propagates.
"""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from app.contracts import Classification, ExtractedRefund, Intent, ProposedAction
from app.graph.prompts import classify_prompt, extract_prompt, plan_prompt
from app.graph.state import AgentState
from app.llm import Model, ModelOutputInvalid, structured

REFUND_INTENTS = {Intent.DUPLICATE_CHARGE, Intent.REFUND_REQUEST}

# Week 3 replaces this with pgvector retrieval over the real policy documents.
POLICY_STUB: dict[Intent, list[str]] = {
    Intent.DUPLICATE_CHARGE: [
        (
            "[stub] A charge taken twice for one order is refunded in full once the "
            "duplicate is confirmed on the order record."
        ),
    ],
    Intent.REFUND_REQUEST: [
        "[stub] Change-of-mind refunds are allowed within 30 days of delivery.",
    ],
    Intent.ORDER_STATUS: ["[stub] Delivery questions are answered by a person."],
    Intent.OTHER: ["[stub] Anything outside refunds and delivery goes to a person."],
}


def escalation(step: str, error: ModelOutputInvalid) -> dict[str, Any]:
    return {
        "proposal": ProposedAction(
            tool="escalate_to_human",
            args={"reason": f"The {step} step could not get a usable answer from the model."},
            confidence=Decimal(0),
            reasoning=f"{step} produced no usable answer after retries",
        ),
        "failure": f"{step}: model output unusable",
        "replies": error.replies,
    }


def ask(
    step: str, model: Model, prompt: str, schema: type[BaseModel]
) -> tuple[BaseModel | None, dict[str, Any]]:
    """Ask once through structured(); on unusable output, return the escalation instead."""
    try:
        answer, replies = structured(model, prompt, schema)
    except ModelOutputInvalid as error:
        return None, escalation(step, error)
    return answer, {"replies": replies}


def classify(state: AgentState, model: Model) -> dict[str, Any]:
    answer, update = ask(
        "classify", model, classify_prompt(state.get("subject"), state["body"]), Classification
    )
    return update if answer is None else {**update, "classification": answer}


def extract(state: AgentState, model: Model) -> dict[str, Any]:
    if state["classification"].intent not in REFUND_INTENTS:
        return {"extraction": None}
    answer, update = ask(
        "extract", model, extract_prompt(state.get("subject"), state["body"]), ExtractedRefund
    )
    return update if answer is None else {**update, "extraction": answer}


def retrieve(state: AgentState) -> dict[str, Any]:
    return {"policy": POLICY_STUB[state["classification"].intent]}


def plan(state: AgentState, model: Model) -> dict[str, Any]:
    prompt = plan_prompt(
        subject=state.get("subject"),
        body=state["body"],
        classification=state["classification"],
        extraction=state.get("extraction"),
        policy=state.get("policy", []),
    )
    answer, update = ask("plan", model, prompt, ProposedAction)
    return update if answer is None else {**update, "proposal": answer}
