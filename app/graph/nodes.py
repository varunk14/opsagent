"""
The four steps.

A step whose model answer stays unusable after retries does not guess: it
proposes handing the case to a person, records why, and keeps the cost of the
failed attempts. A model that cannot be reached at all is different -- that is
an outage to retry later, so it propagates.
"""

from decimal import Decimal

from pydantic import BaseModel

from app.contracts import Classification, ExtractedRefund, Intent, ProposedAction
from app.graph.prompts import classify_prompt, extract_prompt, plan_prompt
from app.graph.state import AgentState, Retriever
from app.llm import Model, ModelOutputInvalid, Reply, structured

REFUND_INTENTS = {Intent.DUPLICATE_CHARGE, Intent.REFUND_REQUEST}

def escalation(step: str, reason: str, why: str, replies: list[Reply]) -> AgentState:
    """Hand the case to a person, saying which step stopped and why."""
    return {
        "proposal": ProposedAction(
            tool="escalate_to_human",
            args={"reason": reason},
            confidence=Decimal(0),
            reasoning=f"{step} could not continue",
        ),
        "failure": f"{step}: {why}",
        "replies": replies,
    }


def missing_classification(step: str) -> AgentState:
    """
    The graph always classifies first. A direct caller that skips it gets an
    escalation, the same outcome as any other step that cannot continue, rather
    than a KeyError.
    """
    return escalation(
        step, f"The {step} step was reached without a classification.", "no classification to act on", []
    )


def ask[T: BaseModel](
    step: str, model: Model, prompt: str, schema: type[T]
) -> tuple[T | None, AgentState]:
    """Ask through structured(); on unusable output, return the escalation instead."""
    try:
        answer, replies = structured(model, prompt, schema)
    except ModelOutputInvalid as error:
        return None, escalation(
            step,
            f"The {step} step could not get a usable answer from the model.",
            "model output unusable",
            error.replies,
        )
    return answer, {"replies": replies}


def classify(state: AgentState, model: Model) -> AgentState:
    """Decide what the customer wants. Everything after depends on this."""
    answer, update = ask(
        "classify", model, classify_prompt(state.get("subject"), state["body"]), Classification
    )
    if answer is not None:
        update["classification"] = answer
    return update


def extract(state: AgentState, model: Model) -> AgentState:
    """Pull out refund facts, but only when a refund is actually in question."""
    classification = state.get("classification")
    if classification is None:
        return missing_classification("extract")
    if classification.intent not in REFUND_INTENTS:
        return {"extraction": None}

    answer, update = ask(
        "extract", model, extract_prompt(state.get("subject"), state["body"]), ExtractedRefund
    )
    if answer is not None:
        update["extraction"] = answer
    return update


def retrieve(state: AgentState, retriever: Retriever) -> AgentState:
    """
    Policy passages nearest in meaning to what the customer wrote.

    The customer's own words are the question: "charged twice" should find the
    duplicate-payment policy even though the policy never uses those words.
    The sources are kept, so a proposal can later be traced to the rule it cited.
    """
    classification = state.get("classification")
    if classification is None:
        return missing_classification("retrieve")

    question = f"{state.get('subject') or ''}\n{state['body']}".strip()
    if not question:
        return {"policy": [], "policy_sources": []}

    passages = retriever.search(question)
    return {
        "policy": [passage.text for passage in passages],
        "policy_sources": [passage.source for passage in passages],
    }


def plan(state: AgentState, model: Model) -> AgentState:
    """Propose one tool call from everything the earlier steps found."""
    classification = state.get("classification")
    if classification is None:
        return missing_classification("plan")

    prompt = plan_prompt(
        subject=state.get("subject"),
        body=state["body"],
        classification=classification,
        extraction=state.get("extraction"),
        policy=state.get("policy", []),
    )
    answer, update = ask("plan", model, prompt, ProposedAction)
    if answer is not None:
        update["proposal"] = answer
    return update
