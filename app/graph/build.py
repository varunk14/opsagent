"""
Wiring the steps into a LangGraph graph.

LangGraph orchestrates here and nothing more. The graph is compiled without a
checkpointer, because the runs table is the one record of a run; a second store
of state would eventually disagree with it.

A run picked back up -- after a tool call, or after its worker died -- starts
from what the runs table already holds. When classification, extraction and
policy are all on record it goes straight to planning, so those steps are not
paid for twice. Anything less than the full record is found again from the start.

Every step is a span, and every model call inside it a `generation` span carrying
the model, tokens in and out, the reference cost, the prompt version and the
latency -- retries included, since each was paid for. Cost lives on the calls and
nowhere else, so a reader adding up a trace cannot count a step twice. A step
records what it decided (the intent, the tool it proposes, why it stopped) and never
what the customer wrote. A failure is recorded by its class alone: an exception's
message can quote whatever the model or the customer wrote.
"""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from opentelemetry.trace import Span, StatusCode

from app.baseline import rate_for, token_cost
from app.graph import nodes
from app.graph.prompts import PROMPT_VERSIONS
from app.graph.state import AgentState, Retriever
from app.llm import Model, Reply, ServiceUnavailable
from app.tracing import Attr, tracer

AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]

# Every earlier step's result must be on record before planning can resume.
# extraction counts even when it is None: a status question has nothing to extract.
RESUMABLE = ("classification", "extraction", "policy")

# A model's name is set by whoever runs the worker, never by a customer, but every
# call's span copies it, so it is bounded like everything else that reaches a trace.
MAX_MODEL_NAME_CHARS = 120


@contextmanager
def failure_recorded(name: str, kind: str | None = None) -> Iterator[Span]:
    """A span that, if what it wraps raises, is marked failed with the failure's class and nothing else."""
    with tracer().start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
        if kind is not None:
            span.set_attribute(Attr.TYPE, kind)
        try:
            yield span
        except ServiceUnavailable as outage:
            span.set_status(StatusCode.ERROR, outage.failure_class)
            raise
        except Exception as error:
            span.set_status(StatusCode.ERROR, type(error).__name__)
            raise


def task_of(prompt: str) -> str:
    """The TASK line every prompt opens with, which names the prompt and so its version."""
    return prompt.split("\n", 1)[0].removeprefix("TASK:").strip()


class TracedModel:
    """
    The model, with every call recorded as a generation under the step that made it.

    A call that never answered is marked failed and is not a generation: with no
    tokens it would read as a free model call.
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    def generate(self, prompt: str) -> Reply:
        task = task_of(prompt)
        with failure_recorded(f"{task}.generate") as span:
            reply = self.model.generate(prompt)
            # Taken from the reply, not from what was wrapped. A ladder is one wrapper with more
            # than one model behind it, so only the answer knows which one produced it, and at
            # what rate: pricing every call at the dearer tier would hide the whole saving.
            answered = reply.model[:MAX_MODEL_NAME_CHARS]
            span.set_attributes(
                {
                    Attr.TYPE: "generation",
                    Attr.MODEL: answered,
                    Attr.INPUT_TOKENS: reply.prompt_tokens,
                    Attr.OUTPUT_TOKENS: reply.completion_tokens,
                    Attr.LATENCY_MS: reply.latency_ms,
                }
            )
            # A model nobody has priced is a gap in the accounting, not a reason to fail a
            # customer's run: the call is recorded with its tokens and no cost, so the hole is
            # visible to whoever reads the trace rather than filled with somebody else's rate.
            try:
                cost = token_cost(reply.prompt_tokens, reply.completion_tokens, rate_for(reply.model))
            except KeyError:
                span.set_attribute(Attr.COST_DETAILS, json.dumps({"unpriced": answered}))
            else:
                span.set_attributes({Attr.COST_USD: str(cost), Attr.COST_DETAILS: json.dumps({"total": float(cost)})})
            version = PROMPT_VERSIONS.get(task)
            if version is not None:
                span.set_attributes({Attr.PROMPT_VERSION: version, Attr.PROMPT_VERSION_METADATA: version})
            return reply


def decided(update: AgentState) -> dict[str, Any]:
    """What a step decided, as span attributes. Values the model chose from our own lists, never free text."""
    found: dict[str, Any] = {}
    if "classification" in update:
        found[Attr.INTENT] = update["classification"].intent.value
        found[Attr.CLASSIFICATION_CONFIDENCE] = str(update["classification"].confidence)
    if "extraction" in update:
        extraction = update["extraction"]
        found[Attr.ORDER_ID_FOUND] = extraction is not None and extraction.order_id is not None
        found[Attr.AMOUNT_FOUND] = extraction is not None and extraction.amount_paise is not None
    if "policy_sources" in update:
        found[Attr.PASSAGES] = len(update["policy_sources"])
        found[Attr.SOURCES] = list(update["policy_sources"])
    if "proposal" in update:
        found[Attr.TOOL] = update["proposal"].tool
        found[Attr.PROPOSAL_CONFIDENCE] = str(update["proposal"].confidence)
    if "failure" in update:
        # Built by the steps from fixed phrases ("classify: model output unusable"), not from the reply.
        found[Attr.FAILURE] = update["failure"]
    return found


def traced(step: str, run: Callable[[AgentState], AgentState]) -> Callable[..., AgentState]:
    """A step, run inside a span named for it that records what it decided."""

    def step_in_a_span(state: AgentState) -> AgentState:
        with failure_recorded(step, "chain") as span:
            update = run(state)
            span.set_attributes(decided(update))
            return update

    return step_in_a_span


def start_at(state: AgentState) -> str:
    """Planning, when every earlier step is on record; otherwise the beginning."""
    return "plan" if all(key in state for key in RESUMABLE) else "classify"


def stop_if_proposed(next_step: str) -> Callable[[AgentState], str]:
    """A step that already escalated has decided; later steps must not plan over it."""

    def route(state: AgentState) -> str:
        return END if "proposal" in state else next_step

    return route


def build_graph(model: Model, retriever: Retriever) -> AgentGraph:
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    traced_model = TracedModel(model)

    graph.add_node("classify", traced("classify", lambda state: nodes.classify(state, traced_model)))
    graph.add_node("extract", traced("extract", lambda state: nodes.extract(state, traced_model)))
    graph.add_node("retrieve", traced("retrieve", lambda state: nodes.retrieve(state, retriever)))
    graph.add_node("plan", traced("plan", lambda state: nodes.plan(state, traced_model)))

    graph.add_conditional_edges(START, start_at, ["classify", "plan"])
    graph.add_conditional_edges("classify", stop_if_proposed("extract"), ["extract", END])
    graph.add_conditional_edges("extract", stop_if_proposed("retrieve"), ["retrieve", END])
    graph.add_conditional_edges("retrieve", stop_if_proposed("plan"), ["plan", END])
    graph.add_edge("plan", END)

    return graph.compile()


def run_graph(
    graph: AgentGraph, subject: str | None, body: str, prior: AgentState | None = None
) -> AgentState:
    """
    Walk one message through the graph and return everything it found.

    `prior` is what earlier ticks of this run already found. Its replies are
    never carried in: the result holds only the model calls made by this call,
    so a run is not charged twice for the same reply.

    Streams the state after each step rather than calling invoke(), so that if
    a model or the policy store goes down partway through, the replies from the steps that did
    finish are still known -- and attached to the outage -- instead of vanishing
    with the half-built state.
    """
    start = cast(AgentState, {**(prior or {}), "subject": subject, "body": body, "replies": []})
    latest: dict[str, Any] = {}
    try:
        for latest in graph.stream(start, stream_mode="values"):
            pass
    except ServiceUnavailable as outage:
        outage.replies = list(latest.get("replies", [])) + outage.replies
        raise
    return cast(AgentState, latest)
