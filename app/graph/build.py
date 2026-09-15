"""
Wiring the steps into a LangGraph graph.

LangGraph orchestrates here and nothing more. The graph is compiled without a
checkpointer, because the runs table is the one record of a run; a second store
of state would eventually disagree with it.

A run picked back up -- after a tool call, or after its worker died -- starts
from what the runs table already holds. When classification, extraction and
policy are all on record it goes straight to planning, so those steps are not
paid for twice. Anything less than the full record is found again from the start.
"""

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph import nodes
from app.graph.state import AgentState, Retriever
from app.llm import Model, ServiceUnavailable

AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]

# Every earlier step's result must be on record before planning can resume.
# extraction counts even when it is None: a status question has nothing to extract.
RESUMABLE = ("classification", "extraction", "policy")


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

    graph.add_node("classify", lambda state: nodes.classify(state, model))
    graph.add_node("extract", lambda state: nodes.extract(state, model))
    graph.add_node("retrieve", lambda state: nodes.retrieve(state, retriever))
    graph.add_node("plan", lambda state: nodes.plan(state, model))

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
