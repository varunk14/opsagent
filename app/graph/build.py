"""
Wiring the steps into a LangGraph graph.

LangGraph orchestrates here and nothing more. The graph is compiled without a
checkpointer, because the runs table is the one record of a run; a second store
of state would eventually disagree with it.
"""

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph import nodes
from app.graph.state import AgentState, Retriever
from app.llm import Model, ModelUnavailable

AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


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

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", stop_if_proposed("extract"), ["extract", END])
    graph.add_conditional_edges("extract", stop_if_proposed("retrieve"), ["retrieve", END])
    graph.add_conditional_edges("retrieve", stop_if_proposed("plan"), ["plan", END])
    graph.add_edge("plan", END)

    return graph.compile()


def run_graph(graph: AgentGraph, subject: str | None, body: str) -> AgentState:
    """
    Walk one message through the graph and return everything it found.

    Streams the state after each step rather than calling invoke(), so that if
    the model goes down partway through, the replies from the steps that did
    finish are still known -- and attached to the outage -- instead of vanishing
    with the half-built state.
    """
    latest: dict[str, Any] = {}
    try:
        for latest in graph.stream({"subject": subject, "body": body, "replies": []}, stream_mode="values"):
            pass
    except ModelUnavailable as outage:
        outage.replies = list(latest.get("replies", [])) + outage.replies
        raise
    return cast(AgentState, latest)
