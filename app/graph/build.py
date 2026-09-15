"""
Wiring the steps into a LangGraph graph.

LangGraph orchestrates here and nothing more. The graph is compiled without a
checkpointer, because the runs table is the one record of a run; a second store
of state would eventually disagree with it.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import AgentState
from app.llm import Model


def stop_if_proposed(next_step: str) -> Any:
    """A step that already escalated has decided; later steps must not plan over it."""

    def route(state: AgentState) -> str:
        return END if "proposal" in state else next_step

    return route


def build_graph(model: Model) -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("classify", lambda state: nodes.classify(state, model))
    graph.add_node("extract", lambda state: nodes.extract(state, model))
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("plan", lambda state: nodes.plan(state, model))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", stop_if_proposed("extract"), ["extract", END])
    graph.add_conditional_edges("extract", stop_if_proposed("retrieve"), ["retrieve", END])
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", END)

    return graph.compile()


def run_graph(graph: Any, subject: str | None, body: str) -> AgentState:
    """Walk one message through the graph and return everything it found."""
    result: AgentState = graph.invoke({"subject": subject, "body": body, "replies": []})
    return result
