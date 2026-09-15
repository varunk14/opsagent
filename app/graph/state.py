"""
The state carried between steps, and the one port to the outside world.

Replies accumulate, so every model call is costed. Retrieval is reached only
through the Retriever protocol, which keeps the graph free of storage code.
Observations are what this run's earlier tool calls returned, read back from
the runs table by the driver; the graph never fetches them itself.
"""

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypedDict

from app.contracts import Classification, ExtractedRefund, ProposedAction
from app.llm import Reply


@dataclass(frozen=True)
class PolicyPassage:
    """One retrieved policy passage, with where it came from and how near it was."""

    document: str
    chunk_index: int
    text: str
    distance: float

    @property
    def source(self) -> str:
        return f"{self.document}#{self.chunk_index}"


class Retriever(Protocol):
    """Finds the policy passages nearest in meaning to a question."""

    def search(self, question: str) -> list[PolicyPassage]: ...


class AgentState(TypedDict, total=False):
    subject: str | None
    body: str
    classification: Classification
    extraction: ExtractedRefund | None
    policy: list[str]
    policy_sources: list[str]
    observations: list[dict[str, Any]]
    proposal: ProposedAction
    failure: str
    replies: Annotated[list[Reply], operator.add]
