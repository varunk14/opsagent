"""The state carried between steps. Replies accumulate, so every model call is costed."""

import operator
from typing import Annotated, TypedDict

from app.contracts import Classification, ExtractedRefund, ProposedAction
from app.llm import Reply


class AgentState(TypedDict, total=False):
    subject: str | None
    body: str
    classification: Classification
    extraction: ExtractedRefund | None
    policy: list[str]
    proposal: ProposedAction
    failure: str
    replies: Annotated[list[Reply], operator.add]
