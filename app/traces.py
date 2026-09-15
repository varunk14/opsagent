"""
Reading runs and their traces back, for the screen.

list_runs is the list a person picks a run from, newest first. trace_of rebuilds one
run's trace from the spans table as the tree it was recorded as -- each step under
its tick, each model call under its step -- with every span's own cost and the cost of
everything beneath it, the calls' total beside the run's recorded total, and the
run's approvals.

What a screen may show is decided here, by the columns selected: never the run's raw
state, never locked_by. A span whose parent is missing, or whose parents loop, is
shown at the top rather than dropped -- a trace with a hole in it is still evidence.
Nothing here commits.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg

MICRO_DOLLAR = Decimal("0.000001")  # runs.cost_usd is numeric(10, 6)
MAX_RUNS = 100

_RUN_COLUMNS = """
    SELECT id, status, created_at,
           state -> 'untrusted' ->> 'sender',
           state -> 'untrusted' ->> 'subject',
           coalesce(jsonb_array_length(state -> 'agent' -> 'steps'), 0),
           cost_usd, prompt_version, failure_class
      FROM runs
"""
LIST_RUNS = _RUN_COLUMNS + " ORDER BY created_at DESC, id LIMIT %s"
ONE_RUN = _RUN_COLUMNS + " WHERE id = %s"

SPANS = """
    SELECT span_id, parent_span_id, name, kind, started_at, ended_at, status, status_message,
           model, prompt_version, input_tokens, output_tokens, cost_usd, attributes
      FROM spans
     WHERE trace_id = %s
     ORDER BY started_at, span_id
"""

APPROVALS = """
    SELECT id, status, reason, created_at, decided_by, decided_at, decision_note, executed_at
      FROM approvals
     WHERE run_id = %s
     ORDER BY created_at, id
"""


@dataclass(frozen=True)
class RunSummary:
    """One run as the runs list shows it."""

    id: UUID
    status: str
    received_at: datetime
    sender: str | None
    subject: str | None
    steps: int
    cost_usd: Decimal
    prompt_version: str | None
    failure_class: str | None


@dataclass
class TraceSpan:
    """One span, with the spans recorded beneath it."""

    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    started_at: datetime
    ended_at: datetime
    status: str
    status_message: str | None
    model: str | None
    prompt_version: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: Decimal | None
    attributes: dict[str, Any]
    children: list["TraceSpan"] = field(default_factory=list)

    @property
    def own_cost(self) -> Decimal | None:
        """What this span itself cost: set on model calls only, so a step's is None, not zero."""
        return self.cost_usd

    @property
    def total_cost(self) -> Decimal:
        """This span's cost and everything beneath it, exact: rounding happens once, for the run."""
        return sum((node.cost_usd or Decimal(0) for node in walk([self])), Decimal(0))

    @property
    def duration_ms(self) -> int:
        return (self.ended_at - self.started_at) // timedelta(milliseconds=1)


@dataclass(frozen=True)
class ApprovalEvent:
    """A decision the run waited on: when it was asked, and who decided what."""

    id: int
    status: str
    reason: str
    asked_at: datetime
    decided_by: str | None
    decided_at: datetime | None
    note: str | None
    executed_at: datetime | None


@dataclass(frozen=True)
class RunTrace:
    """One run, its spans as a tree, and its approvals."""

    run: RunSummary
    roots: list[TraceSpan]
    approvals: list[ApprovalEvent]

    @property
    def calls_cost(self) -> Decimal:
        """The run's model calls added up exactly and rounded once, as the run's own cost was."""
        calls = (node.cost_usd or Decimal(0) for node in walk(self.roots) if node.kind == "generation")
        return sum(calls, Decimal(0)).quantize(MICRO_DOLLAR)


def walk(roots: Sequence[TraceSpan]) -> Iterator[TraceSpan]:
    """Every span in the trees under `roots`, parents before their children."""
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def build_tree(spans: Sequence[TraceSpan]) -> list[TraceSpan]:
    """
    The spans as trees, each list of children in the order its spans started.

    A span is placed under its parent when the parent is among them. One that names a
    missing parent, or itself, or sits in a loop of parents, becomes a root: every span
    given comes back exactly once.
    """
    ordered = sorted(spans, key=lambda node: (node.started_at, node.span_id))
    by_id = {node.span_id: node for node in ordered}
    for node in ordered:
        node.children = []
    roots: list[TraceSpan] = []
    for node in ordered:
        parent = by_id.get(node.parent_span_id) if node.parent_span_id not in (None, node.span_id) else None
        (parent.children if parent is not None else roots).append(node)

    reached = {node.span_id for node in walk(roots)}
    for node in ordered:
        if node.span_id not in reached:
            # Only a loop of parents gets here: cut it at its earliest span.
            by_id[str(node.parent_span_id)].children.remove(node)
            roots.append(node)
            reached.update(child.span_id for child in walk([node]))
    roots.sort(key=lambda node: (node.started_at, node.span_id))
    return roots


def list_runs(connection: psycopg.Connection, limit: int = MAX_RUNS) -> list[RunSummary]:
    """The newest runs first, at most `limit` of them and never more than MAX_RUNS."""
    bounded = max(1, min(limit, MAX_RUNS))
    return [RunSummary(*row) for row in connection.execute(LIST_RUNS, (bounded,)).fetchall()]


def span_from(values: Sequence[Any]) -> TraceSpan:
    """One row of SPANS as a span, each column by name."""
    (
        span_id, parent_span_id, name, kind, started_at, ended_at, status, status_message,
        model, prompt_version, input_tokens, output_tokens, cost_usd, attributes,
    ) = values
    return TraceSpan(
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        kind=kind,
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        status_message=status_message,
        model=model,
        prompt_version=prompt_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        attributes=dict(attributes or {}),
    )


def trace_of(connection: psycopg.Connection, run_id: UUID) -> RunTrace | None:
    """One run's trace and approvals, or None when there is no such run."""
    row = connection.execute(ONE_RUN, (run_id,)).fetchone()
    if row is None:
        return None
    spans = [span_from(values) for values in connection.execute(SPANS, (run_id,)).fetchall()]
    approvals = [ApprovalEvent(*values) for values in connection.execute(APPROVALS, (run_id,)).fetchall()]
    return RunTrace(run=RunSummary(*row), roots=build_tree(spans), approvals=approvals)
