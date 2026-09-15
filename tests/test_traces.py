"""
Reading runs and their traces back, for the screen.

list_runs shows the newest runs first with what a person needs to pick one: when it
came in, from whom and about what, where it stands, how many steps it took, what it
cost and which prompts it ran under.

trace_of rebuilds one run's spans as the tree they were recorded as -- each step
under its tick, each model call under its step -- with every span's own cost and the
cost of everything beneath it, and the run's approvals beside them. A span whose
parent is missing is shown at the top, never dropped: a trace with a hole in it is
still evidence.

Neither ever carries locked_by or the run's raw state.
"""

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

from app.traces import (
    RunSummary,
    RunTrace,
    TraceSpan,
    build_tree,
    list_runs,
    trace_of,
    walk,
)
from tests.test_approval_path import decide_on, refund_model, work
from tests.test_run_agent import happy_model, ledger, queue

T0 = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def span(
    span_id: str,
    parent: str | None,
    start_ms: int,
    end_ms: int,
    *,
    name: str | None = None,
    kind: str = "span",
    cost: str | None = None,
) -> TraceSpan:
    return TraceSpan(
        span_id=span_id,
        parent_span_id=parent,
        name=name or span_id,
        kind=kind,
        started_at=T0 + timedelta(milliseconds=start_ms),
        ended_at=T0 + timedelta(milliseconds=end_ms),
        status="ok",
        status_message=None,
        model="llama3.1:8b" if kind == "generation" else None,
        prompt_version=None,
        input_tokens=10 if kind == "generation" else None,
        output_tokens=5 if kind == "generation" else None,
        cost_usd=None if cost is None else Decimal(cost),
        attributes={},
    )


def names(nodes: list[TraceSpan]) -> list[str]:
    return [node.name for node in nodes]


# --- the tree -------------------------------------------------------------------------


def test_children_hang_under_their_parents_in_the_order_they_started():
    spans = [
        span("plan", "tick", 30, 40),
        span("tick", None, 0, 50),
        span("classify", "tick", 1, 20),
        span("classify.generate", "classify", 2, 19, kind="generation", cost="0.0000045"),
    ]

    (tick,) = build_tree(spans)

    assert tick.name == "tick"
    assert names(tick.children) == ["classify", "plan"]
    assert names(tick.children[0].children) == ["classify.generate"]


def test_ticks_are_listed_in_the_order_they_ran():
    (first, second) = build_tree([span("later", None, 100, 120), span("earlier", None, 0, 50)])

    assert (first.name, second.name) == ("earlier", "later")


def test_a_span_whose_parent_is_missing_is_kept_at_the_top():
    roots = build_tree([span("tick", None, 0, 50), span("orphan", "0000000000000bad", 10, 20)])

    assert names(roots) == ["tick", "orphan"]


def test_a_span_naming_itself_as_its_parent_is_kept_at_the_top():
    assert names(build_tree([span("loop", "loop", 0, 5)])) == ["loop"]


def test_spans_whose_parents_loop_are_each_kept_once_at_the_top():
    """No SDK writes a loop, but a tree built from one must still end and lose nothing."""
    roots = build_tree([span("a", "b", 0, 10), span("b", "a", 5, 8), span("c", "a", 6, 7)])

    assert sorted(node.name for node in walk(roots)) == ["a", "b", "c"]
    assert names(roots) == ["a"]
    assert roots[0].total_cost == Decimal(0)


def test_a_step_costs_everything_beneath_it_and_nothing_of_its_own():
    (tick,) = build_tree(
        [
            span("tick", None, 0, 50),
            span("classify", "tick", 1, 20),
            span("classify.generate", "classify", 2, 9, kind="generation", cost="0.0000045"),
            span("classify.generate.retry", "classify", 10, 19, kind="generation", cost="0.0000045"),
            span("act", "tick", 30, 40),
        ]
    )

    classify, act = tick.children
    assert classify.own_cost is None
    assert classify.total_cost == Decimal("0.0000090")
    assert act.total_cost == Decimal(0)
    assert tick.total_cost == Decimal("0.0000090")


def test_a_span_knows_how_long_it_took_in_milliseconds():
    (only,) = build_tree([span("tick", None, 0, 1234)])

    assert only.duration_ms == 1234


def test_walking_a_trace_gives_each_parent_then_its_children_in_order():
    roots = build_tree(
        [
            span("tick", None, 0, 50),
            span("classify", "tick", 1, 20),
            span("classify.generate", "classify", 2, 9, kind="generation", cost="0.0000045"),
            span("plan", "tick", 30, 40),
            span("tick 2", None, 60, 90),
        ]
    )

    assert names(list(walk(roots))) == ["tick", "classify", "classify.generate", "plan", "tick 2"]


def test_a_loop_cut_loose_takes_its_place_among_the_roots_by_when_it_started():
    roots = build_tree([span("a", "b", 0, 10), span("b", "a", 5, 8), span("tick", None, 20, 30)])

    assert names(roots) == ["a", "tick"]


def test_building_the_same_spans_twice_gives_the_same_tree():
    spans = [span("tick", None, 0, 50), span("classify", "tick", 1, 20)]
    build_tree(spans)

    (tick,) = build_tree(spans)

    assert names(tick.children) == ["classify"]


def test_the_calls_total_adds_up_model_calls_only_and_is_rounded_once():
    run = RunSummary(
        id=uuid4(), status="done", received_at=T0, sender=None, subject=None,
        steps=1, cost_usd=Decimal("0.000001"), prompt_version=None, failure_class=None,
    )
    roots = build_tree(
        [
            span("tick", None, 0, 50),
            span("classify.generate", "tick", 1, 9, kind="generation", cost="0.0000004"),
            span("plan.generate", "tick", 10, 19, kind="generation", cost="0.0000004"),
            span("embed_query", "tick", 20, 25, kind="embedding", cost="0.5"),
        ]
    )

    assert RunTrace(run=run, roots=roots, approvals=[]).calls_cost == Decimal("0.000001")


# --- runs, as a person picks one ------------------------------------------------------


def insert_run(dsn: str, *, minute: int, sender: str, subject: str, status: str = "done", steps: int = 0) -> str:
    run_id = uuid4()
    state = {
        "untrusted": {"sender": sender, "subject": subject, "body": "the customer's words"},
        "agent": {"steps": [{"step": number} for number in range(1, steps + 1)]},
    }
    with psycopg.connect(dsn) as connection:
        connection.execute(
            "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key, cost_usd, prompt_version, "
            "failure_class, locked_by, created_at) VALUES (%s, 'email', %s, 'act', %s, %s, %s, %s, %s, %s, %s)",
            (
                run_id,
                status,
                Jsonb(state),
                f"email_msg_{run_id.hex}",
                Decimal("0.000018"),
                "438944a9d8fd",
                "rejected" if status == "done" and steps == 0 else None,
                "worker-on-host-4242",
                T0 + timedelta(minutes=minute),
            ),
        )
    return str(run_id)


@pytest.mark.db
def test_runs_are_listed_newest_first_with_what_a_person_needs(fresh_database):
    older = insert_run(fresh_database, minute=0, sender="priya@example.com", subject="Charged twice", steps=2)
    newer = insert_run(fresh_database, minute=5, sender="arjun@example.com", subject="Where is my parcel?", status="waiting_approval", steps=1)

    with psycopg.connect(fresh_database) as connection:
        runs = list_runs(connection)

    assert [str(run.id) for run in runs] == [newer, older]
    first = runs[0]
    assert (first.status, first.sender, first.subject, first.steps) == (
        "waiting_approval",
        "arjun@example.com",
        "Where is my parcel?",
        1,
    )
    assert (first.cost_usd, first.prompt_version) == (Decimal("0.000018"), "438944a9d8fd")
    assert first.received_at == T0 + timedelta(minutes=5)


@pytest.mark.db
def test_the_list_is_capped(fresh_database):
    for minute in range(3):
        insert_run(fresh_database, minute=minute, sender="a@example.com", subject="s")

    with psycopg.connect(fresh_database) as connection:
        assert len(list_runs(connection, limit=2)) == 2


@pytest.mark.db
def test_the_list_never_holds_more_than_a_hundred_runs_or_fewer_than_one(fresh_database):
    with psycopg.connect(fresh_database) as connection:
        connection.execute(
            "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
            "SELECT gen_random_uuid(), 'email', 'done', 'act', '{}'::jsonb, 'email_msg_' || n "
            "FROM generate_series(1, 101) AS n"
        )

    with psycopg.connect(fresh_database) as connection:
        assert len(list_runs(connection, limit=1000)) == 100
        assert len(list_runs(connection, limit=0)) == 1


def test_nothing_read_back_carries_the_worker_or_the_raw_state():
    for shape in (RunSummary, RunTrace, TraceSpan):
        names_of_fields = {field.name for field in fields(shape)}
        assert "locked_by" not in names_of_fields, shape
        assert "state" not in names_of_fields, shape


@pytest.mark.db
def test_the_worker_is_nowhere_in_what_is_read_back(fresh_database):
    run_id = insert_run(fresh_database, minute=0, sender="priya@example.com", subject="Charged twice", steps=1)

    with psycopg.connect(fresh_database) as connection:
        runs = list_runs(connection)
        trace = trace_of(connection, UUID(run_id))

    assert "worker-on-host" not in repr(runs)
    assert "worker-on-host" not in repr(trace)


# --- one run's trace ---------------------------------------------------------------------


@pytest.mark.db
def test_an_unknown_run_has_no_trace(fresh_database):
    with psycopg.connect(fresh_database) as connection:
        assert trace_of(connection, uuid4()) is None


@pytest.mark.db
def test_a_run_not_worked_yet_has_an_empty_trace(fresh_database):
    run_id = queue(fresh_database)

    with psycopg.connect(fresh_database) as connection:
        trace = trace_of(connection, UUID(run_id))

    assert trace is not None
    assert (trace.run.status, trace.roots, trace.approvals) == ("queued", [], [])


@pytest.mark.db
def test_a_trace_holds_only_its_own_runs_spans_and_approvals(fresh_database, exported):
    ledger(fresh_database)
    worked = queue(fresh_database)
    work(fresh_database, refund_model(720_000))
    other = insert_run(fresh_database, minute=0, sender="arjun@example.com", subject="Where is my parcel?")

    with psycopg.connect(fresh_database) as connection:
        assert trace_of(connection, UUID(worked)).approvals
        trace = trace_of(connection, UUID(other))

    assert (trace.roots, trace.approvals) == ([], [])


@pytest.mark.db
def test_a_worked_run_reads_back_as_its_ticks_and_steps(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, happy_model())

    with psycopg.connect(fresh_database) as connection:
        trace = trace_of(connection, UUID(run_id))

    first, second = trace.roots
    assert names(first.children) == ["classify", "extract", "retrieve", "plan", "act"]
    assert names(second.children) == ["plan", "act"]
    assert names(first.children[0].children) == ["classify.generate"]


@pytest.mark.db
def test_the_costs_in_a_trace_agree_with_the_runs_total(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, happy_model())

    with psycopg.connect(fresh_database) as connection:
        trace = trace_of(connection, UUID(run_id))

    assert trace.calls_cost == trace.run.cost_usd
    assert trace.calls_cost > 0
    assert sum((root.total_cost for root in trace.roots), Decimal(0)).quantize(Decimal("0.000001")) == trace.run.cost_usd


@pytest.mark.db
def test_a_trace_shows_the_runs_approvals_and_who_decided(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(720_000))
    decide_on(fresh_database, run_id, approved=True)

    with psycopg.connect(fresh_database) as connection:
        trace = trace_of(connection, UUID(run_id))

    (approval,) = trace.approvals
    assert (approval.status, approval.decided_by) == ("approved", "asha")
    assert approval.reason.startswith("Rs 7,200 is not under the Rs 5,000 limit")
    assert approval.asked_at <= approval.decided_at
    assert approval.executed_at is None
