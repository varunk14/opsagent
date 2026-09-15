"""
The spans table: what a run's trace is made of, and the shape it must keep.

A trace is evidence, so the table is append-only -- no UPDATE, DELETE or TRUNCATE --
and it insists on its own invariants: ids are hex of the right length, a span ends
no earlier than it starts, a kind is one Langfuse knows, and a model call is always
costed.

Several checks run against one connection, so each refused insert is wrapped in a
savepoint: without one the first refusal would abort the transaction and every
later statement would fail for that reason instead of its own.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

pytestmark = pytest.mark.db

STARTED = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
ENDED = STARTED + timedelta(milliseconds=412)
KINDS = ("span", "chain", "generation", "retriever", "embedding", "tool", "guardrail")
COSTED = {"model": "llama3.1:8b", "input_tokens": 310, "output_tokens": 40, "cost_usd": "0.0000705"}


def a_run(connection) -> str:
    run_id = uuid4()
    connection.execute(
        "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
        "VALUES (%s, 'email', 'running', 'classify', %s, %s)",
        (run_id, Jsonb({}), f"email_msg_{run_id.hex}"),
    )
    return str(run_id)


def insert(connection, run_id: str, **overrides) -> None:
    row = {
        "span_id": "1a2b3c4d5e6f7081",
        "trace_id": run_id,
        "parent_span_id": None,
        "name": "tick",
        "kind": "span",
        "started_at": STARTED,
        "ended_at": ENDED,
        "status": "ok",
        "status_message": None,
        "model": None,
        "prompt_version": None,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "attributes": Jsonb({}),
    }
    row.update(overrides)
    columns = ", ".join(row)
    placeholders = ", ".join(f"%({column})s" for column in row)
    connection.execute(f"INSERT INTO spans ({columns}) VALUES ({placeholders})", row)


def refused(connection, run_id: str, constraint: str, **overrides) -> None:
    with pytest.raises(psycopg.errors.CheckViolation, match=constraint), connection.transaction():
        insert(connection, run_id, **overrides)


def test_a_plain_span_is_accepted(db):
    run_id = a_run(db)

    insert(db, run_id)

    assert db.execute("SELECT count(*) FROM spans WHERE trace_id = %s", (run_id,)).fetchone()[0] == 1


def test_a_span_belongs_to_a_run_that_exists(db):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        insert(db, str(uuid4()))


def test_ids_are_sixteen_hex_characters(db):
    run_id = a_run(db)

    refused(db, run_id, "spans_span_id_is_hex", span_id="not-hex-at-all!!")
    refused(db, run_id, "spans_span_id_is_hex", span_id="1a2b3c4d5e6f70")
    refused(db, run_id, "spans_span_id_is_hex", span_id="1A2B3C4D5E6F7081")
    refused(db, run_id, "spans_parent_is_hex", parent_span_id="XYZ")


def test_a_span_ends_no_earlier_than_it_starts(db):
    run_id = a_run(db)

    refused(db, run_id, "spans_ends_after_start", ended_at=STARTED - timedelta(microseconds=1))
    insert(db, run_id, ended_at=STARTED)  # zero duration is allowed


def test_only_known_kinds_and_statuses(db):
    run_id = a_run(db)

    refused(db, run_id, "spans_kind", kind="llm")
    refused(db, run_id, "spans_status", status="failed")
    for number, kind in enumerate(KINDS, start=1):
        costs = COSTED if kind == "generation" else {}
        insert(db, run_id, span_id=f"{number:016x}", kind=kind, **costs)


def test_a_generation_is_always_costed(db):
    """No model, no tokens or no cost on a generation would read as a free model call."""
    run_id = a_run(db)

    insert(db, run_id, kind="generation", **COSTED)
    for missing in COSTED:
        refused(
            db,
            run_id,
            "spans_generation_is_costed",
            span_id="2a2b3c4d5e6f7081",
            kind="generation",
            **{**COSTED, missing: None},
        )


def test_counts_and_costs_are_never_negative(db):
    run_id = a_run(db)

    refused(db, run_id, "spans_input_tokens", input_tokens=-1)
    refused(db, run_id, "spans_output_tokens", output_tokens=-1)
    refused(db, run_id, "spans_cost_usd", cost_usd="-0.000001")


def test_a_prompt_version_is_twelve_hex_characters(db):
    run_id = a_run(db)

    refused(db, run_id, "spans_prompt_version_is_hex", prompt_version="v3")
    insert(db, run_id, prompt_version="4c0e5dd7b3a9")


def test_a_name_is_never_blank(db):
    refused(db, a_run(db), "spans_name", name=" ")


def test_a_span_cannot_be_changed_once_written(db):
    run_id = a_run(db)
    insert(db, run_id)

    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        db.execute("UPDATE spans SET name = 'edited' WHERE trace_id = %s", (run_id,))


def test_a_span_cannot_be_deleted(db):
    run_id = a_run(db)
    insert(db, run_id)

    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        db.execute("DELETE FROM spans WHERE trace_id = %s", (run_id,))


def test_the_table_cannot_be_truncated(db):
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        db.execute("TRUNCATE spans")


def test_a_trace_is_read_by_run_and_time(db):
    """The run page reads one trace in time order; that read must not scan every span ever written."""
    indexes = [
        definition
        for (definition,) in db.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'spans'").fetchall()
    ]

    assert any("(trace_id, started_at" in definition for definition in indexes)
