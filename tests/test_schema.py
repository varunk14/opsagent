"""
The six tables, and the constraints that have to bite.

A schema test that only checks columns exist is close to worthless -- the column
list is visible in the migration file. What is worth testing is whether the
database will actually REFUSE bad data, because every one of these constraints
is standing in for a failure the agent would otherwise commit silently:

  UNIQUE on runs.idempotency_key   -> polling twice creates two runs
  CHECK on runs.status             -> a typo invents a seventh state nothing handles
  numeric on runs.cost_usd         -> week 9's cost comparison drifts
  bigint on orders.amount_paise    -> money stored as a float
  FK on orders.customer_email      -> an order belonging to nobody

Each test below drives the database into the bad state and asserts it is stopped.
"""

import uuid

import psycopg
import pytest

pytestmark = pytest.mark.db

EXPECTED_TABLES = {
    "runs",
    "customers",
    "orders",
    "approvals",
    "tool_calls",
    "policy_chunks",
}


def insert_run(db, *, key: str, status: str = "queued") -> uuid.UUID:
    run_id = uuid.uuid4()
    db.execute(
        """
        INSERT INTO runs (id, channel, status, current_node, state, idempotency_key)
        VALUES (%s, 'email', %s, 'intake', '{}'::jsonb, %s)
        """,
        (run_id, status, key),
    )
    return run_id


def column_type(db, table: str, column: str) -> str:
    """The type as Postgres itself renders it, e.g. 'numeric(10,6)'."""
    row = db.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        WHERE a.attrelid = %s::regclass AND a.attname = %s AND NOT a.attisdropped
        """,
        (table, column),
    ).fetchone()
    assert row is not None, f"{table}.{column} does not exist"
    return row[0]


# --- the tables are there ---------------------------------------------------


def test_all_six_tables_exist(db):
    rows = db.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    ).fetchall()

    assert EXPECTED_TABLES <= {row[0] for row in rows}


def test_pgvector_is_installed(db):
    """Without the extension, policy_chunks.embedding cannot exist at all."""
    row = db.execute(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()

    assert row is not None


def test_applying_the_migrations_again_changes_nothing(db):
    """
    Migrations run on every start-up. If a second run raised, every restart after
    the first would fail, which is the sort of thing that is discovered at 3am.
    """
    from app.db import apply_migrations

    already_applied = db.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]

    apply_migrations(db)

    assert db.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == already_applied


# --- the constraints bite ---------------------------------------------------


def test_the_same_idempotency_key_cannot_be_inserted_twice(db):
    """
    The whole point of the runs table. Two pollers racing on the same message
    must produce one row, and the database is the only thing that can promise
    that -- a read-then-write check in Python loses the race.
    """
    insert_run(db, key="email_msg_9f2a")

    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_run(db, key="email_msg_9f2a")


def test_two_different_messages_both_get_a_row(db):
    """Guards against over-correcting the test above into deduplicating everything."""
    insert_run(db, key="email_msg_9f2a")
    insert_run(db, key="email_msg_7c1b")

    count = db.execute(
        "SELECT count(*) FROM runs WHERE idempotency_key LIKE 'email_msg_%'"
    ).fetchone()[0]
    assert count == 2


def test_a_status_outside_the_state_machine_is_refused(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_run(db, key="email_msg_bad", status="mostly_done")


@pytest.mark.parametrize(
    "status", ["queued", "running", "waiting_approval", "done", "failed", "dead"]
)
def test_every_real_status_is_accepted(db, status):
    """The CHECK must not be so tight that it rejects states the agent needs."""
    insert_run(db, key=f"email_msg_{status}", status=status)


def test_an_order_cannot_belong_to_a_customer_who_does_not_exist(db):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        db.execute(
            """
            INSERT INTO orders (id, customer_email, amount_paise, status)
            VALUES ('4821', 'ghost@example.com', 360000, 'paid')
            """
        )


def test_a_tool_call_key_can_only_be_recorded_once(db):
    """The table exists to replay a repeated call, so the key must be the identity."""
    run_id = insert_run(db, key="email_msg_tools")
    db.execute(
        "INSERT INTO tool_calls (idempotency_key, run_id, tool, args) "
        "VALUES (%s, %s, 'issue_refund', '{}'::jsonb)",
        (f"{run_id}:step_5:issue_refund", run_id),
    )

    with pytest.raises(psycopg.errors.UniqueViolation):
        db.execute(
            "INSERT INTO tool_calls (idempotency_key, run_id, tool, args) "
            "VALUES (%s, %s, 'issue_refund', '{}'::jsonb)",
            (f"{run_id}:step_5:issue_refund", run_id),
        )


# --- money and cost are stored exactly --------------------------------------


def test_cost_is_numeric_not_floating_point(db):
    assert column_type(db, "runs", "cost_usd") == "numeric(10,6)"


def test_an_order_amount_is_a_whole_number_of_paise(db):
    assert column_type(db, "orders", "amount_paise") == "bigint"


def test_a_fractional_paisa_is_silently_rounded_not_refused(db):
    """
    A finding, recorded rather than papered over.

    The obvious assumption is that a bigint column rejects 360000.5. It does not.
    Postgres casts it, rounds it, and stores 360001 -- half a paisa conjured out
    of nothing, no error raised, nothing in any log. Exactly the shape of failure
    this project exists to catch.

    The column type is therefore NOT the defence. The defence has to be the typed
    boundary in app/contracts.py, which refuses a float before it can get here,
    the same way RunRecord already refuses a float cost. This test exists so that
    if anyone later assumes the database is guarding this, the assumption is
    contradicted in writing.
    """
    db.execute(
        "INSERT INTO customers (email, name) VALUES ('priya@example.com', 'Priya')"
    )

    db.execute(
        """
        INSERT INTO orders (id, customer_email, amount_paise, status)
        VALUES ('4821', 'priya@example.com', 360000.5, 'paid')
        """
    )

    stored = db.execute("SELECT amount_paise FROM orders WHERE id = '4821'").fetchone()[0]
    assert stored == 360001, "Postgres rounded rather than refused; guard at the boundary"


# --- the shape retrieval and the worker depend on --------------------------


def test_embeddings_are_768_dimensional(db):
    """Must match the local embedding model; a mismatch fails only at query time."""
    assert column_type(db, "policy_chunks", "embedding") == "vector(768)"


def test_the_worker_can_find_due_runs_without_a_sequential_scan(db):
    """
    The worker polls `status = 'queued' AND next_retry_at <= now()` on every tick.
    Unindexed, that is a full scan of every run ever recorded.
    """
    indexed = db.execute(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'runs'"
    ).fetchall()

    assert any(
        "status" in definition and "next_retry_at" in definition
        for (definition,) in indexed
    )


def test_the_worker_can_find_expired_locks_without_a_sequential_scan(db):
    """Lock expiry looks for running runs by how old their lock is, on every claim."""
    indexed = db.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'runs'").fetchall()

    assert any("locked_at" in definition and "running" in definition for (definition,) in indexed)


# --- week 4: dead letters -----------------------------------------------------


def insert_dead_letter(db, *, kind: str, run_id: uuid.UUID | None = None, key: str = "email_msg_dead") -> None:
    db.execute(
        "INSERT INTO dead_letters (kind, run_id, idempotency_key, payload, reason) "
        "VALUES (%s, %s, %s, '{}'::jsonb, 'test')",
        (kind, run_id, key),
    )


def test_a_dead_letter_is_about_a_run_or_a_message_and_nothing_else(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_dead_letter(db, kind="mystery")


def test_a_dead_run_letter_names_its_run(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_dead_letter(db, kind="run", run_id=None)


def test_a_quarantined_message_names_no_run(db):
    """It never became a run of its own: that is why it is quarantined."""
    run_id = insert_run(db, key="email_msg_real")

    with pytest.raises(psycopg.errors.CheckViolation):
        insert_dead_letter(db, kind="message", run_id=run_id)


def test_a_dead_letter_cannot_name_a_run_that_does_not_exist(db):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        insert_dead_letter(db, kind="run", run_id=uuid.uuid4())


def test_a_dead_run_has_at_most_one_open_dead_letter(db):
    run_id = insert_run(db, key="email_msg_dead_once", status="dead")
    insert_dead_letter(db, kind="run", run_id=run_id)

    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_dead_letter(db, kind="run", run_id=run_id)


def test_a_requeued_run_that_dies_again_gets_a_new_dead_letter(db):
    run_id = insert_run(db, key="email_msg_dies_again", status="dead")
    insert_dead_letter(db, kind="run", run_id=run_id)
    db.execute("UPDATE dead_letters SET requeued_at = now() WHERE run_id = %s", (run_id,))

    insert_dead_letter(db, kind="run", run_id=run_id)

    assert db.execute("SELECT count(*) FROM dead_letters WHERE run_id = %s", (run_id,)).fetchone()[0] == 2


# --- week 5: guardrails --------------------------------------------------------


def test_the_guardrail_starts_at_the_handbook_defaults(db):
    """Rs 5,000 and 0.85: a fresh database is safe before anyone configures it."""
    limit, confidence = db.execute(
        "SELECT auto_refund_limit_paise, min_confidence FROM guardrails"
    ).fetchone()

    assert (limit, str(confidence)) == (500_000, "0.85")


def test_there_is_only_ever_one_guardrail_row(db):
    """Two rows would make "the limit" ambiguous, and whichever was read first would win."""
    with pytest.raises(psycopg.errors.UniqueViolation):
        db.execute(
            "INSERT INTO guardrails (auto_refund_limit_paise, min_confidence, updated_by) "
            "VALUES (100, 0.5, 'test')"
        )


def test_a_second_guardrail_row_cannot_dodge_the_key(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "INSERT INTO guardrails (singleton, auto_refund_limit_paise, min_confidence, updated_by) "
            "VALUES (false, 100, 0.5, 'test')"
        )


def test_the_guardrail_row_cannot_be_deleted(db):
    """With no row there is no limit; the database refuses rather than leaving it to Python."""
    with pytest.raises(psycopg.errors.RaiseException):
        db.execute("DELETE FROM guardrails")


@pytest.mark.parametrize(
    "assignment",
    [
        "auto_refund_limit_paise = -1",
        "min_confidence = 1.5",
        "min_confidence = -0.1",
        "updated_by = '   '",
    ],
)
def test_a_guardrail_outside_its_range_is_refused(db, assignment):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(f"UPDATE guardrails SET {assignment}")  # noqa: S608 - fixed test strings


def test_a_limit_of_zero_is_allowed(db):
    """Zero is the kill switch: every refund needs a person."""
    db.execute("UPDATE guardrails SET auto_refund_limit_paise = 0")


# --- week 5: approvals --------------------------------------------------------


def insert_approval(db, run_id: uuid.UUID, **columns) -> None:
    values = {"action": '{"tool": "issue_refund"}', "evidence": "{}", "reason": "test"} | columns
    names = ", ".join(values)
    placeholders = ", ".join(["%s"] * len(values))
    db.execute(
        f"INSERT INTO approvals (run_id, {names}) VALUES (%s, {placeholders})",  # noqa: S608 - test-owned names
        (run_id, *values.values()),
    )


def test_a_run_has_at_most_one_pending_approval(db):
    """Two would let one refund be approved twice, once per row."""
    run_id = insert_run(db, key="email_msg_two_pending", status="waiting_approval")
    insert_approval(db, run_id)

    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_approval(db, run_id)


def test_a_decided_approval_does_not_block_a_new_one(db):
    run_id = insert_run(db, key="email_msg_asked_again", status="waiting_approval")
    insert_approval(db, run_id, status="rejected", decided_by="asha", decided_at="2026-09-15T10:00:00Z")

    insert_approval(db, run_id)


def test_a_run_has_at_most_one_approved_action_waiting_to_execute(db):
    run_id = insert_run(db, key="email_msg_two_approved", status="queued")
    decided = {"status": "approved", "decided_by": "asha", "decided_at": "2026-09-15T10:00:00Z"}
    insert_approval(db, run_id, **decided)

    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_approval(db, run_id, **decided)


def test_an_approval_needs_a_reason(db):
    """A person asked to decide must be told why they are being asked."""
    run_id = insert_run(db, key="email_msg_no_reason", status="waiting_approval")

    with pytest.raises(psycopg.errors.NotNullViolation):
        insert_approval(db, run_id, reason=None)


@pytest.mark.parametrize(
    "columns",
    [
        {"status": "approved"},
        {"status": "approved", "decided_by": "asha"},
        {"status": "rejected", "decided_at": "2026-09-15T10:00:00Z"},
        {"status": "approved", "decided_by": "  ", "decided_at": "2026-09-15T10:00:00Z"},
        {"decided_by": "asha", "decided_at": "2026-09-15T10:00:00Z"},
        {"executed_at": "2026-09-15T10:00:00Z"},
        {"status": "rejected", "decided_by": "asha", "decided_at": "2026-09-15T10:00:00Z",
         "executed_at": "2026-09-15T10:00:00Z"},
    ],
    ids=[
        "approved-by-nobody",
        "approved-at-no-time",
        "rejected-by-nobody",
        "approved-by-a-blank-name",
        "pending-but-decided",
        "pending-but-executed",
        "rejected-but-executed",
    ],
)
def test_a_decision_is_recorded_whole_or_not_at_all(db, columns):
    run_id = insert_run(db, key="email_msg_half_decided", status="waiting_approval")

    with pytest.raises(psycopg.errors.CheckViolation):
        insert_approval(db, run_id, **columns)
