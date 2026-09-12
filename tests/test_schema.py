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


def test_a_fractional_paisa_is_refused(db):
    """
    Rupees as floats lose money; paise as integers do not. The column type is
    what enforces that, so prove it rejects the fraction rather than rounding it.
    """
    db.execute(
        "INSERT INTO customers (email, name) VALUES ('priya@example.com', 'Priya')"
    )

    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        db.execute(
            """
            INSERT INTO orders (id, customer_email, amount_paise, status)
            VALUES ('4821', 'priya@example.com', 360000.5, 'paid')
            """
        )


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
