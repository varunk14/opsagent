"""
The `voice_replies` table: one wav per run, keyed on the run.

A run does not owe a second synthesised reply, and the PK is what makes that a fact rather than a
convention. The migration itself is idempotent: applying it twice against a fresh database changes
no row and adds no constraint the second time.
"""

from uuid import uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

from app.db import apply_migrations

pytestmark = pytest.mark.db


def test_two_replies_for_the_same_run_are_refused(fresh_database):
    run_id = uuid4()
    with psycopg.connect(fresh_database) as connection:
        connection.execute(
            "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
            "VALUES (%s, 'voice', 'done', 'act', %s, %s)",
            (run_id, Jsonb({"untrusted": {"body": "hi"}}), "voice_msg_dupe"),
        )
        connection.execute(
            "INSERT INTO voice_replies (run_id, audio) VALUES (%s, %s)", (run_id, b"first")
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(
                "INSERT INTO voice_replies (run_id, audio) VALUES (%s, %s)", (run_id, b"second")
            )


def test_the_migration_is_idempotent(empty_database):
    """A fresh box migrated twice must arrive at the same schema, unchanged."""
    with psycopg.connect(empty_database) as connection:
        apply_migrations(connection)
        apply_migrations(connection)
        row = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'voice_replies'"
        ).fetchone()
    assert row is not None
    assert row[0] == 1
