"""
Does the migration runner actually put the schema on disk?

The rest of the suite runs against a database that a previous session already
migrated, so it proves the schema is THERE without ever proving the runner is
what put it there. That gap hid a real bug: the runner executed every statement,
returned a list of what it had applied, and then lost the lot when the
connection closed, because nothing ever committed. No exception, nothing logged,
and a return value that read like success.

So these tests build a brand-new empty database each time, migrate it, and check
from a SEPARATE connection that the tables survived.
"""

import psycopg
import pytest

from app.db import apply_migrations

pytestmark = pytest.mark.db


def test_migrating_an_empty_database_applies_every_migration(empty_database):
    with psycopg.connect(empty_database) as connection:
        applied = apply_migrations(connection)

    assert applied == ["001_schema.sql"]


def test_the_schema_survives_the_connection_that_created_it(empty_database):
    """
    The bug this file was written for.

    psycopg rolls back an open transaction when a connection closes, and says
    nothing at all. A runner that does not commit therefore reports success and
    leaves an empty database behind -- and every start-up after would "migrate"
    again, forever, on a schema that never appears.
    """
    connection = psycopg.connect(empty_database)
    apply_migrations(connection)
    connection.close()

    with psycopg.connect(empty_database) as verifier:
        tables = {
            row[0]
            for row in verifier.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        }

    assert "runs" in tables, "the runner reported success but committed nothing"


def test_a_second_run_applies_nothing(empty_database):
    with psycopg.connect(empty_database) as connection:
        apply_migrations(connection)

    with psycopg.connect(empty_database) as connection:
        assert apply_migrations(connection) == []


def test_a_migration_that_would_sort_wrongly_is_refused(tmp_path, monkeypatch):
    """
    Files are applied in filename order, which is string order. `10_x.sql` sorts
    before `9_x.sql`, so an unpadded name would run migrations out of sequence
    and the damage would be silent. Refuse the name instead of trusting nobody
    ever drops the padding.
    """
    monkeypatch.setattr("app.db.MIGRATIONS_DIR", tmp_path)
    (tmp_path / "9_add_column.sql").write_text("SELECT 1;")

    with pytest.raises(ValueError, match="9_add_column.sql"):
        from app.db import pending_migrations

        pending_migrations(applied=set())


def test_the_database_url_comes_from_the_environment(monkeypatch):
    """Nothing real should ever run against the committed local default."""
    from app.db import database_url

    monkeypatch.setenv("OPSAGENT_DATABASE_URL", "postgresql://elsewhere/opsagent")

    assert database_url() == "postgresql://elsewhere/opsagent"


def test_the_local_default_is_used_when_nothing_is_configured(monkeypatch):
    from app.db import DEFAULT_DSN, database_url

    monkeypatch.delenv("OPSAGENT_DATABASE_URL", raising=False)

    assert database_url() == DEFAULT_DSN


def test_connect_uses_the_configured_database(monkeypatch, empty_database):
    """`connect()` is the entry point everything else will use; prove it works."""
    from app.db import connect

    monkeypatch.setenv("OPSAGENT_DATABASE_URL", empty_database)

    with connect() as connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)
