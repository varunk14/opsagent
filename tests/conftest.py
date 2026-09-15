"""
Shared fixtures for the tests that need a real database.

There is deliberately no in-memory stand-in here. The things this project has to
get right are behaviours of Postgres, not of Python: a UNIQUE constraint that
actually bites, money that stays exact, and later `FOR UPDATE SKIP LOCKED`. A
mock of those passes happily while the real system loses money.

So these tests talk to a live Postgres, and if it is not running the suite FAILS
rather than skips. A skipped test that reads as green is precisely the failure
mode this whole project exists to demonstrate.

    docker compose up -d db
    .venv/bin/python -m pytest

To run only the tests that need no database:

    .venv/bin/python -m pytest -m "not db" --no-cov
"""

import os
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from app.db import apply_migrations

# The local development stack from docker-compose.yml. Overridable so another
# environment can point elsewhere; the password here is a throwaway local one
# and the real DSN is never committed.
TEST_DSN = os.environ.get(
    "OPSAGENT_TEST_DATABASE_URL",
    "postgresql://opsagent:dev@localhost:5432/opsagent_test",
)


@pytest.fixture(scope="session")
def migrated_database() -> str:
    """Bring opsagent_test up to the current schema, once for the whole run."""
    try:
        connection = psycopg.connect(TEST_DSN, connect_timeout=5)
    except psycopg.OperationalError as exc:  # pragma: no cover - environment failure
        pytest.exit(
            "\n\nCannot reach the test database.\n"
            f"  tried: {TEST_DSN}\n"
            f"  error: {exc}\n\n"
            "Start it with `docker compose up -d db`, or run the tests that do\n"
            'not need it with `pytest -m "not db" --no-cov`.\n',
            returncode=1,
        )

    with connection:
        apply_migrations(connection)
    connection.close()
    return TEST_DSN


@pytest.fixture
def db(migrated_database: str):
    """
    A connection whose work is always rolled back.

    Tests therefore never see each other's rows, and the test database does not
    accumulate junk between runs.
    """
    connection = psycopg.connect(migrated_database)
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


@pytest.fixture(scope="session")
def tracing():
    """
    The one tracing set-up a test process gets, exporting straight into memory.

    OpenTelemetry allows one global provider per process, so it is installed once
    here and every test that traces shares it; `exported` empties it per test.
    """
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from app.tracing import Tracing

    installed = Tracing(InMemorySpanExporter(), immediate=True).install()
    yield installed
    installed.shutdown()


@pytest.fixture
def exported(tracing):
    """The in-memory exporter, empty at the start of every test, with an empty span buffer."""
    tracing.exporter.clear()
    tracing.recorder.clear()
    return tracing.exporter


@pytest.fixture
def fixture_inbox() -> Path:
    """The sample inbox committed to the repository."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "inbox.jsonl"


def dsn_for(database: str) -> str:
    return make_conninfo(**{**conninfo_to_dict(TEST_DSN), "dbname": database})


@pytest.fixture
def empty_database():
    """
    A database that has never been migrated, dropped again afterwards.

    Created through an autocommit connection to the maintenance database, since
    CREATE DATABASE cannot run inside a transaction. Anything that needs to
    commit belongs here rather than in the shared test database, where committed
    rows would outlive the test that wrote them.
    """
    name = f"opsagent_scratch_{uuid.uuid4().hex[:12]}"
    admin = psycopg.connect(dsn_for("postgres"), autocommit=True)
    admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield dsn_for(name)
    finally:
        admin.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )
        admin.close()


@pytest.fixture
def fresh_database(empty_database: str) -> str:
    """An empty database with the schema already applied."""
    with psycopg.connect(empty_database) as connection:
        apply_migrations(connection)
    return empty_database
