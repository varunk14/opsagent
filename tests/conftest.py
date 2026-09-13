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
from pathlib import Path

import psycopg
import pytest

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


@pytest.fixture
def fixture_inbox() -> Path:
    """The sample inbox committed to the repository."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "inbox.jsonl"
