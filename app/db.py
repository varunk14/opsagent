"""
Connecting to Postgres, and getting the schema into it.

The migration runner is about thirty lines and deliberately so. It applies every
`.sql` file in migrations/ once, in filename order, recording what it has done in
a table. Running it twice is a no-op, which matters because it runs on every
start-up: a runner that raised the second time would turn every restart after
the first into an outage.

`000_bootstrap.sql` is skipped. It creates databases, which cannot happen inside
a normal transaction, so the container runs it once at first start instead.
"""

import os
import re
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# Run by the container at first start, against the default database. It creates
# opsagent_test, and CREATE DATABASE cannot run in the transaction used here.
CONTAINER_BOOTSTRAP = "000_bootstrap.sql"

DEFAULT_DSN = "postgresql://opsagent:dev@localhost:5432/opsagent"

# Files are applied in filename order, which is string order, so `10_x.sql` would
# run before `9_x.sql`. Rather than trust that nobody ever drops the padding,
# refuse the name.
MIGRATION_NAME = re.compile(r"^\d{3}_[a-z0-9_]+\.sql$")


def database_url() -> str:
    """
    Where to connect.

    The default is the throwaway local stack in docker-compose.yml. Anything real
    sets OPSAGENT_DATABASE_URL in the environment; no credential is committed.
    """
    return os.environ.get("OPSAGENT_DATABASE_URL", DEFAULT_DSN)


def connect(dsn: str | None = None) -> psycopg.Connection:
    return psycopg.connect(dsn or database_url())


def pending_migrations(applied: set[str]) -> list[Path]:
    """
    The migration files, in order, that this database has not seen.

    Raises if any filename would sort wrongly. Migrations applied out of sequence
    fail quietly and leave two environments with different schemas, which is
    found much later and by someone else.
    """
    everything = sorted(MIGRATIONS_DIR.glob("*.sql"))
    for path in everything:
        if not MIGRATION_NAME.match(path.name):
            raise ValueError(
                f"{path.name} is not named NNN_lower_snake_case.sql, so it would "
                "not sort into the right position"
            )

    return [
        path
        for path in everything
        if path.name != CONTAINER_BOOTSTRAP and path.name not in applied
    ]


def apply_migrations(connection: psycopg.Connection) -> list[str]:
    """
    Bring `connection`'s database up to date. Returns what it applied, which is
    empty on every run after the first.

    Commits before returning, deliberately. Without that, closing the connection
    rolls the whole thing back -- psycopg does this silently, with no exception
    and nothing logged -- while this function still returns a list of filenames
    that reads exactly like success. The caller is left believing the schema
    exists. Owning the commit here means the function is correct however it is
    called.
    """
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename   text PRIMARY KEY,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    applied = {
        row[0]
        for row in connection.execute("SELECT filename FROM schema_migrations").fetchall()
    }

    freshly_applied = []
    for path in pending_migrations(applied):
        connection.execute(path.read_text())
        connection.execute(
            "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
        )
        freshly_applied.append(path.name)

    connection.commit()
    return freshly_applied
