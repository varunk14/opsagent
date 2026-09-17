"""
The replay link, and the one constraint that has to bite.

A replay is a fresh run that carries an old run's message, and `replay_of` is the thread back to
the original. The database must refuse a link to a run that does not exist -- a replay pointing at
nothing is a diff with only one side. An ordinary run has no origin, so its `replay_of` is null.
"""

import uuid

import psycopg
import pytest

pytestmark = pytest.mark.db


def a_run(db, *, key: str, replay_of: uuid.UUID | None = None) -> uuid.UUID:
    run_id = uuid.uuid4()
    db.execute(
        """
        INSERT INTO runs (id, channel, status, current_node, state, idempotency_key, replay_of)
        VALUES (%s, 'email', 'queued', 'intake', '{}'::jsonb, %s, %s)
        """,
        (run_id, key, replay_of),
    )
    return run_id


def test_an_ordinary_run_has_no_origin(fresh_database):
    with psycopg.connect(fresh_database) as db:
        run_id = a_run(db, key="email_msg_<one@example.com>")
        origin = db.execute("SELECT replay_of FROM runs WHERE id = %s", (run_id,)).fetchone()[0]
    assert origin is None


def test_a_replay_can_point_at_a_real_run(fresh_database):
    with psycopg.connect(fresh_database) as db:
        original = a_run(db, key="email_msg_<one@example.com>")
        replay = a_run(db, key=f"replay:{uuid.uuid4()}", replay_of=original)
        origin = db.execute("SELECT replay_of FROM runs WHERE id = %s", (replay,)).fetchone()[0]
    assert origin == original


def test_a_replay_of_no_run_is_refused(fresh_database):
    with psycopg.connect(fresh_database) as db, pytest.raises(psycopg.errors.ForeignKeyViolation):
        a_run(db, key=f"replay:{uuid.uuid4()}", replay_of=uuid.uuid4())
