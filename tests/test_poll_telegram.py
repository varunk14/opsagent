"""
One pass over a bot, end to end.

The mailbox pass can afford a mistake in the ordering: a message marked read is still sitting in the
mailbox, and a person can go and look. Telegram cannot. Moving the cursor is the one irreversible
act in this system -- the updates below it are gone, for everyone, with no copy anywhere.

So the test that matters here is the same one that mattered there, and it matters more: a pass that
fails must leave the cursor exactly where it found it.
"""

import psycopg
import pytest

from app.poll import poll_telegram
from tests.test_adapters_telegram import FakeBot, an_update

pytestmark = pytest.mark.db


def runs_in(dsn: str) -> int:
    with psycopg.connect(dsn) as connection:
        return connection.execute("SELECT count(*) FROM runs").fetchone()[0]


def letters_in(dsn: str) -> int:
    with psycopg.connect(dsn) as connection:
        return connection.execute(
            "SELECT count(*) FROM dead_letters WHERE kind = 'message'"
        ).fetchone()[0]


def test_every_readable_update_becomes_a_run(fresh_database):
    bot = FakeBot([an_update(update_id=700, message_id=11), an_update(update_id=701, message_id=12)])

    with psycopg.connect(fresh_database) as connection:
        summary = poll_telegram(connection, bot)

    assert summary.accepted == 2
    assert runs_in(fresh_database) == 2


def test_the_cursor_moves_only_after_the_runs_are_committed(fresh_database):
    bot = FakeBot([an_update(update_id=700), an_update(update_id=701, message_id=12)])

    with psycopg.connect(fresh_database) as connection:
        poll_telegram(connection, bot)

    assert bot.asked[-1][0] == 702


def test_a_pass_that_dies_part_way_leaves_the_cursor_alone(fresh_database, monkeypatch):
    """
    The one that matters, and the reason this channel gets its own version of it.

    An update confirmed before its run is committed is not recoverable. There is no unread flag to
    put back and no copy in a folder: Telegram has thrown it away, and the customer is waiting for a
    reply to a message that no longer exists anywhere.
    """
    bot = FakeBot([an_update(update_id=700), an_update(update_id=701, message_id=12)])

    def refuse(*args, **kwargs):
        raise psycopg.OperationalError("the database went away mid-pass")

    monkeypatch.setattr("app.poll.accept", refuse)

    with pytest.raises(psycopg.OperationalError), psycopg.connect(fresh_database) as connection:
        poll_telegram(connection, bot)

    assert [asked for asked in bot.asked if asked[0] is not None] == [], "the cursor must not move"
    assert runs_in(fresh_database) == 0


def test_a_second_pass_over_the_same_updates_adds_nothing(fresh_database):
    """The fake never hides what it confirmed, so this is the worst case: everything re-offered."""
    bot = FakeBot([an_update(update_id=700)])

    with psycopg.connect(fresh_database) as connection:
        poll_telegram(connection, bot)
    with psycopg.connect(fresh_database) as connection:
        second = poll_telegram(connection, bot)

    assert second.accepted == 0
    assert second.duplicates == 1
    assert runs_in(fresh_database) == 1


def test_an_update_that_is_not_a_message_is_counted_and_left_unrecorded(fresh_database):
    """
    Someone adds the bot to a group. That is not a customer, and it is not a failure either.

    A dead letter for every one of these would bury the refusals that need a person under noise that
    does not. Not confirming them would be worse: the cursor never moves past what is never handled,
    so one group event would stall the channel permanently.
    """
    bot = FakeBot([{"update_id": 700, "my_chat_member": {"chat": {"id": 1}}}])

    with psycopg.connect(fresh_database) as connection:
        summary = poll_telegram(connection, bot)

    assert summary.ignored == 1
    assert summary.accepted == 0
    assert runs_in(fresh_database) == 0
    assert letters_in(fresh_database) == 0
    assert bot.asked[-1][0] == 701, "confirmed, or it stalls the channel forever"


def test_an_update_that_cannot_be_read_is_recorded_and_confirmed(fresh_database):
    bot = FakeBot([an_update(update_id=700, text=None)])

    with psycopg.connect(fresh_database) as connection:
        summary = poll_telegram(connection, bot)

    assert summary.refused == 1
    assert letters_in(fresh_database) == 1
    assert bot.asked[-1][0] == 701


def test_one_refusal_does_not_cost_the_message_behind_it(fresh_database):
    bot = FakeBot([an_update(update_id=700, text=None), an_update(update_id=701, message_id=12)])

    with psycopg.connect(fresh_database) as connection:
        summary = poll_telegram(connection, bot)

    assert summary.accepted == 1
    assert summary.refused == 1


def test_a_pass_needs_its_own_transaction(fresh_database):
    bot = FakeBot([an_update()])

    with psycopg.connect(fresh_database) as connection:
        connection.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="own transaction"):
            poll_telegram(connection, bot)
