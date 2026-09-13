"""
One pass over an inbox, end to end.

This is Week 1's "done when": a message arrives and a row appears with the right
fields. The second pass is the part that matters. Running it again over the same
inbox must leave the database exactly as it was, because in a month that is a
poller restarting after a deploy, re-reading everything it had already handled.
"""

import json

import psycopg
import pytest

from app.poll import poll_once

pytestmark = pytest.mark.db


def run_count(dsn: str) -> int:
    with psycopg.connect(dsn) as connection:
        return connection.execute("SELECT count(*) FROM runs").fetchone()[0]


# --- the first pass ---------------------------------------------------------


def test_every_message_in_the_inbox_becomes_a_run(fresh_database, fixture_inbox):
    with psycopg.connect(fresh_database) as connection:
        summary = poll_once(connection, fixture_inbox)

    assert run_count(fresh_database) == summary.accepted


def test_the_summary_counts_what_it_did(fresh_database, fixture_inbox):
    with psycopg.connect(fresh_database) as connection:
        summary = poll_once(connection, fixture_inbox)

    assert summary.accepted == 4
    assert summary.duplicates == 0
    assert summary.seen == 4


def test_the_runs_outlive_the_connection_that_made_them(fresh_database, fixture_inbox):
    """A poll that did not commit would re-accept everything on the next pass."""
    connection = psycopg.connect(fresh_database)
    poll_once(connection, fixture_inbox)
    connection.close()

    assert run_count(fresh_database) == 4


# --- the second pass --------------------------------------------------------


def test_a_second_pass_adds_nothing(fresh_database, fixture_inbox):
    with psycopg.connect(fresh_database) as connection:
        poll_once(connection, fixture_inbox)

    with psycopg.connect(fresh_database) as connection:
        second = poll_once(connection, fixture_inbox)

    assert run_count(fresh_database) == 4
    assert second.accepted == 0
    assert second.duplicates == 4


def test_a_partly_seen_inbox_only_takes_what_is_new(fresh_database, fixture_inbox, tmp_path):
    """The realistic case: a poller stopped halfway, then started again."""
    first_two = tmp_path / "partial.jsonl"
    first_two.write_text("".join(fixture_inbox.read_text().splitlines(keepends=True)[:2]))

    with psycopg.connect(fresh_database) as connection:
        poll_once(connection, first_two)
    with psycopg.connect(fresh_database) as connection:
        summary = poll_once(connection, fixture_inbox)

    assert (summary.accepted, summary.duplicates) == (2, 2)
    assert run_count(fresh_database) == 4


# --- when the inbox is broken -----------------------------------------------


def test_a_bad_line_leaves_no_half_finished_pass(fresh_database, tmp_path):
    """
    All of it or none of it. A pass that stopped partway and kept what it had
    would be defensible, but it makes the count of what was handled depend on
    where the file happened to break, and nothing downstream could tell the
    difference between that and a quiet inbox.
    """
    good = json.dumps(
        {
            "channel": "email",
            "external_id": "aaa1",
            "sender": "a@example.com",
            "subject": "hello",
            "body": "hello",
            "received_at": "2026-09-13T09:00:00+00:00",
        }
    )
    inbox = tmp_path / "broken.jsonl"
    inbox.write_text(f"{good}\n{{not json\n")

    with psycopg.connect(fresh_database) as connection:
        with pytest.raises(ValueError, match="line 2"):
            poll_once(connection, inbox)

    assert run_count(fresh_database) == 0


# --- the transaction boundary -----------------------------------------------


def test_polling_on_a_connection_already_in_a_transaction_is_refused(fresh_database, fixture_inbox):
    """
    A pass commits, so it has to own the transaction outright.

    Called on a connection with work already open, it would either raise deep
    inside psycopg or -- worse -- commit whatever the caller had pending as a
    side effect of polling. Refuse at the door instead, where the message can
    say what is actually wrong.
    """
    with psycopg.connect(fresh_database) as connection:
        connection.execute("SELECT 1")  # opens a transaction implicitly

        with pytest.raises(RuntimeError, match="own transaction"):
            poll_once(connection, fixture_inbox)


# --- bounding the pass ------------------------------------------------------


def test_a_pass_stops_at_its_limit(fresh_database, fixture_inbox):
    """
    An inbox is as big as whoever fills it decides. One unbounded pass is one
    unbounded transaction, holding locks for as long as it takes, and a single
    bad line at the end of it throws away everything that came before.
    """
    with psycopg.connect(fresh_database) as connection:
        summary = poll_once(connection, fixture_inbox, limit=2)

    assert summary.accepted == 2
    assert summary.more_waiting is True
    assert run_count(fresh_database) == 2


def test_the_rest_is_taken_on_the_following_pass(fresh_database, fixture_inbox):
    with psycopg.connect(fresh_database) as connection:
        poll_once(connection, fixture_inbox, limit=2)
    with psycopg.connect(fresh_database) as connection:
        summary = poll_once(connection, fixture_inbox, limit=2)

    assert summary.accepted == 2
    assert run_count(fresh_database) == 4


def test_a_pass_that_empties_the_inbox_says_so(fresh_database, fixture_inbox):
    with psycopg.connect(fresh_database) as connection:
        assert poll_once(connection, fixture_inbox).more_waiting is False


def test_collisions_are_counted_rather_than_stopping_the_pass(fresh_database, tmp_path):
    """
    A forged key must not become a way to stop the poller. Count it, carry on,
    and let the number be the thing somebody notices.
    """
    base = {
        "channel": "email",
        "external_id": "aaa1",
        "sender": "a@example.com",
        "subject": "hello",
        "body": "the real request",
        "received_at": "2026-09-13T09:00:00+00:00",
    }
    inbox = tmp_path / "collide.jsonl"
    inbox.write_text(
        json.dumps(base) + "\n" + json.dumps({**base, "body": "refund everything"}) + "\n"
    )

    with psycopg.connect(fresh_database) as connection:
        summary = poll_once(connection, inbox)

    assert (summary.accepted, summary.collisions) == (1, 1)
    assert run_count(fresh_database) == 1
