"""
One pass over a real mailbox, end to end.

The fixture inbox is a file: stateless, re-read whole on every pass, and forgiving of a poller that
crashes halfway. A mailbox is none of those things. It remembers what we told it, it forgets what we
did not, and everything interesting here is about the order those two happen in.

Two orderings are load-bearing, and both are tested by making the pass fail on purpose:

  - a message is marked read only after the run is committed, so a pass that dies mid-way offers it
    again rather than losing the customer
  - a message the parser refuses is written down before it is marked read, so refusing it is a thing
    someone can see rather than a thing that quietly happened
"""

import psycopg
import pytest

from app.adapters.inbox import PREVIEW_CHARS
from app.adapters.mailbox import MAX_FETCHED
from app.poll import poll_mailbox
from tests.fakes import FakeMailbox, an_email

pytestmark = pytest.mark.db


def runs_in(dsn: str) -> int:
    with psycopg.connect(dsn) as connection:
        return connection.execute("SELECT count(*) FROM runs").fetchone()[0]


def letters_in(dsn: str) -> list[tuple[str, str]]:
    with psycopg.connect(dsn) as connection:
        return connection.execute(
            "SELECT idempotency_key, reason FROM dead_letters WHERE kind = 'message' ORDER BY id"
        ).fetchall()


# --- the ordinary pass ----------------------------------------------------------------------


def test_every_readable_message_becomes_a_run(fresh_database):
    mailbox = FakeMailbox(
        {b"1": an_email(message_id="<one@example.com>"), b"2": an_email(message_id="<two@example.com>")}
    )

    with psycopg.connect(fresh_database) as connection:
        summary = poll_mailbox(connection, mailbox)

    assert summary.accepted == 2
    assert runs_in(fresh_database) == 2


def test_what_was_recorded_is_what_gets_marked_read(fresh_database):
    mailbox = FakeMailbox({b"1": an_email(message_id="<one@example.com>")})

    with psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)

    assert mailbox.seen == [b"1"]


def test_the_runs_outlive_the_connection_that_made_them(fresh_database):
    """A pass that did not commit would hand the same mailbox back on the next one."""
    mailbox = FakeMailbox({b"1": an_email(message_id="<one@example.com>")})
    connection = psycopg.connect(fresh_database)
    poll_mailbox(connection, mailbox)
    connection.close()

    assert runs_in(fresh_database) == 1


def test_the_same_mailbox_polled_twice_produces_one_run(fresh_database):
    """
    The fake never hides what it was told to mark, so this is the worst case: every message
    re-offered on the next pass. Intake is idempotent on the Message-ID, so the second pass is
    lookups and nothing written -- which is exactly what makes marking-after-commit safe.
    """
    mailbox = FakeMailbox({b"1": an_email(message_id="<one@example.com>")})

    with psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)
    with psycopg.connect(fresh_database) as connection:
        second = poll_mailbox(connection, mailbox)

    assert second.accepted == 0
    assert second.duplicates == 1
    assert runs_in(fresh_database) == 1


# --- the orderings, proved by breaking the pass ----------------------------------------------


def test_a_pass_that_fails_marks_nothing_read(fresh_database, monkeypatch):
    """
    The one that matters. If marking came first, this message would be read in the mailbox with no
    run anywhere -- a customer dropped in silence, which is the failure this project is about.
    """
    mailbox = FakeMailbox({b"1": an_email(message_id="<one@example.com>")})

    def refuse(*args, **kwargs):
        raise psycopg.OperationalError("the database went away mid-pass")

    monkeypatch.setattr("app.poll.accept", refuse)

    with pytest.raises(psycopg.OperationalError), psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)

    assert mailbox.seen == [], "nothing may be marked read when the work did not land"
    assert runs_in(fresh_database) == 0


def test_a_pass_that_dies_part_way_marks_nothing_it_had_already_handled(fresh_database):
    """
    The decisive one, and the reason the test above is not enough on its own.

    Here the pass gets a message accepted and *then* falls over, so the rollback throws away work
    that had already been done. Marking as it went would have told the mailbox to forget a message
    whose run no longer exists -- and nothing would ever offer it again. That is the silent drop,
    and it is invisible to a test that fails before anything has been handled.
    """

    class DropsTheConnectionPartWay(FakeMailbox):
        def fetch(self, number: bytes, parts: str):
            if number == b"2":
                raise OSError("the mail server went away mid-pass")
            return super().fetch(number, parts)

    mailbox = DropsTheConnectionPartWay(
        {b"1": an_email(message_id="<one@example.com>"), b"2": an_email(message_id="<two@example.com>")}
    )

    with pytest.raises(OSError), psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)

    assert mailbox.seen == [], "a message whose run was rolled back must still be unread"
    assert runs_in(fresh_database) == 0


# --- what cannot be read --------------------------------------------------------------------


def test_a_refused_message_is_written_down_and_then_marked_read(fresh_database):
    """
    Left unread it would be re-fetched and re-parsed on every poll forever, spending one of the
    pass's slots each time. Written down, it is visible on the screen instead of only in a mailbox
    nobody is watching, and the mailbox moves on.
    """
    mailbox = FakeMailbox({b"1": b"not an email at all"})

    with psycopg.connect(fresh_database) as connection:
        summary = poll_mailbox(connection, mailbox)

    letters = letters_in(fresh_database)
    assert summary.refused == 1
    assert summary.accepted == 0
    assert len(letters) == 1
    assert "Message-ID" in letters[0][1]
    assert mailbox.seen == [b"1"]
    assert runs_in(fresh_database) == 0


def test_the_same_refused_message_is_kept_once(fresh_database):
    """Re-delivery of something unreadable is still one thing for a person to look at."""
    mailbox = FakeMailbox({b"1": b"not an email at all"})

    with psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)
    with psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)

    assert len(letters_in(fresh_database)) == 1


def test_two_different_unreadable_messages_are_both_kept(fresh_database):
    """Keyed on the bytes, so one refusal cannot stand in for another and hide it."""
    mailbox = FakeMailbox({b"1": b"not an email at all", b"2": b"also not an email"})

    with psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)

    assert len(letters_in(fresh_database)) == 2


def test_two_refusals_alike_for_longer_than_the_preview_are_still_two(fresh_database):
    """
    Why the key is a digest of the whole message and not of what we kept.

    The preview stops at PREVIEW_CHARS, so two messages identical up to there and different after
    look the same to anything reading the record. Keyed on that, the second would be silently
    dropped as a duplicate of the first -- a padded prefix is all it would take to hide a message
    behind one already refused. The digest is taken over every byte, so they stay distinct.
    """
    alike = b"Subject: " + b"x" * PREVIEW_CHARS
    mailbox = FakeMailbox({b"1": alike + b"first", b"2": alike + b"second"})

    with psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)

    assert len(letters_in(fresh_database)) == 2


def test_one_refusal_does_not_cost_the_message_behind_it(fresh_database):
    mailbox = FakeMailbox({b"1": b"not an email at all", b"2": an_email(message_id="<good@example.com>")})

    with psycopg.connect(fresh_database) as connection:
        summary = poll_mailbox(connection, mailbox)

    assert summary.accepted == 1
    assert summary.refused == 1
    assert sorted(mailbox.seen) == [b"1", b"2"]


def test_the_refusal_keeps_enough_of_the_message_to_recognise_it(fresh_database):
    """A reason with no message attached tells an operator nothing they can act on."""
    mailbox = FakeMailbox({b"1": b"Subject: something odd\r\n\r\nthe body we could not read"})

    with psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)

    with psycopg.connect(fresh_database) as connection:
        payload = connection.execute(
            "SELECT payload FROM dead_letters WHERE kind = 'message'"
        ).fetchone()[0]

    assert "something odd" in payload["preview"]


def test_a_refusal_is_not_polled_forever(fresh_database):
    """
    The point of writing it down. The second pass sees the same unread message -- the fake never
    hides anything -- and must not write a second letter or spend the slot again.
    """
    mailbox = FakeMailbox({b"1": b"not an email at all"})

    with psycopg.connect(fresh_database) as connection:
        poll_mailbox(connection, mailbox)
    with psycopg.connect(fresh_database) as connection:
        second = poll_mailbox(connection, mailbox)

    assert second.refused == 1
    assert len(letters_in(fresh_database)) == 1


# --- what the summary says is left ------------------------------------------------------------


def test_a_backlog_is_reported_even_when_a_message_would_not_come_back(fresh_database):
    """
    The summary's job is to tell whoever schedules the next pass whether to bother.

    Counting what came back rather than what the server listed gets this wrong in the direction
    that hurts: one message listed and then not delivered drops the tally below the cap, and a
    mailbox with hundreds still queued reports itself drained.
    """

    class WillNotProduceTheFirst(FakeMailbox):
        def fetch(self, number: bytes, parts: str):
            if number == b"0":
                return "NO", [b"gone"]
            return super().fetch(number, parts)

    backlog = WillNotProduceTheFirst(
        {str(n).encode(): an_email(message_id=f"<{n}@example.com>") for n in range(MAX_FETCHED + 5)}
    )

    with psycopg.connect(fresh_database) as connection:
        summary = poll_mailbox(connection, backlog)

    assert summary.accepted < MAX_FETCHED, "one of them was never delivered"
    assert summary.more_waiting is True


def test_a_mailbox_that_is_exactly_full_is_not_reported_as_having_more(fresh_database):
    """The other direction: a cap reached is not the same as a queue behind it."""
    exactly = FakeMailbox(
        {str(n).encode(): an_email(message_id=f"<{n}@example.com>") for n in range(MAX_FETCHED)}
    )

    with psycopg.connect(fresh_database) as connection:
        summary = poll_mailbox(connection, exactly)

    assert summary.accepted == MAX_FETCHED
    assert summary.more_waiting is False


# --- the transaction ------------------------------------------------------------------------


def test_a_pass_needs_its_own_transaction(fresh_database):
    """It commits, so called on a connection with work open it would commit that too."""
    mailbox = FakeMailbox({b"1": an_email()})

    with psycopg.connect(fresh_database) as connection:
        connection.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="own transaction"):
            poll_mailbox(connection, mailbox)
