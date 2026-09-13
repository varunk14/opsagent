"""
The first intake adapter: messages from a file.

Gmail comes later and is the same shape -- something that produces
IncomingMessage objects and knows nothing about what happens next. Starting with
a file is not a shortcut. It means the whole intake path is testable with no
credential, runs identically in CI, and is replayable: the same file in, the same
run ids out, which is exactly what week 7's golden set will need.

JSONL because that is the format the eval set will use, and having one format
for both means a case that breaks in production can be appended to the golden
file without translation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.adapters.fixture import read_messages
from app.contracts import Channel

PRIYA = {
    "channel": "email",
    "external_id": "9f2a",
    "sender": "priya@example.com",
    "subject": "Charged twice for order #4821",
    "body": "Hi, I think I was charged twice for order #4821 last Tuesday.",
    "received_at": "2026-09-13T09:00:00+00:00",
}


def write_lines(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "inbox.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


# --- reading ----------------------------------------------------------------


def test_each_line_becomes_a_message(tmp_path):
    path = write_lines(tmp_path, json.dumps(PRIYA), json.dumps({**PRIYA, "external_id": "7c1b"}))

    messages = list(read_messages(path))

    assert [m.external_id for m in messages] == ["9f2a", "7c1b"]


def test_the_message_is_fully_typed(tmp_path):
    path = write_lines(tmp_path, json.dumps(PRIYA))

    message = next(iter(read_messages(path)))

    assert message.channel is Channel.EMAIL
    assert message.received_at == datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc)
    assert message.idempotency_key == "email_msg_9f2a"


def test_reading_the_same_file_twice_gives_the_same_keys(tmp_path):
    """
    Replay depends on this. If a key changed between reads, re-running a case
    would create a new run rather than being recognised as the same one.
    """
    path = write_lines(tmp_path, json.dumps(PRIYA))

    first = [m.idempotency_key for m in read_messages(path)]
    second = [m.idempotency_key for m in read_messages(path)]

    assert first == second


def test_blank_lines_are_ignored(tmp_path):
    path = write_lines(tmp_path, json.dumps(PRIYA), "", "   ")

    assert len(list(read_messages(path))) == 1


def test_an_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "inbox.jsonl"
    path.write_text("")

    assert list(read_messages(path)) == []


# --- refusing to guess ------------------------------------------------------


def test_a_line_that_is_not_json_names_itself(tmp_path):
    """
    Silence here would be the worst outcome: a malformed line skipped quietly is
    a customer whose email was never answered, and nothing anywhere says so.
    """
    path = write_lines(tmp_path, json.dumps(PRIYA), "{not json")

    with pytest.raises(ValueError, match="line 2"):
        list(read_messages(path))


def test_a_line_missing_a_required_field_names_itself(tmp_path):
    path = write_lines(tmp_path, json.dumps({k: v for k, v in PRIYA.items() if k != "body"}))

    with pytest.raises(ValueError, match="line 1"):
        list(read_messages(path))


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_messages(tmp_path / "nothing.jsonl"))


# --- the committed fixture --------------------------------------------------


def test_the_repository_fixture_parses(fixture_inbox):
    """The file the README tells a reader to run must actually work."""
    messages = list(read_messages(fixture_inbox))

    assert len(messages) >= 3
    assert len({m.idempotency_key for m in messages}) == len(messages)
