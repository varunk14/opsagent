"""
The voice sender: text becomes wav, and the wav is stored against the run.

Same ordering as the other senders -- the drain marks the row sent AFTER `send` returns, so a
sender that raises leaves the row pending for the next drain. What makes voice different is that
the wav is stored in a table rather than posted to a chat: the customer plays it back on the
screen. So a successful send commits a row in `voice_replies` and returns.
"""

from uuid import uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

from app.replies.send import STORE_REPLY, Outgoing, VoiceSender

pytestmark = pytest.mark.db


class FakeTts:
    def __init__(self, audio: bytes = b"WAV_BYTES") -> None:
        self.audio = audio
        self.calls: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return self.audio


class Broken:
    def synthesize(self, text: str) -> bytes:
        raise RuntimeError("Sarvam refused")


def a_voice_run(connection: psycopg.Connection, key: str = "voice_msg_abcd1234") -> str:
    """One voice run, shaped as the intake would leave it."""
    run_id = uuid4()
    connection.execute(
        "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
        "VALUES (%s, 'voice', 'done', 'act', %s, %s)",
        (run_id, Jsonb({"untrusted": {"body": "hi"}}), key),
    )
    return str(run_id)


def an_outgoing(run_id: str, body: str = "Thanks for your message.") -> Outgoing:
    return Outgoing(
        id=1,
        run_id=run_id,  # type: ignore[arg-type]
        channel="voice",
        reply_to=run_id,
        thread_ref=None,
        template="enquiry",
        body=body,
    )


def test_a_synthesised_reply_lands_in_voice_replies(fresh_database: str):
    tts = FakeTts(b"WAV_BYTES_HERE")
    with psycopg.connect(fresh_database) as setup:
        run_id = a_voice_run(setup)
        setup.commit()

    sender = VoiceSender(tts=tts, dsn=fresh_database)
    sender.send(an_outgoing(run_id))

    assert tts.calls == ["Thanks for your message."]
    with psycopg.connect(fresh_database) as check:
        row = check.execute(
            "SELECT audio FROM voice_replies WHERE run_id = %s", (run_id,)
        ).fetchone()
    assert row is not None
    assert bytes(row[0]) == b"WAV_BYTES_HERE"


def test_synthesis_failing_raises_and_writes_nothing(fresh_database: str):
    """The drain reads a raise as a pending row that will be retried; nothing must persist."""
    with psycopg.connect(fresh_database) as setup:
        run_id = a_voice_run(setup)
        setup.commit()

    sender = VoiceSender(tts=Broken(), dsn=fresh_database)

    with pytest.raises(RuntimeError, match="refused"):
        sender.send(an_outgoing(run_id))

    with psycopg.connect(fresh_database) as check:
        row = check.execute(
            "SELECT count(*) FROM voice_replies WHERE run_id = %s", (run_id,)
        ).fetchone()
    assert row is not None
    assert row[0] == 0


def test_a_repeat_send_does_not_overwrite_the_wav(fresh_database: str):
    """`ON CONFLICT DO NOTHING` on the PK: the first wav wins, so a retry cannot silently change it."""
    with psycopg.connect(fresh_database) as setup:
        run_id = a_voice_run(setup)
        setup.commit()

    VoiceSender(tts=FakeTts(b"FIRST"), dsn=fresh_database).send(an_outgoing(run_id))
    VoiceSender(tts=FakeTts(b"SECOND"), dsn=fresh_database).send(an_outgoing(run_id))

    with psycopg.connect(fresh_database) as check:
        row = check.execute(
            "SELECT audio FROM voice_replies WHERE run_id = %s", (run_id,)
        ).fetchone()
    assert row is not None
    assert bytes(row[0]) == b"FIRST"


def test_the_store_query_uses_do_nothing():
    """A rewritten INSERT that dropped the guard could overwrite a wav; check the SQL is what we think."""
    assert "ON CONFLICT (run_id) DO NOTHING" in STORE_REPLY
