"""
Turning a voice clip into an IncomingMessage.

The adapter calls Sarvam's speech-to-text endpoint, which authenticates with a header rather than a
URL path -- so the credential does not travel in the request line the way Telegram's does. It is
still a credential, so the tests here still hold the line: never in a traceback, never in a repr,
and the size cap has to fire before any of it leaves the box.
"""

import io
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from app.adapters.voice import (
    MAX_AUDIO_BYTES,
    TOKEN_VAR,
    SarvamSttClient,
    VoiceSettings,
    settings_from_env,
    transcribe,
)
from app.contracts import Channel

SECRET = "sarvam-fake-abcdefghijklmnopqrstuvwxyz012345"
CLIP = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 128
WHEN = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


class FakeStt:
    """The part of Sarvam this adapter uses. Records what was asked so a cap test can prove silence."""

    def __init__(self, transcript: str = "where is my order") -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, str]] = []

    def transcribe(self, audio: bytes, filename: str) -> dict:
        self.calls.append((audio, filename))
        return {"transcript": self.transcript, "language_code": "en-IN"}


# --- turning one clip into a message -----------------------------------------------------------


def test_a_transcript_becomes_a_voice_message():
    stt = FakeStt("i was charged twice for order 4821")

    found = transcribe(stt, CLIP, "note.wav", sender="+911234567890", received_at=WHEN)

    assert found.channel == Channel.VOICE
    assert found.body == "i was charged twice for order 4821"
    assert found.sender == "+911234567890"
    assert found.subject is None
    assert found.received_at == WHEN


def test_the_external_id_hashes_the_audio():
    """Identical clips must dedupe. The id is derived from the bytes, not the filename or the time."""
    stt = FakeStt()

    a = transcribe(stt, CLIP, "one.wav", sender="+91", received_at=WHEN)
    b = transcribe(stt, CLIP, "two.mp3", sender="+91", received_at=WHEN)

    assert a.external_id == b.external_id
    assert sha256(CLIP).hexdigest()[:24] in a.external_id


def test_a_different_clip_is_a_different_message():
    stt = FakeStt()

    a = transcribe(stt, CLIP, "one.wav", sender="+91", received_at=WHEN)
    b = transcribe(stt, CLIP + b"\x01", "two.wav", sender="+91", received_at=WHEN)

    assert a.external_id != b.external_id


# --- refusals that never touch the network -----------------------------------------------------


def test_audio_over_the_cap_is_refused_before_upload():
    """A cap Sarvam would enforce with a 413 has to fire here, so a huge blob never leaves the box."""
    stt = FakeStt()
    huge = b"\x00" * (MAX_AUDIO_BYTES + 1)

    with pytest.raises(ValueError, match="too large"):
        transcribe(stt, huge, "big.wav", sender="+91", received_at=WHEN)

    assert stt.calls == []


def test_an_unknown_extension_is_refused():
    stt = FakeStt()

    with pytest.raises(ValueError, match="extension"):
        transcribe(stt, CLIP, "note.txt", sender="+91", received_at=WHEN)

    assert stt.calls == []


def test_an_empty_transcript_is_refused():
    """A run with no body is a run with nothing to plan against."""
    stt = FakeStt(transcript="   ")

    with pytest.raises(ValueError, match="empty"):
        transcribe(stt, CLIP, "note.wav", sender="+91", received_at=WHEN)


# --- the credential never escapes --------------------------------------------------------------


def test_settings_repr_hides_the_secret():
    """
    The prefix is kept, so a log line can say which key was in use without saying how to be it.
    The rest is a fixed placeholder, so the length of the secret is not printed either.
    """
    settings = VoiceSettings(api_key=SECRET)

    assert "sarv" in repr(settings)
    assert "abcdefghijklmnopqrstuvwxyz012345" not in repr(settings)
    assert SECRET not in repr(settings)


def test_settings_from_env_refuses_a_missing_key():
    with pytest.raises(ValueError, match=TOKEN_VAR):
        settings_from_env({})


def test_the_api_key_does_not_leak_when_the_endpoint_is_unreachable():
    """
    urllib puts the URL in its exceptions, and Sarvam's key rides in a header rather than the URL --
    but the settings object still holds the key, and letting a handler frame escape would keep the
    key reachable to anything that walks frame locals. Rebuilt outside the handler, like Telegram's.
    """
    client = SarvamSttClient(VoiceSettings(api_key=SECRET), endpoint="http://127.0.0.1:1")

    with pytest.raises(ValueError) as refused:
        client.transcribe(CLIP, "note.wav")

    assert SECRET not in str(refused.value)
    assert refused.value.__context__ is None, "the chained error may still carry the key"

    # The deepest frame -- `_post`, where the raise happens -- must hold no local whose repr
    # prints the key, and must not hold `self` at all: `self.settings.api_key` is otherwise
    # walkable straight out. The `transcribe` frame further up still holds `self` by necessity
    # (it had to reach `_post` somehow); the defence there is that its repr does not leak.
    frames = []
    tb = refused.value.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    for frame in frames:
        for value in frame.f_locals.values():
            assert SECRET not in repr(value), f"the key's repr survives in {frame.f_code.co_name}"

    post_frames = [f for f in frames if f.f_code.co_name == "_post"]
    assert post_frames, "the traceback did not pass through _post at all"
    for frame in post_frames:
        assert "self" not in frame.f_locals, "_post kept its client instance on the frame"


def test_a_reply_too_large_to_read_is_refused_as_this_channels_problem():
    """A cap that borrowed the model client's exception would send the log looking at Ollama."""
    from app.adapters.voice import MAX_RESPONSE_BYTES, read_capped

    with pytest.raises(ValueError, match="too large"):
        read_capped(io.BytesIO(b"x" * (MAX_RESPONSE_BYTES + 1)), MAX_RESPONSE_BYTES)

    assert read_capped(io.BytesIO(b"{}"), MAX_RESPONSE_BYTES) == b"{}"
