"""
Synthesising a reply, and keeping the api key out of every log and traceback.

Same shape as the STT adapter next door: the JSON reply carries the wav bytes as base64, and the
tests here pin what happens when Sarvam is not there. The security discipline is the same as its
STT sibling -- the key rides in a header, never in a URL, and every failure inside the request is
rebuilt outside the handler so the settings object is not held on a frame that could be walked.
"""

import base64
from hashlib import sha256

import pytest

from app.adapters.voice_tts import (
    MAX_TTS_TEXT_CHARS,
    TOKEN_VAR,
    SarvamTtsClient,
    TtsSettings,
    settings_from_env,
    synthesize,
)

SECRET = "sarvam-fake-abcdefghijklmnopqrstuvwxyz012345"
WAV = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 64


class FakeTts:
    """The part of Sarvam this adapter uses; records inputs so a cap test can prove silence."""

    def __init__(self, audio: bytes = WAV) -> None:
        self.audio = audio
        self.calls: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return self.audio


# --- turning text into wav ---------------------------------------------------------------------


def test_a_line_becomes_wav_bytes():
    tts = FakeTts()

    audio = synthesize(tts, "your refund is on its way")

    assert audio == WAV
    assert tts.calls == ["your refund is on its way"]


def test_the_same_text_produces_the_same_bytes():
    """A cache above this layer keys on the text; identical input must not vary here."""
    tts = FakeTts()

    a = synthesize(tts, "thanks for your patience")
    b = synthesize(tts, "thanks for your patience")

    assert sha256(a).digest() == sha256(b).digest()


def test_text_over_the_cap_is_refused_before_upload():
    tts = FakeTts()
    long = "x" * (MAX_TTS_TEXT_CHARS + 1)

    with pytest.raises(ValueError, match="too long"):
        synthesize(tts, long)

    assert tts.calls == []


def test_empty_text_is_refused():
    tts = FakeTts()

    with pytest.raises(ValueError, match="empty"):
        synthesize(tts, "   ")

    assert tts.calls == []


# --- the credential never escapes --------------------------------------------------------------


def test_settings_repr_hides_the_secret():
    settings = TtsSettings(api_key=SECRET)

    assert "sarv" in repr(settings)
    assert "abcdefghijklmnopqrstuvwxyz012345" not in repr(settings)
    assert SECRET not in repr(settings)


def test_settings_from_env_refuses_a_missing_key():
    with pytest.raises(ValueError, match=TOKEN_VAR):
        settings_from_env({})


def test_the_api_key_does_not_leak_when_the_endpoint_is_unreachable():
    """Same guarantee the STT adapter holds: `_post`'s frame drops `self` before it raises."""
    client = SarvamTtsClient(TtsSettings(api_key=SECRET), endpoint="http://127.0.0.1:1")

    with pytest.raises(ValueError) as refused:
        client.synthesize("hello")

    assert SECRET not in str(refused.value)
    assert refused.value.__context__ is None, "the chained error may still carry the key"

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


def test_a_reply_missing_audio_is_refused(monkeypatch):
    """`{"audios": []}` is Sarvam's shape for a reply that carried no audio; refuse it as a network fault."""
    from app.adapters import voice_tts

    monkeypatch.setattr(
        voice_tts.SarvamTtsClient, "_post",
        lambda self, path, data, content_type: {"audios": []},
    )
    client = SarvamTtsClient(TtsSettings(api_key=SECRET))

    with pytest.raises(ValueError, match="no audio"):
        client.synthesize("hello")


def test_the_wav_bytes_match_the_base64_reply(monkeypatch):
    """A round trip through the JSON shape Sarvam actually sends, without touching the network."""
    from app.adapters import voice_tts

    encoded = base64.b64encode(b"WAV_BYTES_HERE").decode()
    monkeypatch.setattr(
        voice_tts.SarvamTtsClient, "_post",
        lambda self, path, data, content_type: {"audios": [encoded]},
    )
    client = SarvamTtsClient(TtsSettings(api_key=SECRET))

    assert client.synthesize("hello") == b"WAV_BYTES_HERE"
