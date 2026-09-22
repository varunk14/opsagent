"""
Turning a fixed reply into wav bytes, by way of Sarvam's text-to-speech.

Same shape as `voice.py` next door: it produces bytes and knows nothing about the outbox or the
run. The only user of it is `VoiceSender`, which reads a pending outbox row's body and stores the
returned bytes against the run.

The security discipline is the STT sibling's, unchanged: the api key rides in a header, every
failure inside `_post` is rebuilt outside the handler so no frame keeps the settings alive, and
the request payload is capped before it leaves the box.

The body Sarvam sends back is JSON with `{"audios": [<base64 wav>]}` -- one entry per input line,
and this adapter sends one line at a time. `audios[0]` is decoded and returned; anything else is
refused as a network fault, the same shape STT's callers already expect.
"""

import base64
import binascii
import json
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import IO, Any, Protocol

TOKEN_VAR = "OPSAGENT_SARVAM_API_KEY"
ENDPOINT = "https://api.sarvam.ai"
DEFAULT_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = "shubh"
DEFAULT_LANGUAGE = "en-IN"

TIMEOUT_SECONDS = 30
MAX_TTS_RESPONSE_BYTES = 8_000_000
# Sarvam's own cap for bulbul:v3. A reply that would not fit is refused before the call, so a
# template that has grown to run over is caught here rather than at the network.
MAX_TTS_TEXT_CHARS = 2500


@dataclass(frozen=True)
class TtsSettings:
    """
    The Sarvam api key, and nothing else.

    `repr` is written out because the generated one prints every field and these get held in
    frames. The first four characters are kept so a log line can say which key was in use without
    saying how to be it; the tail is a fixed placeholder, so the length of the secret is not
    printed either.
    """

    api_key: str

    def __repr__(self) -> str:
        head = self.api_key[:4] if self.api_key else ""
        return f"TtsSettings(api_key={head!r}+...)"


def settings_from_env(environ: Mapping[str, str]) -> TtsSettings:
    """The api key, refused by name if it is not there. Same env var as STT: one account, one key."""
    key = environ.get(TOKEN_VAR, "").strip()
    if not key:
        raise ValueError(f"the voice sender needs {TOKEN_VAR} set")
    return TtsSettings(api_key=key)


class SarvamTts(Protocol):
    """The part of Sarvam this adapter uses, so the tests can stand in for the real client."""

    def synthesize(self, text: str) -> bytes: ...


class SarvamTtsClient:
    """The real thing. Talks to Sarvam's text-to-speech endpoint over HTTPS."""

    def __init__(
        self,
        settings: TtsSettings,
        endpoint: str = ENDPOINT,
        model: str = DEFAULT_MODEL,
        speaker: str = DEFAULT_SPEAKER,
        language_code: str = DEFAULT_LANGUAGE,
    ) -> None:
        self.settings = settings
        self.endpoint = endpoint
        self.model = model
        self.speaker = speaker
        self.language_code = language_code

    def synthesize(self, text: str) -> bytes:
        payload = json.dumps({
            "text": text,
            "target_language_code": self.language_code,
            "speaker": self.speaker,
            "model": self.model,
        }).encode()
        reply = self._post("text-to-speech", payload, "application/json")
        return _first_audio(reply)

    def _post(self, path: str, data: bytes, content_type: str) -> dict[str, Any]:
        """
        One POST to a Sarvam path, with the api key kept out of whatever goes wrong.

        The same shape as `voice.py`'s `_post`: only the type name of the underlying failure
        survives, the raise happens after the handler so `__context__` is not the URL, and every
        local that could hold the key or the URL is dropped before the raise. `self` is dropped
        too -- otherwise `frame.f_locals['self'].settings.api_key` reads it straight out.
        """
        request = urllib.request.Request(
            f"{self.endpoint}/{path}",
            data=data,
            headers={
                "api-subscription-key": self.settings.api_key,
                "Content-Type": content_type,
            },
            method="POST",
        )

        failed_with: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                body = read_capped(response, MAX_TTS_RESPONSE_BYTES)
        except ValueError as exc:
            failed_with = str(exc)
        except (OSError, TimeoutError) as exc:  # HTTPError and URLError are both OSErrors
            failed_with = type(exc).__name__
        else:
            try:
                parsed = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                failed_with = f"the reply was not JSON: {exc}"
            else:
                if not isinstance(parsed, dict):
                    # Not a TypeError: callers already catch network shape refusals as ValueError.
                    failed_with = "Sarvam sent something that is not a reply"
                else:
                    return parsed

        del request
        del data
        del self
        raise ValueError(f"could not reach Sarvam TTS: {failed_with}")


def read_capped(stream: IO[bytes], limit: int) -> bytes:
    """
    Read a reply, refusing one larger than `limit` bytes.

    A plain ValueError -- the same shape STT uses -- so the drain that owns the send catches it as
    an ordinary network refusal rather than mistaking it for a fault in the language model.
    """
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"the Sarvam reply was too large to read: over {limit} bytes")
    return body


def synthesize(tts: SarvamTts, text: str) -> bytes:
    """
    Turn one line of reply into wav.

    The refusal for over-long text is here, before any network call, so a template that has grown
    past Sarvam's cap is caught by our own bound rather than by a 400. An empty line is refused
    too: a zero-byte wav on the run page reads as "the reply is missing", and that is the shape
    the customer would see.
    """
    if len(text) > MAX_TTS_TEXT_CHARS:
        raise ValueError(
            f"the reply is too long to synthesize: {len(text)} characters, over the "
            f"cap of {MAX_TTS_TEXT_CHARS}"
        )
    if not text.strip():
        raise ValueError("the reply is empty; nothing to synthesize")
    return tts.synthesize(text)


def _first_audio(reply: dict[str, Any]) -> bytes:
    """
    The wav bytes out of Sarvam's `{"audios": [<base64>]}` reply, decoded.

    A reply without an audio, or with an unreadable base64 payload, is refused as a network
    fault -- the same shape STT's callers catch, so no code above needs a new branch.
    """
    audios = reply.get("audios")
    if not isinstance(audios, list) or not audios:
        raise ValueError("Sarvam replied with no audio")
    head = audios[0]
    if not isinstance(head, str):
        # ValueError, not TypeError: callers already catch network shape refusals as ValueError.
        raise ValueError("Sarvam's audio was not a base64 string")  # noqa: TRY004
    try:
        return base64.b64decode(head, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Sarvam's audio could not be decoded: {exc}") from None
