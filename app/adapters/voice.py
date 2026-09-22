"""
Turning a voice clip into an IncomingMessage by way of Sarvam's speech-to-text.

The shape is the same as the mailbox and the Telegram adapters: it produces an IncomingMessage and
knows nothing about runs, the database, or the agent. Two things shape the code more than the
parsing does.

**The credential.** The api key rides in a header rather than a URL path, so the request line does
not carry it the way Telegram's does. It still travels through this module. The settings object
holds it and its repr scrubs it; every failure inside the request is rebuilt outside the handler so
no frame keeps `settings` or `request` alive on the traceback -- crash reporters and post-mortem
debuggers walk frame locals, and "every error path" should mean every one.

**The upload cap.** A blob refused by Sarvam with a 413 is a blob that already left the box. So the
cap fires here, before any network call, and the tests prove the fake client was not touched.
"""

import io
import json
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import IO, Any, Protocol

from app.contracts import Channel, IncomingMessage

TOKEN_VAR = "OPSAGENT_SARVAM_API_KEY"
ENDPOINT = "https://api.sarvam.ai"
DEFAULT_MODEL = "saaras:v3"

TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 8_000_000
MAX_AUDIO_BYTES = 25_000_000

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")

CONTENT_TYPES: Mapping[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


@dataclass(frozen=True)
class VoiceSettings:
    """
    The Sarvam api key, and nothing else.

    `repr` is written out because the generated one prints every field and these get held in frames.
    The first four characters are kept so a log line can say which key was in use without saying how
    to be it; the tail is a fixed placeholder, so the length of the secret is not printed either.
    """

    api_key: str

    def __repr__(self) -> str:
        head = self.api_key[:4] if self.api_key else ""
        return f"VoiceSettings(api_key={head!r}+...)"


def settings_from_env(environ: Mapping[str, str]) -> VoiceSettings:
    """The api key, refused by name if it is not there. No default: there is no default account."""
    key = environ.get(TOKEN_VAR, "").strip()
    if not key:
        raise ValueError(f"the voice adapter needs {TOKEN_VAR} set")
    return VoiceSettings(api_key=key)


class SarvamStt(Protocol):
    """The part of Sarvam this adapter uses, so the tests can stand in for the real client."""

    def transcribe(self, audio: bytes, filename: str) -> dict[str, Any]: ...


class SarvamSttClient:
    """The real thing. Talks to Sarvam's speech-to-text endpoint over HTTPS."""

    def __init__(
        self,
        settings: VoiceSettings,
        endpoint: str = ENDPOINT,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.settings = settings
        self.endpoint = endpoint
        self.model = model

    def transcribe(self, audio: bytes, filename: str) -> dict[str, Any]:
        content_type = CONTENT_TYPES.get(_ext(filename), "application/octet-stream")
        boundary = f"----opsagent-{uuid.uuid4().hex}"
        body = _multipart(audio, filename, content_type, self.model, boundary)
        raw = self._post("speech-to-text", body, f"multipart/form-data; boundary={boundary}")
        return _payload(raw)

    def _post(self, path: str, data: bytes, content_type: str) -> dict[str, Any]:
        """
        One POST to a Sarvam path, with the api key kept out of whatever goes wrong.

        Nothing from urllib is allowed out: its message and its `url` attribute carry the request
        line, and the settings object -- which holds the key -- is a local of this frame. Only the
        type name of the underlying failure survives, and the refusal is raised after the handler so
        that `__context__` does not keep a reference to the original either.
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
                body = read_capped(response, MAX_RESPONSE_BYTES)
        except ValueError as exc:
            failed_with = str(exc)
        except (OSError, TimeoutError) as exc:  # HTTPError and URLError are both OSErrors
            failed_with = type(exc).__name__
        else:
            try:
                return json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                failed_with = f"the reply was not JSON: {exc}"

        # Drop every local that could hold the key or the URL before the raise. `self` is
        # dropped too: a debugger walking `frame.f_locals['self'].settings.api_key` would
        # otherwise read the raw key straight out of this frame -- the repr scrub only hides
        # it from `print`, not from attribute access. The caller `transcribe` still holds
        # `self` on its own frame; that is a real limit and is the reason its locals are
        # kept to repr-scrubbed things (a debugger walking two levels up can still reach
        # `settings.api_key`; this is the same limit `telegram.py` lives with).
        del request
        del data
        del self
        raise ValueError(f"could not reach Sarvam STT: {failed_with}")


def read_capped(stream: IO[bytes], limit: int) -> bytes:
    """
    Read a reply, refusing one larger than `limit` bytes.

    The refusal is a plain ValueError -- this adapter's callers already catch that shape. Borrowing
    the model client's ModelUnavailable would send whoever read the log looking at Ollama for a
    fault that was never there.
    """
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"the Sarvam reply was too large to read: over {limit} bytes")
    return body


def transcribe(
    stt: SarvamStt,
    audio: bytes,
    filename: str,
    *,
    sender: str,
    received_at: datetime,
) -> IncomingMessage:
    """
    Turn one voice clip into a message the agent can read.

    The refusals here are the ones the network can't make for us: an oversized blob that would
    otherwise leave the box before Sarvam refused it, and an unknown extension that Sarvam would
    accept and quietly transcribe as nothing. An empty transcript is refused after the call --
    Sarvam did answer, it just had nothing to say, and there is nothing for the agent to plan
    against.
    """
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError(f"the voice clip is too large to upload: over {MAX_AUDIO_BYTES} bytes")
    if _ext(filename) not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"the file extension is not one Sarvam accepts: {filename!r}; "
            f"try one of {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    result = stt.transcribe(audio, filename)
    transcript = str(result.get("transcript", "")).strip()
    if not transcript:
        raise ValueError("the transcript was empty; nothing for the agent to read")

    return IncomingMessage(
        channel=Channel.VOICE,
        external_id=f"voice_{sha256(audio).hexdigest()[:24]}",
        sender=sender,
        subject=None,
        body=transcript,
        received_at=received_at,
    )


def _ext(filename: str) -> str:
    """The lowercased suffix, with the dot. Empty string when there is none."""
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot >= 0 else ""


def _multipart(
    audio: bytes,
    filename: str,
    content_type: str,
    model: str,
    boundary: str,
) -> bytes:
    """
    A minimal multipart body: one file part and one model part.

    Filename and boundary are both escape hatches -- the filename comes from the caller and the
    boundary is a fresh uuid, so neither can be user-controlled in a way that closes the boundary.
    Written by hand rather than by `email.mime` to avoid its header line-wrapping, which would
    otherwise mangle the content-type for some replies.
    """
    safe = filename.replace('"', "").replace("\r", "").replace("\n", "")
    buf = io.BytesIO()
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(f'Content-Disposition: form-data; name="file"; filename="{safe}"\r\n'.encode())
    buf.write(f"Content-Type: {content_type}\r\n\r\n".encode())
    buf.write(audio)
    buf.write(f"\r\n--{boundary}\r\n".encode())
    buf.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    buf.write(model.encode())
    buf.write(f"\r\n--{boundary}--\r\n".encode())
    return buf.getvalue()


def _payload(raw: Any) -> dict[str, Any]:
    """
    Read the parts of the reply this adapter uses. Any other keys are ignored.

    Sarvam does not always send `language_code`, so a missing key is not a refusal here -- the
    caller only requires a transcript, and it decides what to do when there is none.
    """
    if not isinstance(raw, dict):
        # A ValueError, not TypeError, because callers already catch the network refusals as one.
        raise ValueError("Sarvam sent something that is not a reply")  # noqa: TRY004
    return raw
