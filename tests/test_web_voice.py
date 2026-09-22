"""
The voice pages: an upload page that takes a clip, and a route that serves the spoken reply.

CSRF and origin discipline are the same as approvals: the token in the form must match the cookie,
and a form arriving from another site is refused. What is new here is a body cap the transport
layer enforces -- a client that lies about `Content-Length` or streams past the limit has the read
aborted, and the STT adapter never sees a byte over the ceiling.
"""

import re
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from app.web import create_app

pytestmark = pytest.mark.db

LOCAL = "http://127.0.0.1"


class FakeStt:
    def __init__(self, transcript: str = "where is my order") -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, str]] = []

    def transcribe(self, audio: bytes, filename: str) -> dict:
        self.calls.append((audio, filename))
        return {"transcript": self.transcript}


def client_for(dsn: str, stt: FakeStt | None = None, operator: str | None = "asha") -> TestClient:
    return TestClient(
        create_app(dsn=dsn, operator=operator, stt=stt),
        base_url=LOCAL,
    )


def token_from(page: str) -> str:
    found = re.search(r'name="csrf" value="([^"]+)"', page)
    assert found, "the page carries no CSRF token"
    return found.group(1)


def a_voice_reply(dsn: str, audio: bytes = b"WAV_HERE") -> str:
    run_id = uuid4()
    with psycopg.connect(dsn) as connection:
        connection.execute(
            "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
            "VALUES (%s, 'voice', 'done', 'act', %s, %s)",
            (run_id, Jsonb({"untrusted": {"body": "hi"}}), f"voice_msg_{run_id.hex[:16]}"),
        )
        connection.execute(
            "INSERT INTO voice_replies (run_id, audio) VALUES (%s, %s)", (run_id, audio)
        )
    return str(run_id)


# --- the upload page ------------------------------------------------------------------------


def test_the_upload_page_carries_a_csrf_token(fresh_database):
    page = client_for(fresh_database, stt=FakeStt()).get("/voice")

    assert page.status_code == 200
    assert 'name="csrf"' in page.text


def test_the_upload_page_shows_a_note_when_voice_is_not_configured(fresh_database):
    """`stt=None` is the shape the app ends up in without OPSAGENT_SARVAM_API_KEY."""
    page = client_for(fresh_database, stt=None).get("/voice")

    assert "not configured" in page.text


# --- the upload handler ---------------------------------------------------------------------


def test_a_clip_becomes_a_run(fresh_database):
    stt = FakeStt("i was charged twice")
    client = client_for(fresh_database, stt=stt)
    csrf = token_from(client.get("/voice").text)

    response = client.post(
        "/voice",
        data={"csrf": csrf, "sender": "Priya"},
        files={"clip": ("note.wav", b"AUDIO_BYTES_HERE", "audio/wav")},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/runs/")
    assert stt.calls == [(b"AUDIO_BYTES_HERE", "note.wav")]
    with psycopg.connect(fresh_database) as connection:
        row = connection.execute(
            "SELECT channel, state -> 'untrusted' ->> 'body' FROM runs WHERE channel = 'voice'"
        ).fetchone()
    assert row is not None
    assert row[0] == "voice"
    assert row[1] == "i was charged twice"


def test_an_upload_without_csrf_is_refused(fresh_database):
    client = client_for(fresh_database, stt=FakeStt())

    response = client.post(
        "/voice",
        data={"csrf": "not-the-right-token", "sender": "x"},
        files={"clip": ("a.wav", b"x", "audio/wav")},
    )

    assert response.status_code == 403


def test_an_unknown_extension_is_refused(fresh_database):
    stt = FakeStt()
    client = client_for(fresh_database, stt=stt)
    csrf = token_from(client.get("/voice").text)

    response = client.post(
        "/voice",
        data={"csrf": csrf, "sender": "x"},
        files={"clip": ("note.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 400
    assert stt.calls == []


def test_a_declared_body_over_the_cap_is_refused_before_parsing(fresh_database, monkeypatch):
    """
    The middleware runs before FastAPI touches the body. A client that declares an oversized
    body has the whole request refused before the multipart parser reads a byte.
    """
    monkeypatch.setattr("app.web.MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr("app.web.UPLOAD_ENVELOPE_SLACK", 0)
    stt = FakeStt()
    client = client_for(fresh_database, stt=stt)

    # No CSRF token needed: the pre-check refuses on length alone.
    response = client.post(
        "/voice",
        data={"csrf": "x", "sender": "x"},
        files={"clip": ("big.wav", b"x" * 2048, "audio/wav")},
    )

    assert response.status_code == 413
    assert stt.calls == []


def test_a_post_to_voice_without_content_length_is_refused(fresh_database):
    """
    The middleware requires a declared length: a client sending chunked-encoding without a
    Content-Length header is refused with 411 rather than let through to the parser.
    """
    stt = FakeStt()
    # httpx's TestClient always sends Content-Length; a bare request via the ASGI transport can
    # strip it. Ask the ASGI app directly with a minimal scope that carries no such header.
    from app.web import create_app
    app = create_app(dsn=fresh_database, operator="asha", stt=stt)

    # A crafted request with no content-length header. TestClient/HTTPX will still add one for
    # a request with a body, so instead we exercise the middleware branch by monkeypatching the
    # header out through a raw ASGI call:
    import asyncio

    async def call() -> tuple[int, bytes]:
        received: dict[str, object] = {"status": 0, "body": b""}

        async def receive() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(msg: dict) -> None:
            if msg["type"] == "http.response.start":
                received["status"] = msg["status"]
            if msg["type"] == "http.response.body":
                received["body"] += msg.get("body", b"")

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/voice",
            "raw_path": b"/voice",
            "query_string": b"",
            "headers": [(b"host", b"127.0.0.1")],  # no content-length
            "scheme": "http",
            "server": ("127.0.0.1", 8055),
            "client": ("127.0.0.1", 12345),
            "http_version": "1.1",
            "root_path": "",
        }
        await app(scope, receive, send)
        return received["status"], received["body"]  # type: ignore[return-value]

    status, _ = asyncio.run(call())
    assert status == 411
    assert stt.calls == []


def test_a_client_streaming_past_its_declared_length_is_refused(fresh_database, monkeypatch):
    """Defence in depth: bounded_read catches a body that grows past what the header claimed."""
    from unittest.mock import AsyncMock

    from app.web import UploadTooLarge, bounded_read

    huge = b"x" * 4096
    upload = type("U", (), {"read": AsyncMock(side_effect=[huge, huge, huge, b""])})()
    with pytest.raises(UploadTooLarge):
        import asyncio
        asyncio.run(bounded_read(upload, 1024))


def test_the_voice_nav_link_is_on_the_screen(fresh_database):
    """`base.html` extends every screen; the Voice link must appear alongside Approvals."""
    page = client_for(fresh_database, stt=FakeStt()).get("/approvals")

    assert 'href="/voice"' in page.text


# --- the wav route --------------------------------------------------------------------------


def test_a_synthesised_reply_is_served(fresh_database):
    run_id = a_voice_reply(fresh_database, audio=b"REAL_WAV_BYTES")

    response = client_for(fresh_database, stt=FakeStt()).get(f"/runs/{run_id}/reply.wav")

    assert response.status_code == 200
    assert response.content == b"REAL_WAV_BYTES"
    assert response.headers["content-type"] == "audio/wav"


def test_a_run_without_a_reply_is_a_404(fresh_database):
    """A missing wav is not the same as a broken player: a 404 keeps the page honest."""
    with psycopg.connect(fresh_database) as connection:
        run_id = uuid4()
        connection.execute(
            "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
            "VALUES (%s, 'voice', 'running', 'act', %s, %s)",
            (run_id, Jsonb({"untrusted": {"body": "hi"}}), "voice_msg_pending"),
        )

    response = client_for(fresh_database, stt=FakeStt()).get(f"/runs/{run_id}/reply.wav")

    assert response.status_code == 404


def test_a_non_uuid_is_a_404(fresh_database):
    response = client_for(fresh_database, stt=FakeStt()).get("/runs/not-a-uuid/reply.wav")

    assert response.status_code == 404
