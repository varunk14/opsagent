"""
Draining the outbox: each pending reply is sent, and only a reply that was sent is marked sent.

This is the poll ordering from the other side. A poll records the run first and tells the channel
afterwards, because telling it first could lose a message. A drain sends first and marks afterwards,
for the same reason: marking first would let a crash in the gap leave a row that claims to have been
sent with nothing sent. So the send happens, and the mark is what commits it. Delivery is
at-least-once -- a crash after the send but before the mark sends again on the next drain -- because
a customer hearing twice is the safe way to be wrong and hearing nothing is not.

A send that fails is written down, not dropped: its attempt is counted and its error kept, and the
row stays pending until it has been tried MAX_SEND_ATTEMPTS times, after which it rests as failed for
a person to look at rather than being retried forever. One channel's outage costs only its own
replies: a drain claims a row per transaction with FOR UPDATE SKIP LOCKED, so a reply on a working
channel is sent whatever is happening on another.

Run:  .venv/bin/python -m app.replies    one drain over the outbox
"""

import smtplib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol
from uuid import UUID

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.rows import class_row

from app.adapters.mailbox import PASSWORD_VAR as MAIL_PASSWORD_VAR
from app.adapters.mailbox import USER_VAR as MAIL_USER_VAR
from app.adapters.telegram import TOKEN_VAR as TELEGRAM_TOKEN_VAR
from app.adapters.voice_tts import (
    TOKEN_VAR as SARVAM_TOKEN_VAR,
)
from app.adapters.voice_tts import (
    SarvamTts,
    SarvamTtsClient,
    TtsSettings,
    synthesize,
)
from app.db import connect

# Large enough to clear an ordinary backlog in one drain, small enough that a drain stays short.
DEFAULT_LIMIT = 500

# How many times a reply is tried before it rests as failed for a person. A transient outage clears
# well within this; a row that has failed this many times has something wrong a retry will not fix.
MAX_SEND_ATTEMPTS = 5

CLAIM_ONE = """
    SELECT id, run_id, channel, reply_to, thread_ref, template, body
      FROM outbox
     WHERE state = 'pending' AND channel = ANY(%s) AND NOT (id = ANY(%s))
     ORDER BY created_at, id
       FOR UPDATE SKIP LOCKED
     LIMIT 1
"""

# attempts is read before the update, so the ceiling is compared against the count including this try.
COUNT_ATTEMPT = """
    UPDATE outbox
       SET attempts = attempts + 1,
           last_error = %s,
           state = CASE WHEN attempts + 1 >= %s THEN 'failed' ELSE 'pending' END
     WHERE id = %s
"""

MARK_SENT = "UPDATE outbox SET state = 'sent', sent_at = now() WHERE id = %s"


@dataclass(frozen=True)
class Outgoing:
    """One reply to put on its channel. Everything the sender needs is here; it reads no database."""

    id: int
    run_id: UUID
    channel: str
    reply_to: str
    thread_ref: str | None
    template: str
    body: str


class Sender(Protocol):
    def send(self, reply: Outgoing) -> None:
        """Put the reply on the channel. Raises on any failure, so the drain keeps the row pending."""


@dataclass(frozen=True)
class DrainSummary:
    sent: int
    failed: int


def mark_sent(connection: psycopg.Connection, row_id: int) -> None:
    """Marked sent only after the send returned. A module function so a test can break this step."""
    connection.execute(MARK_SENT, (row_id,))


def count_attempt(connection: psycopg.Connection, row_id: int, error: str) -> None:
    connection.execute(COUNT_ATTEMPT, (error, MAX_SEND_ATTEMPTS, row_id))


def drain(
    connection: psycopg.Connection, senders: Mapping[str, Sender], *, limit: int = DEFAULT_LIMIT
) -> DrainSummary:
    """
    Send pending replies for the channels we have a sender for, oldest first, up to `limit`.

    A row is claimed, sent, and marked in one transaction, so two drains never send the same reply
    and a reply is marked sent only once its send has returned. A send that raises leaves the row
    pending with its attempt counted; a send that returns commits the mark. Commits per reply.
    """
    if connection.pgconn.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError(
            "drain commits, so it needs its own transaction: call it on a "
            "connection with no work already open"
        )

    channels = list(senders)
    # A reply whose send failed stays pending for the NEXT drain, not this one: retried here it would
    # be re-claimed the instant it was set down, and one dead channel would spin the whole budget.
    attempted: list[int] = []
    sent = failed = 0
    for _ in range(limit):
        with connection.transaction():
            # Mapped by column name, not position, so reordering CLAIM_ONE's SELECT can never
            # silently swap two same-typed fields -- channel for template, say -- and misroute a reply.
            reply = (
                connection.cursor(row_factory=class_row(Outgoing))
                .execute(CLAIM_ONE, (channels, attempted))
                .fetchone()
            )
            if reply is None:
                break
            attempted.append(reply.id)
            try:
                senders[reply.channel].send(reply)
            except Exception as failure:  # noqa: BLE001 - any send failure is kept and retried, never dropped
                count_attempt(connection, reply.id, str(failure))
                failed += 1
                continue
            mark_sent(connection, reply.id)
            sent += 1

    return DrainSummary(sent=sent, failed=failed)


# --- the real channels --------------------------------------------------------------------------
# Network senders, exercised against the real services and not in the suite: the drain's ordering is
# what the tests pin, with fakes, and that is the part a bug would hide in.

# The mailbox is read over IMAP and replied to over SMTP with the same account, so the user and
# password are the ones the mailbox adapter already reads; only the SMTP host and port are new.
SMTP_HOST_VAR = "OPSAGENT_SMTP_HOST"
SMTP_PORT_VAR = "OPSAGENT_SMTP_PORT"


@dataclass(frozen=True)
class EmailSender:
    """Replies by SMTP, threaded onto the original message so it lands in the same conversation."""

    host: str
    port: int
    user: str
    password: str

    def __repr__(self) -> str:
        # Written out, like MailboxSettings: the generated repr prints the app password, and a
        # sender held in a frame would put it into any traceback or log below this point.
        return f"EmailSender(host={self.host!r}, port={self.port}, user={self.user!r}, password=...)"

    def send(self, reply: Outgoing) -> None:  # pragma: no cover - needs a real SMTP server
        message = EmailMessage()
        message["From"] = self.user
        message["To"] = reply.reply_to
        message["Subject"] = "Re: your support request"
        if reply.thread_ref:
            angled = f"<{reply.thread_ref}>"
            message["In-Reply-To"] = angled
            message["References"] = angled
        message.set_content(reply.body)
        with smtplib.SMTP(self.host, self.port, timeout=30) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(self.user, self.password)
            server.send_message(message)


@dataclass(frozen=True)
class TelegramSender:
    """Replies by sendMessage to the chat the message came from."""

    token: str

    def __repr__(self) -> str:
        # The token is the whole credential -- anyone holding it can post as the bot. Never in a repr.
        return "TelegramSender(token=...)"

    def send(self, reply: Outgoing) -> None:  # pragma: no cover - needs the real Bot API
        payload = urllib.parse.urlencode({"chat_id": reply.reply_to, "text": reply.body}).encode()
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendMessage", data=payload
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"Telegram refused the reply: {response.status}")
        except urllib.error.URLError as unreachable:
            raise RuntimeError(f"Telegram could not be reached: {unreachable.reason}") from None


STORE_REPLY = """
    INSERT INTO voice_replies (run_id, audio)
    VALUES (%s, %s)
    ON CONFLICT (run_id) DO NOTHING
"""


@dataclass(frozen=True)
class VoiceSender:
    """
    Replies by asking Sarvam to speak the reply, then storing the wav against the run.

    The wav is written in this sender's own transaction, on a fresh connection: the drain owns its
    transaction and pushes the outbox row's state, and mixing two writers on one connection would
    put the drain's own mark at the mercy of the synth's commit. On success the drain marks the
    outbox row sent as usual; on a synth or upload failure the drain leaves the row pending, and
    the next drain re-synthesises -- `ON CONFLICT DO NOTHING` is what keeps the wav from being
    written twice.

    The customer listens back on the run's own page: `GET /runs/{id}/reply.wav` reads the row.
    """

    tts: SarvamTts
    dsn: str | None = None  # None means "use OPSAGENT_DATABASE_URL", like every other db call.

    def __repr__(self) -> str:
        # No credential lives on this sender: the tts client holds its own scrubbed settings.
        return "VoiceSender(tts=..., dsn=...)"

    def send(self, reply: Outgoing) -> None:
        audio = synthesize(self.tts, reply.body)
        with connect(self.dsn) as connection:
            connection.execute(STORE_REPLY, (reply.run_id, audio))


def senders_from_env(environ: Mapping[str, str]) -> dict[str, Sender]:  # pragma: no cover - wiring
    """Build a sender for each channel the environment has been told how to reach."""
    senders: dict[str, Sender] = {}
    host = environ.get(SMTP_HOST_VAR)
    user = environ.get(MAIL_USER_VAR)
    password = environ.get(MAIL_PASSWORD_VAR)
    if host and user and password:
        senders["email"] = EmailSender(host, int(environ.get(SMTP_PORT_VAR, "587")), user, password)
    token = environ.get(TELEGRAM_TOKEN_VAR)
    if token:
        senders["telegram"] = TelegramSender(token)
    sarvam_key = environ.get(SARVAM_TOKEN_VAR)
    if sarvam_key:
        senders["voice"] = VoiceSender(tts=SarvamTtsClient(TtsSettings(api_key=sarvam_key)))
    return senders
