"""
Reading messages out of a real mailbox over IMAP.

Same shape as the fixture adapter: it produces IncomingMessage objects and knows nothing about runs,
the database, or the agent. What is different is who is on the other end. A JSONL file is written by
us; a mailbox is written by anyone who learns the address, so most of this module is refusal.

The module is called `mailbox` rather than `email` on purpose -- it has to import the standard
library's `email` package, and a sibling of that name would shadow it.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from email import message_from_bytes as parse_bytes
from email.message import EmailMessage
from email.policy import default as default_policy
from email.utils import parseaddr, parsedate_to_datetime
from hashlib import sha256
from typing import Any, Protocol

from app.contracts import Channel, IncomingMessage

log = logging.getLogger(__name__)

# One poll is one batch of work, not the whole backlog. Without a cap a mailbox holding ten thousand
# unread messages becomes ten thousand queued runs in a single pass, and the rate limits that protect
# the agent are all per sender -- none of them would stop it.
MAX_FETCHED = 50

# A cap on the count bounds how many messages a poll takes, and nothing else. These two bound how
# large one of them may be, which is a separate axis and was open until a security review closed it.
#
# The header cap is the one that matters. Python's structured header parser is worse than quadratic
# in the number of RFC 2047 encoded-words, and the cost lands on *reading* a header, not on parsing
# the message: a 664 KB Subject parses in a millisecond and then takes thirteen seconds to look at.
# So the check is on the raw bytes, before any header is touched. At this cap the worst case
# measured about eleven milliseconds, and real mail -- DKIM signatures, a long Received chain --
# fits inside it comfortably.
MAX_HEADER_BYTES = 32_768
MAX_RAW_BYTES = 2_000_000


# How much of a message we could not read is kept so a person can recognise it. Enough to see a
# subject line and the top of a body; not so much that a refusal costs what the message would have.
PREVIEW_CHARS = 2_000


@dataclass(frozen=True)
class Fetched:
    """
    One message the mailbox offered, read or refused, with the number that marks it.

    Exactly one of `message` and `refusal` is set. Both outcomes are carried rather than one being
    dropped, because a refusal is something a person needs to see -- and the caller cannot mark
    anything read without the number, which is the point of handing it back.

    `digest` and `preview` describe the bytes rather than the message, which is what makes them
    usable when there is no message. The digest is how a refusal gets recorded once however many
    times it is delivered: a message we could not read has no Message-ID we are willing to trust --
    that is frequently the very reason it was refused -- so the bytes are the only stable name it
    has.
    """

    number: bytes
    digest: str
    preview: str
    message: IncomingMessage | None = None
    refusal: str | None = None


class Mailbox(Protocol):
    """The part of imaplib.IMAP4 this adapter uses, so the tests can stand in for a mail server."""

    def select(self, mailbox: str) -> tuple[str, Any]: ...

    def search(self, charset: str | None, *criteria: str) -> tuple[str, Any]: ...

    def fetch(self, number: bytes, parts: str) -> tuple[str, Any]: ...

    def store(self, number: bytes, command: str, flags: str) -> tuple[str, Any]: ...


def message_from_bytes(raw: bytes) -> IncomingMessage:
    """
    Turn one email, as bytes off the wire, into a message the agent can read.

    Raises ValueError, naming the header at fault, for anything it cannot read honestly. Every one of
    those refusals is a thing it could have guessed at instead, and each guess has a cost paid later:
    an invented id makes a redelivery a second run, a defaulted date files a three-week-old complaint
    as new, a body read out of an image is a misquoted customer.
    """
    _refuse_if_oversize(raw)
    message = parse_bytes(raw, policy=default_policy)

    external_id = _identifier(message)
    sender = _address(message)
    received_at = _sent_at(message)
    body = _text(message)
    subject = message["Subject"]

    return IncomingMessage(
        channel=Channel.EMAIL,
        external_id=external_id,
        sender=sender,
        subject=str(subject) if subject is not None else None,
        body=body,
        received_at=received_at,
    )


def _refuse_if_oversize(raw: bytes) -> None:
    """
    Refuse a message too large to read, measured on the bytes, before anything parses them.

    Both refusals cost something real. An email carrying a large attachment is turned away whole,
    and the sentence we actually wanted goes with it; so is one with an unusually long header block.
    Both are visible -- the message stays unread in the mailbox and a person can go and look -- which
    is the trade this project keeps making. The alternative is an intake that anyone who can send
    mail can stop, and a stopped intake is silent.
    """
    if len(raw) > MAX_RAW_BYTES:
        raise ValueError(f"the email is too large to read: {len(raw)} bytes")

    end_of_headers = raw.find(b"\r\n\r\n")
    if end_of_headers == -1:
        end_of_headers = raw.find(b"\n\n")
    headers = raw if end_of_headers == -1 else raw[:end_of_headers]

    if len(headers) > MAX_HEADER_BYTES:
        raise ValueError(f"the email's headers are too large to read: {len(headers)} bytes")


def _identifier(message: EmailMessage) -> str:
    """
    The Message-ID, without its angle brackets.

    This is the only thing a mail server promises stays the same when it redelivers a message, which
    is what makes it the external_id. A hash of the body or the time of arrival would both change on
    a retry, and the same complaint would be answered -- and refunded -- twice.
    """
    header = message["Message-ID"]
    if header is None:
        raise ValueError("the email has no Message-ID, so intake cannot be idempotent on it")

    identifier = str(header).strip().strip("<>").strip()
    if not identifier:
        raise ValueError("the email's Message-ID is empty, so intake cannot be idempotent on it")
    return identifier


def _address(message: EmailMessage) -> str:
    """
    The address out of From, with any display name dropped.

    'Priya Sharma <priya@example.com>' and 'priya@example.com' are one person, and the per-sender
    rate limit only holds if they count as one.
    """
    _, address = parseaddr(str(message["From"] or ""))
    if not address:
        raise ValueError("the email has no From address to reply to")
    return address


def _sent_at(message: EmailMessage) -> datetime:
    """When the customer sent it, never when we happened to read it."""
    header = message["Date"]
    if header is None:
        raise ValueError("the email has no Date")

    try:
        return parsedate_to_datetime(str(header))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the email's Date cannot be read: {exc}") from exc


def _text(message: EmailMessage) -> str:
    """
    The plain-text part, preferred over the HTML one.

    A client that sends both sends the same sentence twice, once wrapped in markup we would otherwise
    pay for by the token on the way to a model. If there is no text part at all -- an image, a bare
    attachment -- that is refused rather than described, because a body the agent invented is worse
    than one it admits it does not have.
    """
    part = message.get_body(preferencelist=("plain",))
    if part is None:
        raise ValueError("the email has no text part to read")

    try:
        return part.get_content()
    except (LookupError, UnicodeError) as exc:
        # A charset this machine has no codec for. Left to escape as LookupError it would not be
        # caught by the poll loop, and one message naming an invented charset would end that poll
        # for everyone behind it.
        raise ValueError(f"the email's text cannot be decoded: {exc}") from exc


def unread_messages(mailbox: Mailbox) -> Iterator[Fetched]:
    """
    Yield every unread message in INBOX, up to MAX_FETCHED. Marks nothing read.

    Reading and marking are deliberately separate, and this is the reason: a message must not be
    marked read until the run it became has actually been committed. Marked here, a pass that then
    rolled back would leave the email read and no run anywhere -- a customer dropped in silence,
    which is the failure this whole project is about. So the number comes back alongside the message
    and the caller marks it once the work is durable.

    That trade is deliberate and it is not free: a pass that dies after committing and before
    marking will offer the same message again. Intake is idempotent on the Message-ID, so the repeat
    is a lookup that writes nothing. At-least-once is the safe direction to be wrong in.

    A message that cannot be read comes back as a refusal rather than being skipped. One unreadable
    message must not block the rest -- a mailbox is not a file, and whoever is behind it is waiting.
    """
    status, _ = mailbox.select("INBOX")
    if status != "OK":
        raise ValueError(f"the mailbox could not be opened: {status}")

    status, data = mailbox.search(None, "UNSEEN")
    if status != "OK":
        raise ValueError(f"the mailbox could not be searched: {status}")

    for number in _numbers(data)[:MAX_FETCHED]:
        raw = _fetch(mailbox, number)
        if raw is None:
            # The server offered the number and then would not produce the message. Nothing is
            # marked, so the next poll asks again.
            log.warning("message %r was listed but could not be fetched", number)
            continue

        digest = sha256(raw).hexdigest()
        # errors="replace" rather than a decode that could raise: this describes a message we may
        # be about to refuse, and it must not be able to fail in its turn.
        preview = raw[:PREVIEW_CHARS].decode("utf-8", errors="replace")

        try:
            message = message_from_bytes(raw)
        except ValueError as exc:
            yield Fetched(number=number, digest=digest, preview=preview, refusal=str(exc))
        else:
            yield Fetched(number=number, digest=digest, preview=preview, message=message)


def mark_read(mailbox: Mailbox, number: bytes) -> bool:
    """
    Mark one message read, reporting whether the flag stuck.

    Callers are expected to carry on when it did not: by the time this is called the run is written
    and committed, and refusing to continue would lose that. It is logged because a transient
    failure and a mailbox that never accepts a flag -- quietly spending every slot of every poll on
    the same message -- look identical from here, and only the second one needs a person.
    """
    status, _ = mailbox.store(number, "+FLAGS", "\\Seen")
    if status != "OK":
        log.warning("could not mark message %r read (%s); it will be offered again", number, status)
        return False
    return True


def _numbers(data: Any) -> list[bytes]:
    """The message numbers out of a search response, which arrives as one space-separated line."""
    if not data or not isinstance(data[0], bytes):
        return []
    return data[0].split()


def _fetch(mailbox: Mailbox, number: bytes) -> bytes | None:
    """
    One whole message, or None if the server did not give us one.

    IMAP fetch responses are a mixed list -- the message arrives as a tuple, and the closing paren
    arrives beside it as a bare bytestring -- so the shape is checked rather than indexed into.
    """
    status, parts = mailbox.fetch(number, "(RFC822)")
    if status != "OK" or not parts:
        return None

    for part in parts:
        if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], bytes):
            return part[1]
    return None
