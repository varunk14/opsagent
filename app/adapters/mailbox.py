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
from datetime import datetime
from email import message_from_bytes as parse_bytes
from email.message import EmailMessage
from email.policy import default as default_policy
from email.utils import parseaddr, parsedate_to_datetime
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


def unread_messages(mailbox: Mailbox) -> Iterator[IncomingMessage]:
    """
    Yield every unread message in INBOX, up to MAX_FETCHED, marking each one read as it goes.

    Marked *after* it parses, never before. A message the parser refuses is left unread, so it stays
    in the mailbox where a person can look at it, rather than being silently consumed by the poll
    that could not understand it.

    One unreadable message does not stop the rest. A mailbox is not a file: whoever is behind the
    malformed one is still waiting.
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
            continue

        try:
            message = message_from_bytes(raw)
        except ValueError:
            continue

        status, _ = mailbox.store(number, "+FLAGS", "\\Seen")
        if status != "OK":
            # Handed on regardless. Dropping it would lose a real customer to a transient IMAP
            # error, where keeping it costs one repeat on the next poll that intake recognises and
            # does not write. Logged because a transient failure and a mailbox that never accepts a
            # flag -- quietly spending every slot of every poll on the same message -- look the same
            # from here, and only the second one needs a person.
            log.warning(
                "could not mark %s read (%s); it will be offered again next poll",
                message.external_id,
                status,
            )

        yield message


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
