"""
Reading messages out of a real mailbox over IMAP.

Same shape as the fixture adapter: it produces IncomingMessage objects and knows nothing about runs,
the database, or the agent. What is different is who is on the other end. A JSONL file is written by
us; a mailbox is written by anyone who learns the address, so most of this module is refusal.

The module is called `mailbox` rather than `email` on purpose -- it has to import the standard
library's `email` package, and a sibling of that name would shadow it.
"""

from collections.abc import Iterator
from datetime import datetime
from email import message_from_bytes as parse_bytes
from email.message import EmailMessage
from email.policy import default as default_policy
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Protocol

from app.contracts import Channel, IncomingMessage

# One poll is one batch of work, not the whole backlog. Without a cap a mailbox holding ten thousand
# unread messages becomes ten thousand queued runs in a single pass, and the rate limits that protect
# the agent are all per sender -- none of them would stop it.
MAX_FETCHED = 50


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

        mailbox.store(number, "+FLAGS", "\\Seen")
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
