"""
Reading messages out of a real mailbox over IMAP.

Same shape as the fixture adapter: it produces IncomingMessage objects and knows nothing about runs,
the database, or the agent. What is different is who is on the other end. A JSONL file is written by
us; a mailbox is written by anyone who learns the address, so most of this module is refusal.

The module is called `mailbox` rather than `email` on purpose -- it has to import the standard
library's `email` package, and a sibling of that name would shadow it.
"""

import imaplib
import logging
import ssl
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from email import message_from_bytes as parse_bytes
from email.message import EmailMessage
from email.policy import default as default_policy
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Protocol

from app.adapters.inbox import Fetched, Unread, described
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

# Where the mailbox's details come from. The password belongs in a .env file that is not committed;
# it is an app password for one mailbox, which can be revoked on its own without touching the
# account it belongs to.
HOST_VAR = "OPSAGENT_IMAP_HOST"
USER_VAR = "OPSAGENT_IMAP_USER"
PASSWORD_VAR = "OPSAGENT_IMAP_PASSWORD"
PORT_VAR = "OPSAGENT_IMAP_PORT"
DEFAULT_PORT = 993


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


@dataclass(frozen=True)
class MailboxSettings:
    """
    What is needed to reach one mailbox. The password is an app password, never an account password.

    `repr` is written out rather than inherited because the generated one prints every field, and
    these get held in frames: one unhandled error anywhere below this and the default would put a
    live app password into a traceback, a log file, and whatever collects that log file.
    """

    host: str
    user: str
    password: str
    port: int = 993

    def __repr__(self) -> str:
        return f"MailboxSettings(host={self.host!r}, user={self.user!r}, port={self.port}, password=...)"


def settings_from_env(environ: Mapping[str, str]) -> MailboxSettings:
    """
    Read the mailbox settings, refusing by name if any is missing.

    Nothing here has a default except the port. A default host would be someone else's mail server,
    tried with a real password; a poller running with no password fails identically to a mail server
    that is down, and someone spends an afternoon on it.
    """
    missing = [name for name in (HOST_VAR, USER_VAR, PASSWORD_VAR) if not environ.get(name, "").strip()]
    if missing:
        raise ValueError(f"the mailbox needs {' and '.join(missing)} set")

    port = environ.get(PORT_VAR, "").strip() or str(DEFAULT_PORT)
    if not port.isdigit():
        # Deliberately quotes the port and nothing else: the password is in the same environment.
        raise ValueError(f"{PORT_VAR} is not a port number: {port!r}")

    return MailboxSettings(
        host=environ[HOST_VAR].strip(),
        user=environ[USER_VAR].strip(),
        # Not stripped. Some providers issue app passwords with spaces in them, and one helpfully
        # removed space is an authentication failure nobody can see the cause of.
        password=environ[PASSWORD_VAR],
        port=int(port),
    )


def open_mailbox(settings: MailboxSettings, connect: Callable[..., Any] = imaplib.IMAP4_SSL) -> Any:
    """
    Open and log in to the mailbox described by `settings`.

    IMAP4_SSL, with no option anywhere for the plain kind. Plain IMAP4 would work against most
    servers and put the app password on the network in the clear, and nothing about the poller's
    behaviour would look the slightest bit different -- which is exactly why it is not offered.

    The ssl_context is passed explicitly, and that is not decoration. `imaplib.IMAP4_SSL(host, port)`
    with no context falls back to `ssl._create_stdlib_context()`, which in CPython is an alias for
    `_create_unverified_context`: CERT_NONE, check_hostname off. Encrypted, and to nobody in
    particular -- anyone on the path can offer a self-signed certificate, be believed, and be handed
    the app password. Encryption without authentication ends up where plaintext does, just with a
    more convincing name.

    A failure is re-raised naming the mailbox and never the secret. The moment a login is rejected
    is the moment the obvious implementation echoes back what it tried.
    """
    failed_with: str | None = None
    try:
        mailbox = connect(settings.host, settings.port, ssl_context=ssl.create_default_context())
        mailbox.login(settings.user, settings.password)
    except (OSError, imaplib.IMAP4.error) as exc:
        # Only the type name is kept. imaplib builds its login error out of the server's own reply
        # text, and the server is the one party that has already been sent the password -- a hostile
        # one can simply echo it back.
        failed_with = type(exc).__name__

    # Raised out here, after the handler, and that placement is the point. Raising inside it would
    # attach the original as __context__ no matter what: `from None` only suppresses the chain when
    # a traceback is *printed*, it does not clear the link, and anything that walks it -- a crash
    # reporter, a debugger -- would read the server's text back out. Outside the handler there is no
    # active exception left to attach.
    if failed_with is not None:
        raise ValueError(f"could not open {settings.user} at {settings.host}: {failed_with}")
    return mailbox


def unread_messages(mailbox: Mailbox) -> Unread:
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

    `waiting` is the count the server gave, not the count we managed to read. Those differ exactly
    when a message is listed and then not delivered, and the difference matters in the wrong
    direction: inferring the backlog from what came back would say "nothing left" while a real one
    sat behind the message that failed, and whoever schedules the next pass would believe it.
    """
    status, _ = mailbox.select("INBOX")
    if status != "OK":
        raise ValueError(f"the mailbox could not be opened: {status}")

    status, data = mailbox.search(None, "UNSEEN")
    if status != "OK":
        raise ValueError(f"the mailbox could not be searched: {status}")

    numbers = _numbers(data)
    return Unread(waiting=len(numbers), messages=_read(mailbox, numbers[:MAX_FETCHED]))


def _read(mailbox: Mailbox, numbers: list[bytes]) -> Iterator[Fetched]:
    """The reading half of `unread_messages`, separated so the count is known before it starts."""
    for number in numbers:
        raw = _fetch(mailbox, number)
        if raw is None:
            # The server offered the number and then would not produce the message. Nothing is
            # marked, so the next poll asks again -- and `waiting` still counts it.
            log.warning("message %r was listed but could not be fetched", number)
            continue

        digest, preview = described(raw)
        handle = number.decode()

        try:
            message = message_from_bytes(raw)
        except ValueError as exc:
            yield Fetched(handle=handle, digest=digest, preview=preview, refusal=str(exc))
        else:
            yield Fetched(handle=handle, digest=digest, preview=preview, message=message)


def mark_read(mailbox: Mailbox, handle: str) -> bool:
    """
    Mark one message read, reporting whether the flag stuck.

    Callers are expected to carry on when it did not: by the time this is called the run is written
    and committed, and refusing to continue would lose that. It is logged because a transient
    failure and a mailbox that never accepts a flag -- quietly spending every slot of every poll on
    the same message -- look identical from here, and only the second one needs a person.
    """
    number = handle.encode()
    try:
        status, _ = mailbox.store(number, "+FLAGS", "\\Seen")
    except (OSError, imaplib.IMAP4.error) as exc:
        # Caught rather than raised, because by here the runs are committed. Letting a dropped
        # connection out would abandon the rest of the marking loop -- every message after this one
        # left unread despite its run existing -- and lose the pass's summary on the way out.
        log.warning("could not mark message %r read (%s); it will be offered again", number, type(exc).__name__)
        return False

    if status != "OK":
        log.warning("could not mark message %r read (%s); it will be offered again", number, status)
        return False
    return True


def _numbers(data: Any) -> list[bytes]:
    """
    The message numbers out of a search response, which arrives as one space-separated line.

    Anything that is not a number is dropped. These go straight back out in the next command, so
    they are the server's input to our command line; imaplib strips control characters before
    sending, which already closes the door, but the door belongs to somebody else's library.
    """
    if not data or not isinstance(data[0], bytes):
        return []
    return [number for number in data[0].split() if number.isdigit()]


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
