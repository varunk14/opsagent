"""
Turning a real email into an IncomingMessage.

Everything here is a boundary test. Anyone who learns the intake address can send anything at all,
so the parser's job is as much refusing as it is reading: a message with no usable id, a body that
is not text, a subject encoded in a charset nobody has heard of, a megabyte of HTML.

No network. The IMAP conversation is faked, because what is under test is the parsing, not
imaplib -- and a test that needs a mailbox is a test nobody runs.
"""

import time
from datetime import UTC, datetime
from email.message import EmailMessage

import pytest

from app.adapters.mailbox import (
    MAX_FETCHED,
    MAX_HEADER_BYTES,
    MAX_RAW_BYTES,
    mark_read,
    message_from_bytes,
    unread_messages,
)
from app.contracts import Channel


def an_email(
    *,
    sender: str = "priya@example.com",
    subject: str | None = "Charged twice for order #4821",
    body: str = "Hi, I think I was charged twice for order #4821.",
    message_id: str | None = "<abc123@example.com>",
    date: str = "Tue, 15 Sep 2026 09:15:00 +0000",
) -> bytes:
    """One plain-text email, as bytes off the wire."""
    message = EmailMessage()
    message["From"] = sender
    if subject is not None:
        message["Subject"] = subject
    if message_id is not None:
        message["Message-ID"] = message_id
    message["Date"] = date
    message.set_content(body)
    return message.as_bytes()


# --- reading one ---------------------------------------------------------------------------------


def test_an_email_becomes_a_message_the_agent_can_read():
    found = message_from_bytes(an_email())

    assert found.channel == Channel.EMAIL
    assert found.sender == "priya@example.com"
    assert found.subject == "Charged twice for order #4821"
    assert "charged twice" in found.body
    assert found.received_at == datetime(2026, 9, 15, 9, 15, tzinfo=UTC)


def test_the_message_id_is_what_makes_intake_idempotent():
    """
    Two deliveries of one email must be one run.

    The Message-ID is the only thing a mail server promises is stable across a redelivery, so it is
    the external_id. Anything else -- a hash of the body, the arrival time -- would make the same
    email a second run every time it was retried.
    """
    assert message_from_bytes(an_email(message_id="<abc123@example.com>")).external_id == "abc123@example.com"


def test_a_display_name_is_not_the_address():
    """'Priya <priya@example.com>' is one sender, and the rate limit counts the address."""
    found = message_from_bytes(an_email(sender="Priya Sharma <priya@example.com>"))

    assert found.sender == "priya@example.com"


def test_an_email_with_no_subject_is_read_without_one():
    assert message_from_bytes(an_email(subject=None)).subject is None


# --- refusing the rest ---------------------------------------------------------------------------


def test_an_email_with_no_message_id_is_refused():
    """
    Without one there is nothing stable to be idempotent on.

    Inventing an id would make a redelivered message a fresh run, which is how one complaint
    becomes two refunds. Better to refuse it and have someone notice.
    """
    with pytest.raises(ValueError, match="Message-ID"):
        message_from_bytes(an_email(message_id=None))


def test_an_email_with_no_address_to_reply_to_is_refused():
    with pytest.raises(ValueError, match="From"):
        message_from_bytes(an_email(sender=""))


def test_a_body_that_is_not_text_is_refused_rather_than_guessed():
    message = EmailMessage()
    message["From"] = "priya@example.com"
    message["Message-ID"] = "<only-an-image@example.com>"
    message["Date"] = "Tue, 15 Sep 2026 09:15:00 +0000"
    message.set_content(b"\x89PNG\r\n\x1a\n", maintype="image", subtype="png")

    with pytest.raises(ValueError, match="no text"):
        message_from_bytes(message.as_bytes())


def test_the_plain_text_part_is_preferred_over_the_html_one():
    """A marketing client sends both. The HTML part is markup we would pay for by the token."""
    message = EmailMessage()
    message["From"] = "priya@example.com"
    message["Message-ID"] = "<multipart@example.com>"
    message["Date"] = "Tue, 15 Sep 2026 09:15:00 +0000"
    message.set_content("I was charged twice.")
    message.add_alternative("<html><body><p>I was charged twice.</p></body></html>", subtype="html")

    found = message_from_bytes(message.as_bytes())

    assert "<html>" not in found.body
    assert "I was charged twice." in found.body


def test_a_body_longer_than_a_body_is_refused_not_cut():
    """The contract refuses over 200,000 characters; a cut message is a misread one."""
    with pytest.raises(ValueError):
        message_from_bytes(an_email(body="x" * 200_001))


def test_a_date_that_cannot_be_read_does_not_become_now():
    """
    A message with an unreadable Date is refused.

    Defaulting to the time we happened to read it would file a three-week-old complaint as new, and
    every latency measured from it would be wrong.
    """
    with pytest.raises(ValueError, match="Date"):
        message_from_bytes(an_email(date="whenever"))


def test_a_body_in_a_charset_nobody_has_is_refused_like_any_other_unreadable_body():
    """
    A charset this machine has no codec for is a refusal, not a crash.

    Left to raise LookupError it would escape the ValueError the mailbox loop catches, and one
    message with an invented charset would end the whole poll -- everyone behind it included.
    """
    raw = (
        b"From: priya@example.com\r\n"
        b"Message-ID: <no-such-charset@example.com>\r\n"
        b"Date: Tue, 15 Sep 2026 09:15:00 +0000\r\n"
        b'Content-Type: text/plain; charset="not-a-real-charset"\r\n'
        b"\r\n"
        b"I was charged twice.\r\n"
    )

    with pytest.raises(ValueError, match="decoded"):
        message_from_bytes(raw)


def test_a_date_with_no_timezone_is_refused():
    """
    A time with no offset is not a time. 09:15 in Mumbai is not 09:15 in Dublin.

    This is refused by the contract rather than here, and it is tested at this level on purpose:
    the refusal only reaches the poll loop because pydantic's ValidationError happens to subclass
    ValueError. That is load-bearing and invisible, so it gets a test of its own.
    """
    with pytest.raises(ValueError):
        message_from_bytes(an_email(date="Tue, 15 Sep 2026 09:15:00"))


def test_a_pile_of_encoded_words_is_refused_before_anything_decodes_it():
    """
    One small email must not be able to stop the poll by being slow rather than malformed.

    Python's header parser is worse than quadratic in the number of RFC 2047 encoded-words: a 664 KB
    Subject takes 13 seconds to read, and 332 KB takes 0.6, on this machine. Nothing else here
    defends against it -- the message is well-formed, so no refusal fires, and `unread_messages`
    catches ValueError, not slowness. The cap is on the headers because that is where the cost is:
    parsing the same message is a millisecond until a header is actually touched.
    """
    word = b"=?utf-8?B?QQ==?="
    raw = (
        b"From: priya@example.com\r\n"
        b"Message-ID: <bomb@example.com>\r\n"
        b"Date: Tue, 15 Sep 2026 09:15:00 +0000\r\n"
        b"Subject: " + b" ".join([word] * 40_000) + b"\r\n"
        b"\r\n"
        b"I was charged twice.\r\n"
    )

    started = time.perf_counter()
    with pytest.raises(ValueError, match="headers"):
        message_from_bytes(raw)
    spent = time.perf_counter() - started

    # Unguarded this call takes about 13 seconds. The bound is loose because it is a slow-CI
    # assertion, not a benchmark: anything under it means the header was never decoded.
    assert spent < 2.0, f"the headers were decoded after all: {spent:.1f}s"


def test_a_message_larger_than_the_cap_is_refused_unread():
    """
    The count cap bounds how many messages a poll takes; nothing bounded how large one could be.

    The cost of this is real and worth stating: an email carrying a large attachment is refused
    whole, and the sentence we wanted goes with it. That is the trade this project keeps making --
    it reads text, the refusal is visible, and the message stays in the mailbox for a person.
    """
    with pytest.raises(ValueError, match="too large"):
        message_from_bytes(b"x" * (MAX_RAW_BYTES + 1))


def test_the_caps_are_small_enough_to_be_caps():
    """
    Pinned because every other test here is written relative to the constants.

    A cap raised to something absurd would leave all of them green while the protection was gone.
    The numbers that matter: at MAX_FETCHED a poll may hold MAX_FETCHED * MAX_RAW_BYTES in memory,
    and the header parser's worst case grows faster than the square of MAX_HEADER_BYTES.
    """
    assert MAX_FETCHED * MAX_RAW_BYTES <= 100_000_000, "one poll could exhaust memory"
    assert MAX_HEADER_BYTES <= 64 * 1024, "the header parser's worst case stops being bounded"


def test_a_subject_in_another_charset_is_decoded_not_mangled():
    message = EmailMessage()
    message["From"] = "priya@example.com"
    message["Message-ID"] = "<charset@example.com>"
    message["Date"] = "Tue, 15 Sep 2026 09:15:00 +0000"
    message["Subject"] = "=?utf-8?B?4KS54KSu4KS+4KSw4KS+IOCkkeCksOCljeCkoeCksA==?="
    message.set_content("where is it")

    assert message_from_bytes(message.as_bytes()).subject == "हमारा ऑर्डर"


# --- the mailbox ---------------------------------------------------------------------------------


class FakeMailbox:
    """imaplib's shape, as much of it as the adapter uses."""

    def __init__(self, messages: dict[bytes, bytes], *, seen: list[bytes] | None = None) -> None:
        self.messages = messages
        self.seen = seen if seen is not None else []
        self.selected: str | None = None
        self.logged_out = False

    def select(self, mailbox: str):
        self.selected = mailbox
        return "OK", [str(len(self.messages)).encode()]

    def search(self, charset, *criteria):
        return "OK", [b" ".join(self.messages)]

    def fetch(self, number: bytes, parts: str):
        return "OK", [(b"", self.messages[number])]

    def store(self, number: bytes, command: str, flags: str):
        self.seen.append(number)
        return "OK", [b""]

    def logout(self):
        self.logged_out = True
        return "BYE", [b""]


def test_every_unread_message_is_offered_with_the_number_that_marks_it():
    """
    Reading does not mark anything read. That is the whole point of the split.

    Whoever consumes these has to write the run down first and mark the message only once that
    commit has landed. So the adapter hands back the number alongside the message and marks nothing
    itself -- it cannot know whether the work behind the message survived.
    """
    mailbox = FakeMailbox(
        {b"1": an_email(message_id="<one@example.com>"), b"2": an_email(message_id="<two@example.com>")}
    )

    found = list(unread_messages(mailbox))

    assert [item.message.external_id for item in found] == ["one@example.com", "two@example.com"]
    assert [item.number for item in found] == [b"1", b"2"]
    assert mailbox.seen == [], "reading must not mark anything read"
    assert mailbox.selected == "INBOX"


def test_a_message_that_cannot_be_read_is_handed_back_as_a_refusal():
    """
    A refusal is an outcome, not an absence.

    Swallowing it here would leave the caller unable to tell a malformed message from a message that
    was never there -- and the refusal is the thing a person needs to see. It carries its number too,
    because whoever records it is the one who gets to mark it read.
    """
    mailbox = FakeMailbox({b"1": b"not an email at all", b"2": an_email(message_id="<good@example.com>")})

    found = list(unread_messages(mailbox))

    assert len(found) == 2
    assert found[0].message is None
    assert "Message-ID" in found[0].refusal
    assert found[0].number == b"1"
    assert found[1].message.external_id == "good@example.com"
    assert found[1].refusal is None


def test_one_unreadable_message_does_not_stop_the_others():
    """A mailbox is not a file: one malformed message must not block every message behind it."""
    mailbox = FakeMailbox({b"1": b"not an email at all", b"2": an_email(message_id="<good@example.com>")})

    readable = [item.message.external_id for item in unread_messages(mailbox) if item.message]

    assert readable == ["good@example.com"]


def test_marking_a_message_read_says_whether_it_worked(caplog):
    """
    A flag that would not stick is worth a line in the log.

    The caller carries on either way -- the run is already written, and intake is idempotent, so the
    repeat next poll is a lookup that writes nothing. But a transient failure and a mailbox that
    never accepts a flag, quietly spending every slot of every poll on the same message, look
    identical from here, and only the second one needs a person.
    """

    class WillNotMarkRead(FakeMailbox):
        def store(self, number: bytes, command: str, flags: str):
            return "NO", [b"over quota"]

    assert mark_read(FakeMailbox({b"1": an_email()}), b"1") is True

    assert mark_read(WillNotMarkRead({b"1": an_email()}), b"1") is False
    assert "NO" in caplog.text


def test_only_so_many_are_taken_from_one_poll():
    """A mailbox with ten thousand unread messages must not become ten thousand runs at once."""
    mailbox = FakeMailbox(
        {str(n).encode(): an_email(message_id=f"<{n}@example.com>") for n in range(MAX_FETCHED + 10)}
    )

    assert len(list(unread_messages(mailbox))) == MAX_FETCHED
