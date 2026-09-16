"""
Turning a real email into an IncomingMessage.

Everything here is a boundary test. Anyone who learns the intake address can send anything at all,
so the parser's job is as much refusing as it is reading: a message with no usable id, a body that
is not text, a subject encoded in a charset nobody has heard of, a megabyte of HTML.

No network. The IMAP conversation is faked, because what is under test is the parsing, not
imaplib -- and a test that needs a mailbox is a test nobody runs.
"""

from datetime import UTC, datetime
from email.message import EmailMessage

import pytest
from app.adapters.mailbox import MAX_FETCHED, message_from_bytes, unread_messages

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


def test_every_unread_message_is_read_and_then_marked_read():
    """Marked only after it is parsed, so a message the parser refuses is still there to look at."""
    mailbox = FakeMailbox(
        {b"1": an_email(message_id="<one@example.com>"), b"2": an_email(message_id="<two@example.com>")}
    )

    found = list(unread_messages(mailbox))

    assert [message.external_id for message in found] == ["one@example.com", "two@example.com"]
    assert mailbox.seen == [b"1", b"2"]
    assert mailbox.selected == "INBOX"


def test_one_unreadable_message_does_not_stop_the_others():
    """
    A mailbox is not a file: one malformed message must not block every message behind it.

    It is left unread deliberately, so it is still in the mailbox to be looked at rather than
    silently consumed -- the same reasoning as the fixture adapter refusing a bad line.
    """
    mailbox = FakeMailbox({b"1": b"not an email at all", b"2": an_email(message_id="<good@example.com>")})

    found = list(unread_messages(mailbox))

    assert [message.external_id for message in found] == ["good@example.com"]
    assert mailbox.seen == [b"2"], "the unreadable one is left unread"


def test_only_so_many_are_taken_from_one_poll():
    """A mailbox with ten thousand unread messages must not become ten thousand runs at once."""
    mailbox = FakeMailbox(
        {str(n).encode(): an_email(message_id=f"<{n}@example.com>") for n in range(MAX_FETCHED + 10)}
    )

    assert len(list(unread_messages(mailbox))) == MAX_FETCHED
