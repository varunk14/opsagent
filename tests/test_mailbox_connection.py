"""
Getting as far as the mailbox: settings, and opening the connection.

No network here either. What is worth testing is not that imaplib can log in -- it can -- but the
things around it that are easy to get wrong and expensive to get wrong: refusing to start with half
a configuration, never putting the password anywhere it can be read back, and never opening a
connection that is not encrypted.

The password is the whole point of the care. It is an app password for a real mailbox, it is typed
once into a .env file that is never committed, and everything here exists so that it does not then
turn up in a log line, a traceback, or a repr in someone's terminal.
"""

import ssl

import pytest

from app.adapters.mailbox import (
    HOST_VAR,
    PASSWORD_VAR,
    PORT_VAR,
    USER_VAR,
    MailboxSettings,
    open_mailbox,
    settings_from_env,
)

SECRET = "abcd efgh ijkl mnop"

COMPLETE = {
    HOST_VAR: "imap.example.com",
    USER_VAR: "support@example.com",
    PASSWORD_VAR: SECRET,
}


class FakeIMAP:
    """imaplib.IMAP4_SSL's shape as far as opening a connection uses it."""

    def __init__(self, host: str, port: int, ssl_context=None) -> None:
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.logged_in_as: tuple[str, str] | None = None

    def login(self, user: str, password: str):
        self.logged_in_as = (user, password)
        return "OK", [b"logged in"]


# --- reading the settings ---------------------------------------------------------------------


def test_a_complete_environment_is_read():
    settings = settings_from_env(COMPLETE)

    assert settings.host == "imap.example.com"
    assert settings.user == "support@example.com"
    assert settings.password == SECRET
    assert settings.port == 993


@pytest.mark.parametrize("missing", [HOST_VAR, USER_VAR, PASSWORD_VAR])
def test_half_a_configuration_is_refused_by_name(missing):
    """
    Named, because the alternative is someone reading an imaplib traceback at the far end.

    A poller that starts with no password and fails on every pass looks exactly like a mail server
    that is down. Refusing at the point the setting is missing is the difference between a minute
    and an afternoon.
    """
    environment = {name: value for name, value in COMPLETE.items() if name != missing}

    with pytest.raises(ValueError, match=missing):
        settings_from_env(environment)


def test_a_setting_that_is_only_whitespace_counts_as_missing():
    """A variable set to "" in a .env file is a variable someone meant to fill in."""
    with pytest.raises(ValueError, match=PASSWORD_VAR):
        settings_from_env({**COMPLETE, PASSWORD_VAR: "   "})


def test_a_port_that_is_not_a_port_is_refused():
    with pytest.raises(ValueError, match=PORT_VAR):
        settings_from_env({**COMPLETE, PORT_VAR: "not a number"})


def test_the_port_can_be_set_for_a_server_that_wants_another_one():
    assert settings_from_env({**COMPLETE, PORT_VAR: "1993"}).port == 1993


# --- keeping the password out of everything else -----------------------------------------------


def test_the_password_is_not_in_the_repr():
    """
    Where a password most often escapes: a dataclass printed in a traceback or a debug line.

    The default repr would print every field, so this is not theoretical -- any unhandled error
    holding these settings would put an app password on someone's screen and into a log file.
    """
    settings = settings_from_env(COMPLETE)

    assert SECRET not in repr(settings)
    assert SECRET not in str(settings)
    assert "imap.example.com" in repr(settings), "the rest is still worth being able to read"


def test_the_password_is_not_in_the_error_when_the_settings_are_wrong():
    with pytest.raises(ValueError) as refused:
        settings_from_env({**COMPLETE, PORT_VAR: "not a number"})

    assert SECRET not in str(refused.value)


def test_the_password_is_not_in_the_error_when_logging_in_fails():
    """A rejected login is the moment a naive implementation echoes what it tried."""

    class RefusesToLogIn(FakeIMAP):
        def login(self, user: str, password: str):
            raise OSError(f"authentication failed for {user}")

    with pytest.raises(ValueError) as refused:
        open_mailbox(settings_from_env(COMPLETE), connect=RefusesToLogIn)

    assert SECRET not in str(refused.value)
    assert "imap.example.com" in str(refused.value), "say which mailbox, just not the secret"


# --- opening it -------------------------------------------------------------------------------


def test_the_connection_goes_to_the_host_and_port_configured():
    opened = open_mailbox(settings_from_env(COMPLETE), connect=FakeIMAP)

    assert opened.host == "imap.example.com"
    assert opened.port == 993
    assert opened.logged_in_as == ("support@example.com", SECRET)


def test_the_connection_checks_who_it_is_talking_to():
    """
    Encrypted is not the same as authenticated, and imaplib's default is the first without the
    second.

    `imaplib.IMAP4_SSL(host, port)` with no ssl_context falls back to `ssl._create_stdlib_context()`
    -- which in CPython is literally an alias for `_create_unverified_context`: verify_mode
    CERT_NONE, check_hostname False. The bytes are encrypted and the certificate is nobody's. Anyone
    on the path can present a self-signed certificate, be believed, and take the app password.

    So the context is passed explicitly, and this asserts what it is rather than which class was
    used. An earlier version of this test checked `open_mailbox`'s default was IMAP4_SSL and passed
    happily while the connection verified nothing.
    """
    opened = open_mailbox(settings_from_env(COMPLETE), connect=FakeIMAP)

    assert opened.ssl_context is not None, "a context must be passed, not left to imaplib's default"
    assert opened.ssl_context.verify_mode == ssl.CERT_REQUIRED
    assert opened.ssl_context.check_hostname is True


def test_imaplibs_own_default_is_the_unverified_one():
    """
    The reason the test above exists, pinned against the standard library itself.

    If a future Python makes the default context a verifying one, this fails and the explicit
    context can be reconsidered. Until then it is documenting why we do not rely on it.
    """
    assert ssl._create_stdlib_context().verify_mode == ssl.CERT_NONE
    assert ssl._create_stdlib_context().check_hostname is False


def test_nothing_of_the_failure_is_kept_where_the_password_could_be():
    """
    `raise ... from None` suppresses the chained exception when a traceback is printed, but does
    not clear it: `__context__` still holds the original.

    That matters because imaplib builds its login error out of the server's own reply text, and the
    server is the one party that has already been sent the password. A hostile server can echo it
    back, and anything that walks `__context__` -- a crash reporter, a debugger, a future error
    handler -- would read it out. Cheaper to drop the link than to rely on everyone downstream
    respecting a suppression flag.
    """

    class EchoesThePasswordBack(FakeIMAP):
        def login(self, user: str, password: str):
            raise OSError(f"NO authentication failed for {password}")

    with pytest.raises(ValueError) as refused:
        open_mailbox(settings_from_env(COMPLETE), connect=EchoesThePasswordBack)

    assert refused.value.__context__ is None
    assert refused.value.__cause__ is None
    assert SECRET not in str(refused.value)


def test_settings_carry_no_default_host():
    """A default would be someone else's mail server, quietly tried with a real password."""
    with pytest.raises(TypeError):
        MailboxSettings(user="a", password="b")  # type: ignore[call-arg]
