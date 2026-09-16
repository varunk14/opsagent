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

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
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


def test_the_default_connection_is_an_encrypted_one():
    """
    Pinned because the failure is silent.

    Plain IMAP4 would work against most servers and send the app password over the network in the
    clear, and nothing in the behaviour of the poller would look any different. There is
    deliberately no setting that turns this off.
    """
    import imaplib

    from app.adapters.mailbox import open_mailbox as opener

    assert opener.__defaults__ == (imaplib.IMAP4_SSL,)
    assert issubclass(imaplib.IMAP4_SSL, imaplib.IMAP4)


def test_settings_carry_no_default_host():
    """A default would be someone else's mail server, quietly tried with a real password."""
    with pytest.raises(TypeError):
        MailboxSettings(user="a", password="b")  # type: ignore[call-arg]
