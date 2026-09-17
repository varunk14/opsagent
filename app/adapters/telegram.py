"""
Reading messages from a Telegram bot.

Same contract as every other adapter: it produces IncomingMessage objects and knows nothing about
runs, the database, or the agent. Two things are genuinely different from a mailbox, and both shape
the code more than the parsing does.

**The cursor.** Telegram does not remember which messages we dealt with; it remembers one number.
Asking for updates from offset N is what tells it to throw away everything below N, irrevocably and
for everyone. There is no equivalent of an unread flag to put back. So the confirmation is coarse,
it happens once at the end of a pass, and it must not happen until the work is committed -- which is
why reading and confirming are separate calls here, as they are for the mailbox.

**The token.** It travels in the URL path rather than a header:
`api.telegram.org/bot<TOKEN>/getUpdates`. The URL *is* the credential. urllib puts the URL in its
exceptions, which means an unhandled error prints the bot's credentials into a log file, a
traceback, and whatever ships those elsewhere. Every error path here is written to prevent that, and
it is the reason this module builds its own exceptions rather than letting urllib's escape.
"""

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import IO, Any, Protocol

from app.adapters.inbox import Fetched, Unread, described
from app.contracts import Channel, IncomingMessage

log = logging.getLogger(__name__)

TOKEN_VAR = "OPSAGENT_TELEGRAM_TOKEN"
ENDPOINT = "https://api.telegram.org"

# One pass is one batch of work. Without a cap a bot left alone over a weekend becomes a thousand
# queued runs in a single pass, and the rate limits that protect the agent are all per sender.
MAX_UPDATES = 50

# The Bot API's own maximum message is 4096 characters, so this is not about ordinary traffic. It
# bounds what a compromised or impersonated endpoint can make us hold in memory before any of the
# contract's limits get a chance to apply.
MAX_RESPONSE_BYTES = 8_000_000
TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class BotSettings:
    """
    The bot's token, and nothing else. It is the whole credential.

    `repr` is written out because the generated one prints every field and these get held in frames.
    A token is `<bot_id>:<secret>`; the id half is the bot's account number, which anyone who has
    ever messaged it already knows, so it is kept. That way a log line says *which* bot without
    saying how to be it.
    """

    token: str

    def __repr__(self) -> str:
        return f"BotSettings(bot={self.token.split(':')[0]!r}, token=...)"


def settings_from_env(environ: Mapping[str, str]) -> BotSettings:
    """The bot token, refused by name if it is not there. No default: there is no default bot."""
    token = environ.get(TOKEN_VAR, "").strip()
    if not token:
        raise ValueError(f"the bot needs {TOKEN_VAR} set")
    return BotSettings(token=token)


class Bot(Protocol):
    """The part of the Bot API this adapter uses, so the tests can stand in for Telegram."""

    def get_updates(self, *, offset: int | None, limit: int) -> list[dict]: ...


class TelegramBot:
    """The real thing. Talks to the Bot API over HTTPS."""

    def __init__(self, settings: BotSettings, endpoint: str = ENDPOINT) -> None:
        self.settings = settings
        self.endpoint = endpoint

    def get_updates(self, *, offset: int | None, limit: int) -> list[dict]:
        """
        One call to getUpdates. Short polling: asks what is there and returns.

        Long polling would hold a connection open for a minute per call, which reads as tidier and
        is worse here -- the pass owns a database transaction, and a pass that spends most of its
        life waiting on a socket is a transaction that does too.
        """
        query = f"limit={limit}" + (f"&offset={offset}" if offset is not None else "")
        return updates_from_payload(self._get(f"getUpdates?{query}"))

    def _get(self, path: str) -> bytes:
        """
        Fetch one Bot API path, with the token kept out of whatever goes wrong.

        Nothing from urllib's exception is allowed out: its message and its `url` attribute both
        carry the full request line, and the full request line contains the token. Only the type
        name survives, and the error is raised after the handler so that `__context__` does not keep
        a reference to the original either -- suppressing a chain when a traceback is printed is not
        the same as breaking it, and anything that walks the chain would read the token straight
        out.
        """
        request = urllib.request.Request(f"{self.endpoint}/bot{self.settings.token}/{path}")

        failed_with: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                body = read_capped(response, MAX_RESPONSE_BYTES)
        except ValueError as exc:
            # The size refusal carries no token in its message, but left to propagate from here its
            # traceback would keep this frame -- and `request`, whose URL is the token -- alive for
            # anything that reads frame locals. So it is rebuilt outside, like every other failure.
            failed_with = str(exc)
        except (OSError, TimeoutError) as exc:  # HTTPError and URLError are both OSErrors
            failed_with = type(exc).__name__
        else:
            return body

        del request
        raise ValueError(f"could not reach the Telegram API for {self.settings!r}: {failed_with}")


def _mapping(value: Any) -> dict | None:
    """The value if it is an object, otherwise None. Every caller turns that None into a refusal."""
    return value if isinstance(value, dict) else None


def _words(value: Any) -> str | None:
    """The value if it is text, otherwise None -- a photo, a sticker, a voice note."""
    return value if isinstance(value, str) else None


def _whole(value: Any) -> int | None:
    """
    The value if it is a whole number, otherwise None.

    `bool` is excluded on purpose: True is an int in Python, so a chat id of `true` would otherwise
    read as chat 1 and quietly file someone's message against a real conversation.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def read_capped(stream: IO[bytes], limit: int) -> bytes:
    """
    Read a reply, refusing one larger than `limit` bytes.

    The model client has a function of the same shape, and borrowing it was the first thing tried.
    It raises ModelUnavailable -- an error about the language model, for a reply from Telegram. That
    would pass straight through this module's handlers, end the pass, and send whoever read the log
    looking at Ollama for a fault that was never there. A refusal has to be the kind of thing its
    own callers already expect to catch.
    """
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"the Telegram API's reply was too large to read: over {limit} bytes")
    return body


def updates_from_payload(raw: bytes) -> list[dict]:
    """
    The updates out of one getUpdates reply.

    `{"ok": false}` is not an exception anywhere in the stack -- a revoked token comes back as a
    perfectly ordinary body, and a proxy or captive portal answers with HTML and a 200. Both would
    otherwise be read as "no updates", and a bot that had been switched off would look exactly like
    a bot nobody had messaged.
    """
    failed: str | None = None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        failed = f"the Telegram API sent something that is not JSON: {exc}"
    except RecursionError:
        # The JSON parser recurses, so a reply nested a few thousand levels deep raises this rather
        # than anything a pass catches. It took 40 KB -- far under the size cap -- to end a pass,
        # and the same reply would have ended every pass after it.
        failed = "the Telegram API sent a reply nested too deeply to read"
    if failed is not None:
        raise ValueError(failed)

    reply = _mapping(payload)
    if reply is None:
        raise ValueError("the Telegram API sent something that is not a reply")
    if not reply.get("ok"):
        # The description is Telegram's own, and it is the half of the exchange the server wrote --
        # it never contains the token.
        raise ValueError(f"the Telegram API refused the request: {reply.get('description')}")

    result = reply.get("result")
    if result is None or not isinstance(result, list):
        raise ValueError("the Telegram API sent a reply with no updates in it")
    return [update for update in result if isinstance(update, dict)]


def message_from_update(update: dict) -> IncomingMessage:
    """
    Turn one update into a message the agent can read.

    Raises ValueError, naming the part at fault, for anything it cannot read honestly -- the same
    bargain the mailbox makes. A guess here costs the same as a guess there: the wrong identity
    makes one complaint two refunds, the wrong time files an old message as new.
    """
    message = _mapping(update.get("message"))
    if message is None:
        raise ValueError("the update carries no message")

    text = _words(message.get("text"))
    if text is None:
        raise ValueError("the message has no text to read")

    return IncomingMessage(
        channel=Channel.TELEGRAM,
        external_id=_identifier(message),
        sender=_sender(message),
        subject=None,  # Telegram has no subject line, and making one from the body would invent it
        body=text,
        received_at=_sent_at(message),
    )


def _identifier(message: dict) -> str:
    """
    The chat and the message together, because a message_id is unique per chat and not beyond it.

    Chat 4821 and chat 9999 both have a message #11. Keyed on the message_id alone, the second
    customer's complaint would be recognised as a redelivery of the first one's and silently get the
    first one's answer. The update_id is deliberately not used: it is the cursor, Telegram may hand
    the same message a different one, and keying on it turns one complaint into two refunds.
    """
    chat = _mapping(message.get("chat")) or {}
    chat_id = _whole(chat.get("id"))
    message_id = _whole(message.get("message_id"))

    if chat_id is None or message_id is None:
        raise ValueError("the message has no chat and message id to be idempotent on")
    return f"{chat_id}:{message_id}"


def _sender(message: dict) -> str:
    """
    The account id, never the username.

    A username is chosen by the person and can be changed whenever they like; the numeric id cannot.
    The per-sender rate limit counts this, so counting usernames would be a limit anyone could reset
    by editing their profile.
    """
    sender = _mapping(message.get("from")) or {}
    account = _whole(sender.get("id"))

    if account is None:
        raise ValueError("the message has no sender to reply to")
    return str(account)


def _sent_at(message: dict) -> datetime:
    """When they sent it, never when we happened to ask."""
    sent = _whole(message.get("date"))
    if sent is None:
        raise ValueError("the message has no date that can be read")

    try:
        return datetime.fromtimestamp(sent, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"the message's date cannot be read: {exc}") from None


def unread_updates(bot: Bot) -> Unread:
    """
    Everything the bot has waiting, up to MAX_UPDATES. Confirms nothing.

    One more than the cap is asked for and the extra is never handled. The Bot API does not say how
    many are queued -- it answers with whatever fits the limit -- so asking for exactly the cap
    makes a full pass and a drained channel look the same, and the pass would have to guess. The
    guess that says "drained" stops the next pass while a backlog waits. One extra update turns that
    guess into an observation.
    """
    updates = bot.get_updates(offset=None, limit=MAX_UPDATES + 1)
    return Unread(waiting=len(updates), messages=_read(updates[:MAX_UPDATES]))


def _read(updates: list[dict]) -> Iterator[Fetched]:
    """The reading half of `unread_updates`, separated so the count is known before it starts."""
    for update in updates:
        handle = str(update.get("update_id", ""))
        digest, preview = described(json.dumps(update, sort_keys=True).encode())

        if "message" not in update:
            # Not a customer writing in: a group membership change, an edit, a poll answer. Counted
            # and confirmed so the cursor moves past it, and not recorded, because a dead letter for
            # every one of these would bury the refusals a person actually needs to read.
            yield Fetched(handle=handle, digest=digest, preview=preview, ignored=_kind(update))
            continue

        try:
            message = message_from_update(update)
        except ValueError as exc:
            yield Fetched(handle=handle, digest=digest, preview=preview, refusal=str(exc))
        else:
            yield Fetched(handle=handle, digest=digest, preview=preview, message=message)


def _kind(update: dict) -> str:
    """What sort of update this was, for the count and the log. Never its contents."""
    named = [key for key in update if key != "update_id"]
    return named[0] if named else "an update with nothing in it"


def confirm_updates(bot: Bot, handled: list[str]) -> None:
    """
    Move the cursor past everything this pass committed.

    Telegram confirms by being asked for the next one: offset N discards every update below N. So it
    is one number rather than one call per message, and it must be the highest handled plus one --
    less re-offers work already done, more discards work never seen.

    Called only after the transaction has committed. This is the irrevocable step: there is no
    unread flag to put back, so a cursor moved too early loses those messages for good.

    An empty pass moves nothing. Sending some default offset for a pass that handled nothing would
    discard whatever happened to sit below it.
    """
    numbers = [int(handle) for handle in handled if handle.isdigit()]
    if not numbers:
        return

    try:
        bot.get_updates(offset=max(numbers) + 1, limit=1)
    except ValueError as exc:
        # The runs are committed by now. Refusing to continue would lose the pass's summary, and the
        # cursor simply stays where it is -- the next pass sees the same updates again and intake
        # recognises them.
        log.warning("could not move the cursor past %d: %s", max(numbers), exc)
