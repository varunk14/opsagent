"""
Turning a Telegram update into an IncomingMessage.

Same shape of problem as the mailbox, with two differences that matter.

The first is the cursor. A mailbox remembers per message; Telegram remembers one number, and asking
for updates past it is what throws the earlier ones away irrevocably. So the confirmation is
coarser and the ordering rule matters more, not less.

The second is the token. It travels in the URL path, not a header, so every string that might
contain the request -- an error, a log line, a traceback -- is somewhere the bot's credentials can
escape. Several tests here exist only to hold that line.
"""

import io
import json

import pytest

from app.adapters.inbox import PREVIEW_CHARS
from app.adapters.telegram import (
    MAX_UPDATES,
    TOKEN_VAR,
    BotSettings,
    confirm_updates,
    message_from_update,
    settings_from_env,
    unread_updates,
)
from app.contracts import Channel

SECRET = "8012345678:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"


def an_update(
    *,
    update_id: int = 700,
    message_id: int = 11,
    chat_id: int = 4821,
    user_id: int | None = 99001,
    username: str | None = "priya_s",
    text: str | None = "I was charged twice for order #4821.",
    date: int | None = 1_789_000_500,
) -> dict:
    """One ordinary text message, shaped the way the Bot API sends it."""
    message: dict = {"message_id": message_id, "chat": {"id": chat_id, "type": "private"}}
    if user_id is not None:
        message["from"] = {"id": user_id, "is_bot": False, "username": username}
    if text is not None:
        message["text"] = text
    if date is not None:
        message["date"] = date
    return {"update_id": update_id, "message": message}


class FakeBot:
    """The Bot API as far as this adapter uses it: one call that lists, one that confirms."""

    def __init__(self, updates: list[dict]) -> None:
        self.updates = updates
        self.asked: list[tuple[int | None, int]] = []

    def get_updates(self, *, offset: int | None, limit: int) -> list[dict]:
        self.asked.append((offset, limit))
        if offset is None:
            return self.updates[:limit]
        return [u for u in self.updates if u["update_id"] >= offset][:limit]


# --- reading one ------------------------------------------------------------------------------


def test_a_telegram_message_becomes_a_message_the_agent_can_read():
    found = message_from_update(an_update())

    assert found.channel == Channel.TELEGRAM
    assert "charged twice" in found.body
    assert found.received_at.tzinfo is not None
    assert found.received_at.year == 2026


def test_the_chat_and_message_together_are_what_makes_intake_idempotent():
    """
    A Telegram message_id is unique per chat, not globally -- chat 1 and chat 2 both have a #11.

    So the pair is the external_id. The message_id alone would make two different customers'
    messages the same run, and the second person would silently get the first one's answer.
    """
    assert message_from_update(an_update(chat_id=4821, message_id=11)).external_id == "4821:11"
    assert message_from_update(an_update(chat_id=9999, message_id=11)).external_id == "9999:11"


def test_the_update_id_is_not_the_identity_of_the_message():
    """
    It is the cursor, and it is not the message.

    Telegram will hand the same message a different update_id in some redelivery cases, and keying
    on it would turn one complaint into two runs and two refunds.
    """
    first = message_from_update(an_update(update_id=700))
    again = message_from_update(an_update(update_id=701))

    assert first.external_id == again.external_id


def test_the_sender_is_the_account_id_not_the_username():
    """
    Usernames are chosen by the user and can be changed at will; the numeric id cannot.

    The per-sender rate limit counts this, so a username would be a limit anyone could reset by
    editing their profile.
    """
    found = message_from_update(an_update(user_id=99001, username="priya_s"))

    assert found.sender == "99001"
    assert "priya_s" not in found.sender


def test_a_telegram_message_has_no_subject():
    assert message_from_update(an_update()).subject is None


# --- refusing the rest ------------------------------------------------------------------------


def test_a_message_with_no_text_is_refused_rather_than_guessed():
    """A photo, a sticker, a voice note. There is no sentence to act on."""
    update = an_update(text=None)
    update["message"]["photo"] = [{"file_id": "abc", "width": 90, "height": 90}]

    with pytest.raises(ValueError, match="no text"):
        message_from_update(update)


def test_a_message_from_nobody_is_refused():
    with pytest.raises(ValueError, match="sender"):
        message_from_update(an_update(user_id=None))


def test_a_message_with_no_date_is_refused():
    with pytest.raises(ValueError, match="date"):
        message_from_update(an_update(date=None))


def test_a_date_that_is_not_a_number_is_refused():
    update = an_update()
    update["message"]["date"] = "yesterday"

    with pytest.raises(ValueError, match="date"):
        message_from_update(update)


def test_an_empty_message_is_refused():
    """An empty body is not a request, and the contract refuses a blank one anyway."""
    with pytest.raises(ValueError):
        message_from_update(an_update(text="   "))


# --- what is not a message at all ---------------------------------------------------------------


def test_an_update_that_carries_no_message_is_passed_over_not_recorded():
    """
    Someone adding the bot to a group, an edit, a poll answer. These are not customers writing in.

    Recorded as refusals they would fill the dead letters with noise nobody can act on, and the
    genuine refusals -- the ones a person needs to see -- would be lost among them. Passed over
    silently they would be offered again forever, because the cursor never moves past them. So they
    are confirmed and counted, and that is all.
    """
    bot = FakeBot([{"update_id": 700, "my_chat_member": {"chat": {"id": 1}}}, an_update(update_id=701)])

    found = list(unread_updates(bot).messages)

    assert [item.handle for item in found] == ["700", "701"]
    assert found[0].message is None and found[0].refusal is None
    assert found[0].ignored is not None
    assert found[1].message is not None


# --- the cursor -------------------------------------------------------------------------------


def test_every_update_is_offered_with_the_handle_that_confirms_it():
    bot = FakeBot([an_update(update_id=700, message_id=11), an_update(update_id=701, message_id=12)])

    found = list(unread_updates(bot).messages)

    assert [item.handle for item in found] == ["700", "701"]
    assert [item.message.external_id for item in found] == ["4821:11", "4821:12"]
    assert bot.asked == [(None, MAX_UPDATES + 1)], "nothing is confirmed by reading"


def test_confirming_moves_the_cursor_past_the_highest_one_handled():
    """
    Telegram confirms by being asked for the next one: offset N throws away everything below N.

    So the confirmation is one number, not one call per message, and it must be the highest handled
    plus one. Anything less re-offers work already done; anything more discards work never seen.
    """
    bot = FakeBot([an_update(update_id=700), an_update(update_id=701)])

    confirm_updates(bot, ["700", "701"])

    assert bot.asked[-1] == (702, 1)


def test_confirming_nothing_asks_for_nothing():
    """An empty pass must not move the cursor, least of all to some default."""
    bot = FakeBot([an_update(update_id=700)])

    confirm_updates(bot, [])

    assert bot.asked == []


def test_the_highest_handled_decides_even_if_they_arrive_out_of_order():
    bot = FakeBot([an_update(update_id=700)])

    confirm_updates(bot, ["701", "700", "699"])

    assert bot.asked[-1] == (702, 1)


def test_only_so_many_are_taken_from_one_poll():
    """
    One more is asked for than will be taken, and the extra one is never handled.

    The Bot API does not say how many are queued; it only answers with what fits the limit. Asking
    for exactly the cap makes a full pass and a drained channel indistinguishable, so the pass would
    have to guess -- and the guess that says "drained" stops the next pass while a backlog waits.
    The peek costs one update and turns the guess into an observation. The mailbox pass gets the
    same answer by being able to see the whole list; the file pass does it by reading one line past
    its limit. Same problem, three shapes.
    """
    bot = FakeBot([an_update(update_id=700 + n, message_id=n) for n in range(MAX_UPDATES + 10)])

    unread = unread_updates(bot)

    assert len(list(unread.messages)) == MAX_UPDATES, "the extra one is looked at, never handled"
    assert unread.waiting > MAX_UPDATES


def test_a_channel_with_nothing_left_says_so():
    bot = FakeBot([an_update(update_id=700 + n, message_id=n) for n in range(MAX_UPDATES)])

    unread = unread_updates(bot)

    assert len(list(unread.messages)) == MAX_UPDATES
    assert unread.waiting == MAX_UPDATES, "a full pass is not the same as a queue behind it"


def test_a_malformed_update_does_not_stop_the_ones_behind_it():
    bot = FakeBot([an_update(update_id=700, text=None), an_update(update_id=701)])

    found = list(unread_updates(bot).messages)

    assert len(found) == 2
    assert found[0].refusal is not None
    assert found[1].message is not None


def test_a_refusal_keeps_enough_of_the_update_to_recognise_it():
    bot = FakeBot([an_update(update_id=700, text=None)])

    found = list(unread_updates(bot).messages)

    assert "700" in found[0].preview
    assert len(found[0].preview) <= PREVIEW_CHARS
    assert found[0].digest


# --- the token --------------------------------------------------------------------------------


def test_the_token_is_read_from_the_environment():
    assert settings_from_env({TOKEN_VAR: SECRET}).token == SECRET


def test_a_missing_token_is_refused_by_name():
    with pytest.raises(ValueError, match=TOKEN_VAR):
        settings_from_env({})


def test_a_blank_token_counts_as_missing():
    with pytest.raises(ValueError, match=TOKEN_VAR):
        settings_from_env({TOKEN_VAR: "   "})


def test_the_token_is_not_in_the_repr():
    settings = settings_from_env({TOKEN_VAR: SECRET})

    assert SECRET not in repr(settings)
    assert SECRET not in str(settings)


def test_the_repr_keeps_the_part_that_is_public_anyway():
    """
    A bot token is <bot_id>:<secret>. The id half is not a secret -- it is the bot's account number
    and anyone who has ever messaged it can see it -- and keeping it makes a log line identify
    which bot without identifying how to be it.
    """
    settings = settings_from_env({TOKEN_VAR: SECRET})

    assert "8012345678" in repr(settings)
    assert "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in repr(settings)


def test_the_token_is_not_in_the_url_that_errors_quote():
    """
    The sharp edge of this channel. The token is in the path, not a header, so the URL *is* the
    credential -- and urllib puts the URL in its exceptions by default.

    An unhandled HTTPError would otherwise print the whole thing into a log, a traceback, and
    whatever ships those somewhere else.
    """
    from app.adapters.telegram import TelegramBot

    bot = TelegramBot(settings_from_env({TOKEN_VAR: SECRET}), endpoint="http://127.0.0.1:1")

    with pytest.raises(ValueError) as refused:
        bot.get_updates(offset=None, limit=1)

    assert SECRET not in str(refused.value)
    assert "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in str(refused.value)
    assert refused.value.__context__ is None, "the chained error carries the URL too"


def test_a_reply_too_large_to_read_is_refused_as_this_channels_problem():
    """
    The cap has to raise something this module's own callers expect.

    Borrowed from the model client, the refusal would arrive as ModelUnavailable -- an error about
    the language model, for a reply from Telegram. It would pass straight through the handler here,
    end the pass, and send whoever read the log looking at Ollama.
    """
    from app.adapters.telegram import MAX_RESPONSE_BYTES, read_capped

    with pytest.raises(ValueError, match="too large"):
        read_capped(io.BytesIO(b"x" * (MAX_RESPONSE_BYTES + 1)), MAX_RESPONSE_BYTES)

    assert read_capped(io.BytesIO(b"{}"), MAX_RESPONSE_BYTES) == b"{}"


def test_a_reply_nested_too_deep_to_parse_is_refused_not_a_crash():
    """
    Found by a security review, and the same shape of hole as the header bomb on the mailbox.

    Python's JSON parser recurses, so a reply nested a few thousand levels deep raises
    RecursionError. That is not a ValueError, so nothing in the pass catches it; the pass dies
    before the cursor moves, the same reply is offered again, and the channel stalls for good. It
    took 40 KB -- a two-hundredth of the size cap -- which is why the cap alone does not cover it.
    """
    from app.adapters.telegram import updates_from_payload

    deep = b'{"ok":true,"result":[{"update_id":1,"message":' + b"[" * 20_000 + b"]" * 20_000 + b"}]}"

    with pytest.raises(ValueError, match="nested"):
        updates_from_payload(deep)


def test_a_reply_too_large_leaves_no_trace_of_the_request_behind(monkeypatch):
    """
    Every failure out of a request is rebuilt outside the handler, including this one.

    The refusal for an oversized reply used to leave the request function directly, so its traceback
    frame still held the request -- whose full URL is the token. Nothing here reads frame locals, but
    crash reporters and post-mortem debuggers do, and "every error path" should mean every one.
    """
    from app.adapters import telegram
    from app.adapters.telegram import TelegramBot

    class Huge:
        def read(self, n):
            return b"x" * n

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(telegram.urllib.request, "urlopen", lambda *a, **k: Huge())
    bot = TelegramBot(settings_from_env({TOKEN_VAR: SECRET}))

    with pytest.raises(ValueError) as refused:
        bot.get_updates(offset=None, limit=1)

    assert refused.value.__context__ is None
    frames = []
    tb = refused.value.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    for frame in frames:
        for value in frame.f_locals.values():
            assert SECRET not in repr(getattr(value, "full_url", "")), f"token held in {frame.f_code.co_name}"


def test_the_bot_api_saying_no_is_not_a_crash():
    """`{"ok": false}` is how the Bot API reports a revoked token; it arrives as a normal reply."""
    from app.adapters.telegram import updates_from_payload

    with pytest.raises(ValueError, match="refused"):
        updates_from_payload(json.dumps({"ok": False, "description": "Unauthorized"}).encode())


def test_a_reply_that_is_not_what_was_asked_for_is_refused():
    from app.adapters.telegram import updates_from_payload

    with pytest.raises(ValueError):
        updates_from_payload(b"<html>502 Bad Gateway</html>")


def test_settings_carry_no_default_token():
    with pytest.raises(TypeError):
        BotSettings()  # type: ignore[call-arg]
