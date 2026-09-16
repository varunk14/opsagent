"""
The vocabulary every adapter speaks, and nothing else.

An adapter's job is to turn whatever a channel offers into IncomingMessage objects. What it cannot
do alone is decide when the channel is allowed to forget an item, because that depends on whether
the run survived -- which the adapter cannot see.

So reading and confirming are two steps, and these are the types that carry the gap between them.
The shapes live here rather than in one adapter because the rule they exist to enforce is the same
for all of them, and a rule written down twice is a rule that will eventually only be true once.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256

from app.contracts import IncomingMessage

# How much of an item we could not read is kept so a person can recognise it. Enough to see a
# subject line and the top of a body; not so much that a refusal costs what the message would have.
PREVIEW_CHARS = 2_000


@dataclass(frozen=True)
class Fetched:
    """
    One item a channel offered, with the handle that confirms it.

    Exactly one of `message`, `refusal` and `ignored` is set, and the three are genuinely different
    things rather than degrees of the same one:

      - `message` read, and it becomes a run
      - `refusal` someone wrote in and we could not read it. A person needs to see this, so it is
        recorded in dead_letters
      - `ignored` not a customer writing in at all -- somebody added the bot to a group, a poll was
        answered. Recording these would bury the refusals that matter under noise nobody can act
        on; passing over them silently would leave them offered forever, because the channel's
        cursor never moves past what is never handled. So they are confirmed, counted, and dropped

    A refusal is carried rather than dropped because it is the outcome a person needs. The caller
    cannot confirm anything without the handle, which is the point of handing that back too.

    `handle` is the channel's own name for the item, opaque to everyone but the adapter that issued
    it: an IMAP message number, a Telegram update id. Whoever polls collects the handles of the work
    it committed and gives them back, and the adapter decides what confirming them means.

    `digest` and `preview` describe the bytes rather than the message, which is what makes them
    usable when there is no message. The digest is how a refusal gets recorded once however many
    times it arrives: something we could not read has no id we are willing to trust -- frequently
    that is the very reason it was refused -- so the bytes are the only stable name it has.
    """

    handle: str
    digest: str
    preview: str
    message: IncomingMessage | None = None
    refusal: str | None = None
    ignored: str | None = None


@dataclass(frozen=True)
class Unread:
    """
    What one pass found: how many the channel said were waiting, and the ones it will read.

    `waiting` counts what was listed, not what was read, and the two differ whenever an item is
    offered and then not delivered. Reporting the second as though it were the first is how a
    backlog gets hidden: the count falls below the cap, the pass looks like it drained the channel,
    and whoever schedules the next one believes it.
    """

    waiting: int
    messages: Iterator[Fetched]


def described(raw: bytes) -> tuple[str, str]:
    """
    The digest and preview of something we may be about to refuse.

    errors="replace" rather than a decode that could raise: this describes input we already suspect,
    and a description that can fail in its turn is no use at the moment it is needed.
    """
    return sha256(raw).hexdigest(), raw[:PREVIEW_CHARS].decode("utf-8", errors="replace")
