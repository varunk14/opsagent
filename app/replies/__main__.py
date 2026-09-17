"""
One drain over the outbox: send every reply a rested run left, and mark each sent.

The mirror of `python -m app.poll`. Poll brings customers' messages in; this puts the
agent's answers back out. Both are one bounded pass meant to be run again and again --
by hand while developing, on a timer in the end -- and both are safe to interrupt,
because a reply is marked sent only after it has been sent.

Run:  .venv/bin/python -m app.replies
"""

import os
import sys

from app.db import connect
from app.replies.send import (
    MAIL_PASSWORD_VAR,
    TELEGRAM_TOKEN_VAR,
    drain,
    senders_from_env,
)


def main() -> int:  # pragma: no cover - the interactive driver
    senders = senders_from_env(os.environ)
    if not senders:
        print("  No channel is configured to reply on.")
        print(f"  Set the mailbox ({MAIL_PASSWORD_VAR} and its host/user) or the bot")
        print(f"  ({TELEGRAM_TOKEN_VAR}) in .env, which is not committed.")
        return 1

    with connect() as connection:
        summary = drain(connection, senders)

    print(f"  reachable {', '.join(sorted(senders))}")
    print(f"  sent      {summary.sent}")
    if summary.failed:
        print(f"\n  FAILED    {summary.failed}")
        print("  A reply could not be sent. It stays pending and is tried again, up")
        print("  to a few times, then rests as failed. `python -m app.dead_letters`")
        print("  is unrelated; a failed reply is on the run's own page.")
    elif summary.sent == 0:
        print("\n  Nothing to send. Run it again whenever a run has rested.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
