"""
Turning a message into a run, exactly once.

The only interesting problem here is the second delivery. Pollers restart,
mailboxes re-serve messages, and a retry after a timeout looks identical to a new
request. Each of those must produce the run that already exists rather than
another one.

The check is left to the database. `INSERT ... ON CONFLICT DO NOTHING` is decided
inside Postgres while the row is being written, so two callers racing on the same
message cannot both pass it. The equivalent written in Python -- look for the
key, insert if absent -- has a gap between the two statements, and the gap is
where the duplicate is born.

DO NOTHING, not DO UPDATE. By the time a message is delivered again the agent may
be partway through the case, and an upsert would send it back to the start.

What DO NOTHING costs, and what is done about it: the second message does not
become a run. For an honest re-delivery that is exactly right, because the
contents are identical. For a forged key it is not, so the difference is
detected rather than assumed -- see IntakeResult.collided -- and the colliding
message is quarantined in dead_letters instead of being lost.
"""

from dataclasses import dataclass
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from app.contracts import IncomingMessage, RunRecord

INSERT_RUN = """
    INSERT INTO runs (
        id, channel, status, current_node, state, attempt, max_attempts,
        next_retry_at, idempotency_key, prompt_version, cost_usd, failure_class,
        created_at
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (idempotency_key) DO NOTHING
    RETURNING id
"""

# How many different colliding texts one key keeps. Past this, a forger varying one
# byte at a time is ignored instead of filling the table; the first few are enough
# for a person to see what was attempted.
MAX_QUARANTINED_PER_KEY = 5

# The same forged text arriving again is kept once, and one key keeps at most
# MAX_QUARANTINED_PER_KEY texts. The conflict target is migration 005's partial
# unique index. Two pollers quarantining for the same key at the same moment can
# each see room for one more, so the cap can be passed by one row per poller.
QUARANTINE = """
    INSERT INTO dead_letters (kind, idempotency_key, payload, reason)
    SELECT 'message', %s, %s, 'idempotency key collision: same key, different text'
     WHERE (SELECT count(*) FROM dead_letters WHERE kind = 'message' AND idempotency_key = %s) < %s
    ON CONFLICT (idempotency_key, md5(payload::text)) WHERE kind = 'message' DO NOTHING
"""


@dataclass(frozen=True)
class IntakeResult:
    """
    What happened, said plainly.

    `created` is false for a message that was already known. Callers mostly do
    not care, but it is the number worth watching: duplicates suddenly rising
    means something upstream is re-delivering.

    `collided` is the alarming one. It means a message arrived carrying a key
    that already exists, but different text -- so it is not a re-delivery of
    anything. Its contents are quarantined in dead_letters, once per distinct
    text, for a person to look at. An email Message-ID is
    chosen by whoever sent the email, so this is reachable by anyone who guesses
    the id a real customer's message will carry.
    """

    run_id: UUID
    created: bool
    collided: bool = False


def accept(connection: psycopg.Connection, message: IncomingMessage) -> IntakeResult:
    """
    Record `message` as a run, or report the run it already produced.

    Does not commit. The caller owns the transaction, because a poller will want
    to record the run and mark the source message as read together or not at all.
    """
    run = RunRecord.from_message(message)

    # Every column comes from the record rather than from the table's defaults.
    # The two agree today, and if they ever stop agreeing, changing the contract
    # should be what decides.
    inserted = connection.execute(
        INSERT_RUN,
        (
            run.id,
            run.channel,
            run.status,
            run.current_node,
            Jsonb(run.state),
            run.attempt,
            run.max_attempts,
            run.next_retry_at,
            run.idempotency_key,
            run.prompt_version,
            run.cost_usd,
            run.failure_class,
            run.created_at,
        ),
    ).fetchone()

    if inserted is not None:
        return IntakeResult(run_id=inserted[0], created=True)

    # The insert was refused, so the row belongs to an earlier message. Read it
    # back rather than assuming anything about what it contains.
    #
    # Postgres holds the second writer on the unique index until the first has
    # committed, so by the time a conflict is observable the row is there. It
    # could still be missing if something deleted it in between; nothing does
    # today, and nothing in the schema forbids it, so the case is handled loudly
    # rather than left to produce a confusing None.
    existing = connection.execute(
        "SELECT id, state FROM runs WHERE idempotency_key = %s", (run.idempotency_key,)
    ).fetchone()

    if existing is None:  # pragma: no cover - needs a concurrent DELETE of that row
        raise RuntimeError(
            f"insert of {run.idempotency_key} conflicted with a row that is not there"
        )

    run_id, stored_state = existing
    collided = stored_state.get("untrusted") != run.state["untrusted"]
    if collided:
        # Kept, not discarded, in the caller's transaction: the run keeps the text
        # that arrived first, and the colliding text waits for a person.
        connection.execute(
            QUARANTINE, (run.idempotency_key, Jsonb(run.state), run.idempotency_key, MAX_QUARANTINED_PER_KEY)
        )
    return IntakeResult(run_id=run_id, created=False, collided=collided)
