-- The outbox: a reply the agent owes a customer, before it is a message anywhere.
--
-- A run that rests owes the customer a word about how it went. That word is a row
-- here, written in the SAME transaction as the outcome (app/run_agent.py): a refund
-- that committed cannot leave without the reply that says so committing with it, and
-- a crash before the send leaves the row pending rather than the customer in silence.
-- The drain (app/replies) sends it AFTER commit and marks it sent -- at-least-once,
-- the same ordering the poll channels keep, because the safe way to be wrong is to
-- send a duplicate, never to drop the only copy.
--
-- The message is fixed by outcome and frozen at write time. `template` names which
-- fixed wording was chosen; `body` is that wording already rendered, so what the
-- customer will read is decided once, in the transaction that earned it, and cannot
-- drift if a template is later reworded. No customer text is interpolated beyond a
-- validated order number and a computed amount, so the body carries nothing untrusted.
--
-- `reply_to` and `thread_ref` are resolved here and not re-derived later, because the
-- run does not store enough to reply from: an email's Message-ID lives only in the
-- idempotency key, and a Telegram reply goes to the chat, not to the sender id the run
-- kept for rate-limiting. Frozen at write time, the drain needs nothing but this row.
--
-- The CHECKs are the contract: a template outside the set is an outcome nothing
-- rendered, a state outside the machine is a row no drain can act on, a channel the
-- run never came from is a reply with nowhere to go. There is deliberately no
-- UNIQUE(run_id): a run handed to a person and later refunded owes two honest replies.

CREATE TABLE IF NOT EXISTS outbox (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id      uuid NOT NULL REFERENCES runs(id),
    channel     text NOT NULL CHECK (channel IN ('email', 'telegram', 'form')),
    reply_to    text NOT NULL,                       -- resolved target: an address, or a chat id
    thread_ref  text,                                -- an email Message-ID to thread on; null otherwise
    template    text NOT NULL CHECK (template IN (
                    'refund_issued', 'handed_to_person', 'order_not_found', 'enquiry'
                )),
    body        text NOT NULL,                       -- the chosen wording, already rendered and frozen
    state       text NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'sent', 'failed')),
    attempts    integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    created_at  timestamptz NOT NULL DEFAULT now(),
    sent_at     timestamptz,
    last_error  text
);

-- The drain claims the oldest pending rows first; only pending rows are on its path.
CREATE INDEX IF NOT EXISTS outbox_pending_created_at_idx
    ON outbox (created_at)
    WHERE state = 'pending';

COMMENT ON COLUMN outbox.reply_to IS
    'Where the reply goes, resolved at write time: an email address, or a Telegram chat id.';
COMMENT ON COLUMN outbox.thread_ref IS
    'The email Message-ID to thread the reply onto; null for channels without threading.';
COMMENT ON COLUMN outbox.body IS
    'The fixed wording for this outcome, already rendered and frozen, carrying no untrusted text.';
