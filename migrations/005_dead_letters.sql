-- Where work that cannot be processed goes, instead of vanishing.
--
-- Two kinds of letter. A dead run -- out of attempts, or its lock expired on its
-- last attempt -- gets one naming it, so an operator can see why and requeue it.
-- A quarantined message never became a run of its own: it arrived carrying the
-- idempotency key of a message already recorded, with different text. Email
-- Message-IDs are chosen by the sender, so that is a forged or colliding key,
-- and until now its text was discarded without a trace.
--
-- A run has at most one open letter; once it has been requeued, dying again
-- opens a new one. The same quarantined message is kept once, however many times
-- it is delivered.

CREATE TABLE IF NOT EXISTS dead_letters (
    id              bigserial PRIMARY KEY,
    kind            text NOT NULL CHECK (kind IN ('run', 'message')),
    run_id          uuid REFERENCES runs (id),
    idempotency_key text NOT NULL,
    payload         jsonb NOT NULL,
    reason          text NOT NULL,
    failure_class   text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    requeued_at     timestamptz,
    CONSTRAINT dead_letters_run_named_only_for_runs CHECK ((kind = 'run') = (run_id IS NOT NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS dead_letters_one_open_per_run
    ON dead_letters (run_id) WHERE kind = 'run' AND requeued_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS dead_letters_message_once
    ON dead_letters (idempotency_key, md5(payload::text)) WHERE kind = 'message';

-- Lock expiry looks for running runs by the age of their lock on every claim.
CREATE INDEX IF NOT EXISTS runs_running_locked_at_idx ON runs (locked_at) WHERE status = 'running';
