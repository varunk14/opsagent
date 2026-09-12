-- The six tables the whole system sits on.
--
-- Applied by app.db.apply_migrations, not by the container, because it has to
-- run against both databases: opsagent for development and opsagent_test for
-- the suite. Only 000_bootstrap.sql runs at container start, since creating a
-- database is the one thing a normal connection cannot do.
--
-- Two deliberate departures from the handbook's sketch, both making the database
-- enforce what was written there only as a comment:
--
--   * status columns carry CHECK constraints. A typo that invents a seventh run
--     state would otherwise sit in the table until a worker failed to match it.
--   * foreign keys are spelled out, so an order cannot belong to no one.

CREATE EXTENSION IF NOT EXISTS vector;

-- The spine: one row per request, durable across restarts.
CREATE TABLE IF NOT EXISTS runs (
    id              uuid PRIMARY KEY,
    channel         text NOT NULL CHECK (channel IN ('email', 'telegram', 'form')),
    status          text NOT NULL CHECK (status IN (
                        'queued', 'running', 'waiting_approval',
                        'done', 'failed', 'dead')),
    current_node    text NOT NULL,
    state           jsonb NOT NULL,          -- short-term memory for this run
    attempt         int NOT NULL DEFAULT 0,
    max_attempts    int NOT NULL DEFAULT 5,
    next_retry_at   timestamptz,
    idempotency_key text UNIQUE,             -- stops the same message being run twice
    locked_by       text,
    locked_at       timestamptz,
    prompt_version  text,
    cost_usd        numeric(10, 6) NOT NULL DEFAULT 0,
    failure_class   text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- The worker polls for due work on every tick. Without this it is a full scan
-- of every run ever recorded.
CREATE INDEX IF NOT EXISTS runs_status_next_retry_at_idx ON runs (status, next_retry_at);

-- The business data the agent acts on.
CREATE TABLE IF NOT EXISTS customers (
    email      text PRIMARY KEY,
    name       text,
    facts      jsonb NOT NULL DEFAULT '{}',  -- long-term memory about this person
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS orders (
    id             text PRIMARY KEY,
    customer_email text NOT NULL REFERENCES customers (email),
    amount_paise   bigint NOT NULL,          -- a whole number of paise. never a float.
    status         text NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- The human-in-the-loop queue.
CREATE TABLE IF NOT EXISTS approvals (
    id         bigserial PRIMARY KEY,
    run_id     uuid NOT NULL REFERENCES runs (id),
    action     jsonb NOT NULL,               -- what the agent wants to do
    evidence   jsonb NOT NULL,               -- why it wants to
    confidence numeric(3, 2),
    status     text NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'approved', 'rejected')),
    decided_by text,
    decided_at timestamptz
);

-- Every state-changing tool call, keyed so a repeat replays instead of re-running.
CREATE TABLE IF NOT EXISTS tool_calls (
    idempotency_key text PRIMARY KEY,
    run_id          uuid NOT NULL REFERENCES runs (id),
    tool            text NOT NULL,
    args            jsonb NOT NULL,
    result          jsonb,                   -- returned again on a repeated call
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- The retrieval corpus. 768 dimensions to match the local embedding model; a
-- mismatch here is only discovered at query time.
CREATE TABLE IF NOT EXISTS policy_chunks (
    id        bigserial PRIMARY KEY,
    document  text NOT NULL,
    chunk     text NOT NULL,
    embedding vector(768)
);

CREATE INDEX IF NOT EXISTS policy_chunks_embedding_idx
    ON policy_chunks USING hnsw (embedding vector_cosine_ops);
