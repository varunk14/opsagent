-- One row per finished OpenTelemetry span: the permanent record of what a run did.
--
-- Written by app.tracing.record_spans inside the transaction that commits the step,
-- so a trace holds exactly the work that committed: a tick that lost its claim or
-- crashed leaves no spans, as it leaves no steps. Langfuse receives a copy over OTLP;
-- this table is what the run page reads, and it outlives anything done to Langfuse.
--
-- A trace is evidence, so the table is append-only, and it enforces its own shape:
-- the same span can be written to a second store without either of them guessing.

CREATE TABLE IF NOT EXISTS spans (
    trace_id        uuid NOT NULL REFERENCES runs (id),   -- a run's trace id is the run's id
    span_id         text NOT NULL,
    parent_span_id  text,
    name            text NOT NULL,
    kind            text NOT NULL,
    started_at      timestamptz NOT NULL,
    ended_at        timestamptz NOT NULL,
    status          text NOT NULL,
    status_message  text,
    model           text,
    prompt_version  text,
    input_tokens    integer,
    output_tokens   integer,
    -- Exact, never rounded per call: calls rounded one by one would not add up to runs.cost_usd.
    -- NULL means not counted, which is not the same as free.
    cost_usd        numeric,
    attributes      jsonb NOT NULL DEFAULT '{}',

    -- Span ids are random 64-bit numbers, unique within their trace, not across all of them.
    PRIMARY KEY (trace_id, span_id),
    CONSTRAINT spans_span_id_is_hex CHECK (span_id ~ '^[0-9a-f]{16}$'),
    CONSTRAINT spans_parent_is_hex CHECK (parent_span_id IS NULL OR parent_span_id ~ '^[0-9a-f]{16}$'),
    CONSTRAINT spans_name CHECK (btrim(name) <> ''),
    -- The observation types Langfuse accepts that this project uses.
    CONSTRAINT spans_kind CHECK (kind IN ('span', 'chain', 'generation', 'retriever', 'embedding', 'tool', 'guardrail')),
    CONSTRAINT spans_status CHECK (status IN ('ok', 'error')),
    CONSTRAINT spans_ends_after_start CHECK (ended_at >= started_at),
    CONSTRAINT spans_input_tokens CHECK (input_tokens >= 0),
    CONSTRAINT spans_output_tokens CHECK (output_tokens >= 0),
    CONSTRAINT spans_cost_usd CHECK (cost_usd >= 0),
    CONSTRAINT spans_prompt_version_is_hex CHECK (prompt_version IS NULL OR prompt_version ~ '^[0-9a-f]{12}$'),
    -- A model call with no model, tokens or cost would read as a free call.
    CONSTRAINT spans_generation_is_costed CHECK (
        kind <> 'generation'
        OR (model IS NOT NULL AND input_tokens IS NOT NULL AND output_tokens IS NOT NULL AND cost_usd IS NOT NULL)
    )
);

-- The run page reads one trace in time order.
CREATE INDEX IF NOT EXISTS spans_trace_id_started_at_idx ON spans (trace_id, started_at);

CREATE OR REPLACE FUNCTION spans_are_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'spans are append-only: % is refused', TG_OP;
END;
$$;

CREATE OR REPLACE TRIGGER spans_keep_the_record
    BEFORE UPDATE OR DELETE ON spans
    FOR EACH ROW EXECUTE FUNCTION spans_are_append_only();

-- TRUNCATE fires no row trigger, so it is refused on its own.
CREATE OR REPLACE TRIGGER spans_keep_the_table
    BEFORE TRUNCATE ON spans
    FOR EACH STATEMENT EXECUTE FUNCTION spans_are_append_only();
