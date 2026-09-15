-- The guardrail, and an approvals table that can hold a whole decision.
--
-- Concept 2.10: which refunds run on their own is decided by numbers in the
-- database, compared in code -- never by the prompt. One row, so "the limit" is
-- never ambiguous; it cannot be deleted, because no row would mean no limit.
-- Changing it is an UPDATE that records who made it, not a deploy.
--
-- The approvals table from 001 could say an action was approved but not why a
-- person was asked, when, or whether the approved action has since run. Every
-- column added here closes one of those, and the CHECKs make a half-recorded
-- decision impossible rather than merely unusual.

CREATE TABLE IF NOT EXISTS guardrails (
    singleton               boolean PRIMARY KEY DEFAULT true
                                CONSTRAINT guardrails_one_row CHECK (singleton),
    -- A refund strictly under this runs on its own. 0 makes every refund manual.
    auto_refund_limit_paise bigint NOT NULL
                                CONSTRAINT guardrails_limit_not_negative CHECK (auto_refund_limit_paise >= 0),
    min_confidence          numeric(3, 2) NOT NULL
                                CONSTRAINT guardrails_confidence_in_range CHECK (min_confidence BETWEEN 0 AND 1),
    updated_by              text NOT NULL
                                CONSTRAINT guardrails_updated_by_named CHECK (btrim(updated_by) <> ''),
    updated_at              timestamptz NOT NULL DEFAULT now()
);

-- The defaults: under Rs 5,000, at least 0.85 confident.
INSERT INTO guardrails (auto_refund_limit_paise, min_confidence, updated_by)
VALUES (500000, 0.85, 'migration 006')
ON CONFLICT DO NOTHING;

CREATE OR REPLACE FUNCTION guardrails_keep_the_row() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'the guardrails row cannot be deleted; set auto_refund_limit_paise = 0 to stop automatic refunds';
END;
$$;

CREATE OR REPLACE TRIGGER guardrails_keep_the_row
    BEFORE DELETE ON guardrails
    FOR EACH ROW EXECUTE FUNCTION guardrails_keep_the_row();

-- TRUNCATE fires no row-level trigger, so it needs its own.
CREATE OR REPLACE TRIGGER guardrails_keep_the_row_on_truncate
    BEFORE TRUNCATE ON guardrails
    FOR EACH STATEMENT EXECUTE FUNCTION guardrails_keep_the_row();

-- reason has no default because there is no honest one to backfill. Nothing wrote
-- approvals before this migration, so the table is empty when it runs; if it were
-- not, this fails loudly rather than inventing a reason for a past decision.
ALTER TABLE approvals
    ADD COLUMN IF NOT EXISTS reason        text NOT NULL,          -- why a person is being asked
    ADD COLUMN IF NOT EXISTS created_at    timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS decision_note text,
    ADD COLUMN IF NOT EXISTS executed_at   timestamptz;           -- when the approved action ran

-- Decided exactly when not pending: by someone with a name, at a time.
ALTER TABLE approvals
    ADD CONSTRAINT approvals_decided_by_whom CHECK ((status = 'pending') = (decided_by IS NULL)),
    ADD CONSTRAINT approvals_decided_when CHECK ((status = 'pending') = (decided_at IS NULL)),
    ADD CONSTRAINT approvals_decided_by_named CHECK (decided_by IS NULL OR btrim(decided_by) <> ''),
    -- Only an approved action can have run.
    ADD CONSTRAINT approvals_executed_only_if_approved CHECK (executed_at IS NULL OR status = 'approved');

-- A record, not a scratchpad. What a person is asked to approve never changes, or
-- the refund they saw on the screen is not the refund that runs. A decision, once
-- made, is not rewritten. An executed approval stays executed, or the worker would
-- run it again. The only changes left are deciding once and stamping execution once.
CREATE OR REPLACE FUNCTION approvals_keep_the_record() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.run_id, NEW.action, NEW.evidence, NEW.confidence, NEW.reason, NEW.created_at)
       IS DISTINCT FROM (OLD.run_id, OLD.action, OLD.evidence, OLD.confidence, OLD.reason, OLD.created_at) THEN
        RAISE EXCEPTION 'approval %: what a person is asked to approve cannot change', OLD.id;
    END IF;
    IF OLD.status <> 'pending'
       AND (NEW.status, NEW.decided_by, NEW.decided_at, NEW.decision_note)
           IS DISTINCT FROM (OLD.status, OLD.decided_by, OLD.decided_at, OLD.decision_note) THEN
        RAISE EXCEPTION 'approval %: a decision cannot be rewritten', OLD.id;
    END IF;
    IF OLD.executed_at IS NOT NULL AND NEW.executed_at IS DISTINCT FROM OLD.executed_at THEN
        RAISE EXCEPTION 'approval %: an executed approval stays executed', OLD.id;
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER approvals_keep_the_record
    BEFORE UPDATE ON approvals
    FOR EACH ROW EXECUTE FUNCTION approvals_keep_the_record();

-- Two pending approvals for one run would let one refund be approved twice.
CREATE UNIQUE INDEX IF NOT EXISTS approvals_one_pending_per_run
    ON approvals (run_id) WHERE status = 'pending';

-- And a run waits to execute at most one approved action.
CREATE UNIQUE INDEX IF NOT EXISTS approvals_one_unexecuted_per_run
    ON approvals (run_id) WHERE status = 'approved' AND executed_at IS NULL;

-- The approvals screen lists pending approvals oldest first, ties broken by id.
CREATE INDEX IF NOT EXISTS approvals_pending_created_at_idx
    ON approvals (created_at, id) WHERE status = 'pending';
