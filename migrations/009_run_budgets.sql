-- What one run may spend before a person takes it.
--
-- The guardrail already said which refunds run without a person. It said nothing
-- about what a run may cost to reach that decision, so a run that kept planning
-- was bounded only by MAX_STEPS -- an indirect limit that says nothing about
-- tokens, money or time, and one that moves whenever the agent's shape changes.
--
-- These ceilings are measured, not guessed. Before any optimisation a run used
-- 3,037 tokens and 19.7 seconds at the median, 31.9 at p95, and cost $0.000541.
-- The defaults are roughly three times that: high enough that no healthy run
-- meets them, low enough that a run spending pathologically stops.
--
-- Seconds mean seconds the agent spent working -- the latency of its own model
-- calls, added up. Not wall-clock since the message arrived: a run waiting for a
-- person to approve a refund can sit for days, and charging that against a
-- budget would hand over every refund anybody took a lunch break over.
--
-- Zero means no ceiling, unlike auto_refund_limit_paise where zero is the kill
-- switch. A budget that stopped every run the moment it was set to zero would
-- make the safe way to switch a budget off indistinguishable from the harshest
-- setting there is.

ALTER TABLE guardrails
    ADD COLUMN IF NOT EXISTS max_tokens_per_run bigint NOT NULL DEFAULT 10000
        CONSTRAINT guardrails_max_tokens_not_negative CHECK (max_tokens_per_run >= 0),
    ADD COLUMN IF NOT EXISTS max_cost_usd_per_run numeric(10, 6) NOT NULL DEFAULT 0.002000
        CONSTRAINT guardrails_max_cost_not_negative CHECK (max_cost_usd_per_run >= 0),
    ADD COLUMN IF NOT EXISTS max_seconds_per_run integer NOT NULL DEFAULT 180
        CONSTRAINT guardrails_max_seconds_not_negative CHECK (max_seconds_per_run >= 0);

COMMENT ON COLUMN guardrails.max_tokens_per_run IS
    'Tokens one run may spend across every model call. 0 means no ceiling.';
COMMENT ON COLUMN guardrails.max_cost_usd_per_run IS
    'Reference cost one run may reach, priced per model. 0 means no ceiling.';
COMMENT ON COLUMN guardrails.max_seconds_per_run IS
    'Seconds of model latency one run may accumulate, not wall-clock. 0 means no ceiling.';
