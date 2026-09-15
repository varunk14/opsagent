-- The failure taxonomy: one named category per failed run.
--
-- Concept 2.13: "it broke" is not actionable. Every run that failed gets exactly one of
-- six fixed categories, so the mix over time says which kind is growing. It sits beside
-- failure_class, which keeps saying why a run died (an outage, an expired lock, a
-- rejection) and drives the dead-letter list; the category says how the agent went
-- wrong. A run that did what it should has neither.
--
-- The CHECK is the taxonomy: a seventh category is a migration, not a typo.

ALTER TABLE runs
    ADD COLUMN IF NOT EXISTS failure_category text
        CONSTRAINT runs_failure_category_in_taxonomy CHECK (failure_category IN (
            'hallucinated_field', 'tool_misuse', 'loop', 'context_overflow', 'wrong_escalation', 'drift'
        ));

-- The failure chart groups rested runs by week and category.
CREATE INDEX IF NOT EXISTS runs_failure_category_created_at_idx
    ON runs (failure_category, created_at)
    WHERE failure_category IS NOT NULL;
