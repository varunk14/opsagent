-- Replay: a fresh run that carries an old run's message, threaded back to it.
--
-- The replay feature re-runs an old case under whatever the prompts, policies and code say now,
-- as a NEW run -- the original is never touched. `replay_of` is the thread back: null for a run
-- that came from a customer, and the original's id for a run that came from replaying it. Seeing
-- the two outcomes either side of that link is how you tell a fix changed anything.
--
-- The FK is the contract: a replay must point at a run that exists, or it is a diff with one side.
-- Nullable and with no default, so intake's insert -- which does not mention this column -- keeps
-- leaving it null for every ordinary run, exactly as before.

ALTER TABLE runs
    ADD COLUMN IF NOT EXISTS replay_of uuid REFERENCES runs (id);

-- "Show me the replays of this run" reads by the origin; only replays carry one.
CREATE INDEX IF NOT EXISTS runs_replay_of_idx
    ON runs (replay_of)
    WHERE replay_of IS NOT NULL;

COMMENT ON COLUMN runs.replay_of IS
    'The run this one is a replay of; null for a run that came from a customer message.';
