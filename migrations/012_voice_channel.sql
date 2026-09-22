-- Voice as a first-class channel.
--
-- Both `runs` and `outbox` pin `channel` with a CHECK, so the database refuses a value the
-- application does not know how to route. Adding voice widens each list by exactly one value.
--
-- The constraint is dropped and re-added under the same name Postgres would have chosen for it,
-- because a NOT VALID / ADD would leave the runs and outbox tables carrying two overlapping
-- checks after later migrations, and a name is what any future migration would drop it by.
-- Idempotent: DROP ... IF EXISTS, and the new CHECK is a superset of the old one, so re-running
-- this migration on a database that already has it succeeds without changing any row.

ALTER TABLE runs   DROP CONSTRAINT IF EXISTS runs_channel_check;
ALTER TABLE runs   ADD  CONSTRAINT runs_channel_check
                        CHECK (channel IN ('email', 'telegram', 'form', 'voice'));

ALTER TABLE outbox DROP CONSTRAINT IF EXISTS outbox_channel_check;
ALTER TABLE outbox ADD  CONSTRAINT outbox_channel_check
                        CHECK (channel IN ('email', 'telegram', 'form', 'voice'));
