-- Where a synthesized voice reply is kept until the customer plays it.
--
-- The outbox row records that a reply is owed, in the same transaction as the outcome; the drain
-- calls Sarvam TTS and stores the wav here, keyed on the run. `GET /runs/{id}/reply.wav` reads
-- the row and streams the bytes. The mark on the outbox row commits after this row is written, so
-- a crash between synthesis and mark means the next drain will resynthesize -- overwriting is
-- refused by the PK, so the retry uses `ON CONFLICT DO NOTHING` and the mark still fires.
--
-- One reply per run, so `run_id` is both the primary key and the foreign key: a run does not owe a
-- second synthesized reply, and looking one up by run id is the only access the screen has. The
-- audio is bytea rather than a filesystem path because the wav lives with the run: a container
-- restart cannot lose a file that has already been committed to the same database as the outbox
-- row that recorded it. Sizes are tens to hundreds of kilobytes for the short replies here, well
-- under the outbox reply cap and any operational concern about row width.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS voice_replies (
    run_id         uuid PRIMARY KEY REFERENCES runs(id),
    audio          bytea NOT NULL,
    synthesized_at timestamptz NOT NULL DEFAULT now()
);
