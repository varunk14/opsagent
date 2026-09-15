-- Makes policy passages reloadable without duplicates or stale vectors.
--
-- chunk_index and the unique (document, chunk_index) index give each passage a
-- stable identity. content_hash lets ingest skip passages whose text has not
-- changed, so an unchanged document is never embedded twice. embedding_model
-- records which model made each vector, because vectors from two models live in
-- different spaces and must never be compared.
--
-- policy_chunks was never written before this migration, so the NOT NULL
-- constraints apply cleanly. If rows without these columns ever exist, this
-- fails loudly rather than guessing values for them.

ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS chunk_index int;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS content_hash text;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS embedding_model text;

ALTER TABLE policy_chunks ALTER COLUMN chunk_index SET NOT NULL;
ALTER TABLE policy_chunks ALTER COLUMN content_hash SET NOT NULL;
ALTER TABLE policy_chunks ALTER COLUMN embedding_model SET NOT NULL;
ALTER TABLE policy_chunks ALTER COLUMN embedding SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS policy_chunks_document_chunk_idx
    ON policy_chunks (document, chunk_index);
