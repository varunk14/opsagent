"""
Loading policy passages and their embeddings into pgvector, idempotently.

Re-running ingest on unchanged documents must embed nothing and change nothing:
it runs on every deploy, and every embedding call costs time. A changed
document replaces its own passages and leaves the others alone.
"""

import psycopg
import pytest
from app.policies import ingest, load_policies

from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.db

POLICIES = {
    "duplicate-payments": "# Duplicate payments\n\n## What we do\n\nThe duplicate is returned.\n\n## Timing\n\nTwo days.",
    "delivery-times": "# Delivery\n\n## Standard\n\nThree to five days.",
}


def rows(dsn: str) -> list[tuple]:
    with psycopg.connect(dsn) as connection:
        return connection.execute(
            "SELECT document, chunk_index, chunk, embedding_model, vector_dims(embedding) "
            "FROM policy_chunks ORDER BY document, chunk_index"
        ).fetchall()


def test_the_schema_can_hold_idempotent_passages(fresh_database):
    with psycopg.connect(fresh_database) as connection:
        columns = {
            row[0]
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'policy_chunks'"
            ).fetchall()
        }
        assert {"chunk_index", "content_hash", "embedding_model"} <= columns

        insert = (
            "INSERT INTO policy_chunks (document, chunk_index, chunk, content_hash, embedding_model, embedding) "
            "VALUES ('d', 0, 'x', 'h', 'm', %s::vector)"
        )
        zero = "[" + ",".join(["0"] * 767 + ["1"]) + "]"
        connection.execute(insert, (zero,))
        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(insert, (zero,))


def test_every_passage_is_stored_with_a_768_dimension_embedding(fresh_database):
    with psycopg.connect(fresh_database) as connection:
        summary = ingest(connection, FakeEmbedder(), POLICIES)

    stored = rows(fresh_database)
    assert summary.added == 3
    assert [(doc, idx) for doc, idx, *_ in stored] == [
        ("delivery-times", 0), ("duplicate-payments", 0), ("duplicate-payments", 1),
    ]
    assert {model for *_, model, _ in stored} == {"fake-embed"}
    assert {dims for *_, dims in stored} == {768}


def test_passages_are_embedded_as_documents(fresh_database):
    embedder = FakeEmbedder()

    with psycopg.connect(fresh_database) as connection:
        ingest(connection, embedder, POLICIES)

    sent = [text for call in embedder.calls for text in call]
    assert sent and all(text.startswith("search_document: ") for text in sent)


def test_ingesting_unchanged_documents_embeds_nothing(fresh_database):
    embedder = FakeEmbedder()
    with psycopg.connect(fresh_database) as connection:
        ingest(connection, embedder, POLICIES)
    calls_after_first = len(embedder.calls)

    with psycopg.connect(fresh_database) as connection:
        again = ingest(connection, embedder, POLICIES)

    assert (again.added, again.unchanged, again.removed) == (0, 2, 0)
    assert len(embedder.calls) == calls_after_first


def test_a_changed_document_replaces_only_its_own_passages(fresh_database):
    with psycopg.connect(fresh_database) as connection:
        ingest(connection, FakeEmbedder(), POLICIES)
    edited = {**POLICIES, "delivery-times": "# Delivery\n\n## Standard\n\nSeven days now."}

    with psycopg.connect(fresh_database) as connection:
        summary = ingest(connection, FakeEmbedder(), edited)

    texts = [chunk for _, _, chunk, *_ in rows(fresh_database)]
    assert summary.added == 1 and summary.unchanged == 1
    assert any("Seven days now." in t for t in texts)
    assert not any("Three to five days." in t for t in texts)
    assert sum("returned" in t for t in texts) == 1


def test_a_document_that_disappears_takes_its_passages_with_it(fresh_database):
    with psycopg.connect(fresh_database) as connection:
        ingest(connection, FakeEmbedder(), POLICIES)

    with psycopg.connect(fresh_database) as connection:
        summary = ingest(connection, FakeEmbedder(), {"delivery-times": POLICIES["delivery-times"]})

    assert summary.removed == 1
    assert {doc for doc, *_ in rows(fresh_database)} == {"delivery-times"}


def test_passages_from_a_different_embedding_model_are_re_embedded(fresh_database):
    """Vectors from two models live in different spaces; comparing them is meaningless."""
    with psycopg.connect(fresh_database) as connection:
        ingest(connection, FakeEmbedder(), POLICIES)

    class OtherModel(FakeEmbedder):
        model = "other-embed"

    with psycopg.connect(fresh_database) as connection:
        summary = ingest(connection, OtherModel(), POLICIES)

    assert summary.added == 3
    assert {model for *_, model, _ in rows(fresh_database)} == {"other-embed"}


def test_ingest_refuses_a_connection_already_in_a_transaction(fresh_database):
    with psycopg.connect(fresh_database) as connection:
        connection.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="own transaction"):
            ingest(connection, FakeEmbedder(), POLICIES)


def test_the_repository_policies_ingest_cleanly(fresh_database, tmp_path):
    from pathlib import Path

    policies = load_policies(Path(__file__).resolve().parent.parent / "policies")

    with psycopg.connect(fresh_database) as connection:
        summary = ingest(connection, FakeEmbedder(), policies)

    assert summary.added >= len(policies)
