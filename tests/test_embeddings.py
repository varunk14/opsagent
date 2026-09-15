"""
Turning text into vectors, and refusing vectors that cannot be trusted.

An embedding of the wrong length would not fit the vector(768) column, and one
from a malformed reply would silently rank every search wrong. Both are refused
at the boundary, the same way model replies are.
"""

import json

import pytest
from app.embeddings import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    QUERY_PREFIX,
    OllamaEmbedder,
    embed_documents,
    embed_query,
    parse_embeddings,
)

from app.llm import ModelUnavailable
from tests.fakes import FakeEmbedder


def body(vectors) -> bytes:
    return json.dumps({"model": "nomic-embed-text", "embeddings": vectors}).encode()


def test_a_well_formed_reply_becomes_vectors():
    vectors = [[0.1] * EMBEDDING_DIMENSIONS, [0.2] * EMBEDDING_DIMENSIONS]

    assert parse_embeddings(body(vectors), expected=2) == vectors


def test_a_vector_of_the_wrong_length_is_refused():
    with pytest.raises(ModelUnavailable, match="768"):
        parse_embeddings(body([[0.1] * 384]), expected=1)


def test_the_wrong_number_of_vectors_is_refused():
    """One vector for two texts would pair a chunk with someone else's meaning."""
    with pytest.raises(ModelUnavailable, match="expected 2"):
        parse_embeddings(body([[0.1] * EMBEDDING_DIMENSIONS]), expected=2)


def test_a_reply_without_embeddings_is_refused():
    with pytest.raises(ModelUnavailable, match="embeddings"):
        parse_embeddings(json.dumps({"model": "x"}).encode(), expected=1)


def test_a_reply_that_is_not_json_is_refused():
    with pytest.raises(ModelUnavailable, match="JSON"):
        parse_embeddings(b"<html>502</html>", expected=1)


def test_a_vector_with_non_finite_numbers_is_refused():
    vector = ", ".join(["0.1"] * (EMBEDDING_DIMENSIONS - 1) + ["NaN"])
    with pytest.raises(ModelUnavailable, match="finite"):
        parse_embeddings(f'{{"embeddings": [[{vector}]]}}'.encode(), expected=1)


def test_documents_and_queries_carry_the_prefixes_nomic_expects():
    """nomic-embed-text is trained with these task prefixes; leaving them off degrades search."""
    embedder = FakeEmbedder()

    embed_documents(embedder, ["a policy passage"])
    embed_query(embedder, "charged twice")

    assert embedder.calls == [[f"{DOCUMENT_PREFIX}a policy passage"], [f"{QUERY_PREFIX}charged twice"]]


def test_embedding_nothing_asks_the_model_nothing():
    embedder = FakeEmbedder()

    assert embed_documents(embedder, []) == []
    assert embedder.calls == []


def test_the_client_defaults_to_local_nomic():
    client = OllamaEmbedder()

    assert client.model == EMBEDDING_MODEL == "nomic-embed-text"
    assert client.endpoint == "http://localhost:11434/api/embed"
