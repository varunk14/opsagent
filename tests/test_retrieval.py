"""
Searching policy passages by meaning, in pgvector.

The fake embedder here puts each keyword on its own axis, so "near" and "far"
are exact and the tests are about the search, not about a model's judgement.
Whether real embeddings rank real policies correctly is the opt-in
tests/test_policy_search_real.py.
"""

import psycopg
import pytest

from app.embeddings import QUERY_PREFIX
from app.policies import ingest
from app.retrieval import MAX_QUERY_CHARS, PolicyRetriever

pytestmark = pytest.mark.db

AXES = {"duplicate": 0, "parcel": 1, "broken": 2}

DOCS = {
    "duplicate-payments": "# Duplicate payments\n\n## What we do\n\nduplicate returned in full",
    "delivery-times": "# Delivery\n\n## Standard\n\nparcel arrives in days",
    "damaged-items": "# Damage\n\n## Remedy\n\nbroken items replaced",
}


class AxisEmbedder:
    model = "axis-embed"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * 767 + [0.1]  # shared component: no zero vectors, finite distances
            for keyword, axis in AXES.items():
                if keyword in text.lower():
                    vector[axis] = 1.0
                    break
            vectors.append(vector)
        return vectors


def loaded(dsn: str, embedder=None) -> AxisEmbedder:
    embedder = embedder or AxisEmbedder()
    with psycopg.connect(dsn) as connection:
        ingest(connection, embedder, DOCS)
    return embedder


def retriever(dsn: str, embedder, **options) -> PolicyRetriever:
    return PolicyRetriever(lambda: psycopg.connect(dsn), embedder, **options)


def test_the_nearest_passage_comes_first(fresh_database):
    embedder = loaded(fresh_database)

    hits = retriever(fresh_database, embedder).search("it looks like a duplicate payment")

    assert hits[0].document == "duplicate-payments"
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


def test_a_passage_says_where_it_came_from(fresh_database):
    embedder = loaded(fresh_database)

    hit = retriever(fresh_database, embedder).search("duplicate")[0]

    assert (hit.document, hit.chunk_index) == ("duplicate-payments", 0)
    assert "duplicate returned in full" in hit.text


def test_passages_too_far_away_are_left_out(fresh_database):
    """Unrelated policy is worse than none: the planner is told nothing applies."""
    embedder = loaded(fresh_database)

    assert retriever(fresh_database, embedder).search("hello there") == []


def test_no_more_than_k_passages_come_back(fresh_database):
    embedder = loaded(fresh_database)

    hits = retriever(fresh_database, embedder, k=1, max_distance=2.0).search("hello there")

    assert len(hits) == 1


def test_only_passages_from_the_same_embedding_model_are_compared(fresh_database):
    loaded(fresh_database)

    class OtherModel(AxisEmbedder):
        model = "other-embed"

    assert retriever(fresh_database, OtherModel(), max_distance=2.0).search("duplicate") == []


def test_an_empty_corpus_returns_nothing(fresh_database):
    assert retriever(fresh_database, AxisEmbedder(), max_distance=2.0).search("duplicate") == []


def test_the_question_is_embedded_as_a_query(fresh_database):
    embedder = loaded(fresh_database)

    retriever(fresh_database, embedder).search("duplicate")

    assert embedder.calls[-1] == [f"{QUERY_PREFIX}duplicate"]


def test_an_overlong_question_is_cut_before_embedding(fresh_database):
    """The question is the customer's message, which may be up to 200,000 characters."""
    embedder = loaded(fresh_database)

    retriever(fresh_database, embedder).search("duplicate " + "x" * 50_000)

    assert len(embedder.calls[-1][0]) == len(QUERY_PREFIX) + MAX_QUERY_CHARS


# --- review findings -------------------------------------------------------------------


def test_the_nearest_passage_wins_even_when_it_was_stored_last(fresh_database):
    """Found in review: with k equal to the corpus size, a missing ORDER BY passed every test."""
    embedder = AxisEmbedder()
    with psycopg.connect(fresh_database) as connection:
        ingest(connection, embedder, dict(reversed(list(DOCS.items()))))

    hits = retriever(fresh_database, embedder, k=1, max_distance=2.0).search("duplicate")

    assert [hit.document for hit in hits] == ["duplicate-payments"]


def test_passages_equally_near_come_back_in_the_same_order_every_time(fresh_database):
    """
    Found in review: equal distances had no tie-break, so equally near passages -- and the
    plan prompt built from them -- could come back in either order, and a recorded reply
    keyed by that prompt would no longer be found on replay.
    """
    embedder = AxisEmbedder()
    same = "# Refunds\n\n## Rule\n\nduplicate returned in full"
    with psycopg.connect(fresh_database) as connection:
        ingest(connection, embedder, {"zeta-refunds": same, "alpha-refunds": same, "mid-refunds": same})

    hits = retriever(fresh_database, embedder, k=3, max_distance=2.0).search("duplicate")

    assert [hit.document for hit in hits] == ["alpha-refunds", "mid-refunds", "zeta-refunds"]


def test_a_database_outage_during_search_is_an_outage_not_a_crash():
    """Found in review: psycopg's OperationalError escaped, crashing the worker mid-batch."""
    from app.llm import ServiceUnavailable
    from app.retrieval import PolicySearchUnavailable

    def unreachable():
        raise psycopg.OperationalError("connection refused")

    with pytest.raises(PolicySearchUnavailable) as raised:
        PolicyRetriever(unreachable, AxisEmbedder()).search("duplicate")

    assert isinstance(raised.value, ServiceUnavailable)
    assert raised.value.failure_class == "policy_search_unavailable"


@pytest.mark.parametrize("schema", ["no tables at all", "before migration 002"])
def test_a_policy_store_without_its_schema_is_an_outage_not_a_crash(empty_database, schema):
    """
    Found in milestone review: UndefinedTable and UndefinedColumn are
    ProgrammingErrors, not OperationalErrors, so they escaped, crashed the
    worker, and left the run marked running with no route to dead.
    """
    from pathlib import Path

    from app.retrieval import PolicySearchUnavailable

    if schema == "before migration 002":
        first = Path(__file__).resolve().parent.parent / "migrations" / "001_schema.sql"
        with psycopg.connect(empty_database) as connection:
            connection.execute(first.read_text())

    with pytest.raises(PolicySearchUnavailable):
        PolicyRetriever(lambda: psycopg.connect(empty_database), AxisEmbedder()).search("duplicate")
