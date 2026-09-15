"""
Spans for policy search: embedding the question, then searching by vector.

The embedding call is recorded with its model and how long it took, and says plainly
that its tokens are not counted yet: embedding cost is not in the run's cost, and a
span showing $0 would claim it was free. The vector search records how many passages
came back, how near the nearest was and which policies they came from -- never the
question, which is the customer's own words.
"""

from uuid import UUID

import psycopg
import pytest

from app.graph.build import build_graph, run_graph
from app.llm import ModelUnavailable
from app.retrieval import PolicyRetriever, PolicySearchUnavailable
from app.tracing import Attr, SpanRow, rows_for, run_context, tracer
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    EXTRACTED_4821,
    PROPOSED_LOOKUP,
    ScriptedModel,
)
from tests.test_retrieval import AxisEmbedder, loaded, retriever

pytestmark = pytest.mark.db

RUN = UUID("a2a97b37-62fb-4a4d-ab07-60172aa05ef9")
QUESTION = "I was charged twice, a duplicate on my card ending 4242"


def search(exported, retriever_: PolicyRetriever) -> list[SpanRow]:
    with run_context(RUN), tracer().start_as_current_span("tick"):
        retriever_.search(QUESTION)
    return rows_for(exported.get_finished_spans())


def named(rows: list[SpanRow], name: str) -> SpanRow:
    (row,) = [row for row in rows if row.name == name]
    return row


def children(rows: list[SpanRow], name: str) -> list[str]:
    parent = named(rows, name)
    return [row.name for row in rows if row.parent_span_id == parent.span_id]


def test_a_search_is_an_embedding_then_a_vector_search(fresh_database, exported):
    rows = search(exported, retriever(fresh_database, loaded(fresh_database)))

    assert children(rows, "tick") == ["embed_query", "vector_search"]
    assert (named(rows, "embed_query").kind, named(rows, "vector_search").kind) == ("embedding", "retriever")


def test_the_embedding_says_its_tokens_are_not_counted(fresh_database, exported):
    rows = search(exported, retriever(fresh_database, loaded(fresh_database)))

    embedding = named(rows, "embed_query")
    assert embedding.model == "axis-embed"
    assert (embedding.input_tokens, embedding.cost_usd) == (None, None)
    assert embedding.attributes[Attr.COST_COUNTED] is False


def test_the_vector_search_records_what_came_back(fresh_database, exported):
    rows = search(exported, retriever(fresh_database, loaded(fresh_database), k=2, max_distance=2.0))

    found = named(rows, "vector_search").attributes
    assert found[Attr.K] == 2
    assert found[Attr.PASSAGES] == 2
    assert found[Attr.SOURCES][0] == "duplicate-payments#0"
    assert 0 <= found[Attr.TOP_DISTANCE] < 1


def test_a_search_that_found_nothing_near_enough_says_so(fresh_database, exported):
    # Nothing can be nearer than a negative distance; the duplicate passage itself sits at exactly 0.
    rows = search(exported, retriever(fresh_database, loaded(fresh_database), max_distance=-1.0))

    found = named(rows, "vector_search").attributes
    assert found[Attr.PASSAGES] == 0
    assert Attr.TOP_DISTANCE not in found


def test_the_question_never_goes_into_a_span(fresh_database, exported):
    rows = search(exported, retriever(fresh_database, loaded(fresh_database)))

    for row in rows:
        written = repr(row.attributes) + (row.status_message or "")
        assert "4242" not in written, row.name
        assert "charged twice" not in written, row.name


def test_a_database_outage_marks_the_vector_search_as_the_failure(exported):
    def unreachable():
        raise psycopg.OperationalError("connection refused")

    with pytest.raises(PolicySearchUnavailable):
        search(exported, PolicyRetriever(unreachable, AxisEmbedder()))

    rows = rows_for(exported.get_finished_spans())
    assert named(rows, "embed_query").status == "ok"
    search_row = named(rows, "vector_search")
    assert (search_row.status, search_row.status_message) == ("error", "policy_search_unavailable")


def test_an_embedding_model_outage_marks_the_embedding_and_searches_nothing(fresh_database, exported):
    class Down(AxisEmbedder):
        def embed(self, texts):
            raise ModelUnavailable("cannot reach the embedding model")

    with pytest.raises(ModelUnavailable):
        search(exported, retriever(fresh_database, Down()))

    rows = rows_for(exported.get_finished_spans())
    embedding = named(rows, "embed_query")
    assert (embedding.status, embedding.status_message) == ("error", "model_unavailable")
    assert [row.name for row in rows if row.name == "vector_search"] == []


def test_inside_the_graph_the_search_sits_under_the_retrieve_step(fresh_database, exported):
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED_4821, plan=PROPOSED_LOOKUP)
    graph = build_graph(model, retriever(fresh_database, loaded(fresh_database)))

    with run_context(RUN), tracer().start_as_current_span("tick"):
        run_graph(graph, "Charged twice", "duplicate charge on order 4821")

    rows = rows_for(exported.get_finished_spans())
    assert children(rows, "retrieve") == ["embed_query", "vector_search"]
