"""
Searching policy passages by meaning, in pgvector.

The question is embedded as a query, then compared by cosine distance against
passages embedded by the same model -- vectors from different models live in
different spaces, so comparing across them would rank by noise.

Two limits keep bad context out of the planning prompt. At most k passages come
back, and anything farther than max_distance is dropped: an unrelated policy
presented as relevant is worse than being told nothing applies.

Each search opens its own short connection. The driver runs the graph with no
transaction open, and retrieval must not change that.
"""

from collections.abc import Callable

import psycopg

from app.embeddings import Embedder, embed_query
from app.graph.state import PolicyPassage
from app.llm import ServiceUnavailable

DEFAULT_K = 3
# Tuned on these policies with nomic-embed-text: relevant passages sat at
# 0.32-0.36 cosine distance, the nearest unrelated ones from 0.41. An evaluation
# golden set is where this gets measured properly rather than eyeballed.
DEFAULT_MAX_DISTANCE = 0.45
# The question is the customer's message, which can be 200,000 characters. The
# opening of a message carries its point; the rest would only dilute the vector.
MAX_QUERY_CHARS = 2_000

SEARCH = """
    SELECT document, chunk_index, chunk, embedding <=> %s::vector AS distance
      FROM policy_chunks
     WHERE embedding_model = %s
     ORDER BY embedding <=> %s::vector
     LIMIT %s
"""


class PolicySearchUnavailable(ServiceUnavailable):
    """The policy store could not be reached while searching."""

    failure_class = "policy_search_unavailable"


def as_pgvector(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


class PolicyRetriever:
    """The Retriever the graph uses in production: nomic embeddings over pgvector."""

    def __init__(
        self,
        connect: Callable[[], psycopg.Connection],
        embedder: Embedder,
        k: int = DEFAULT_K,
        max_distance: float = DEFAULT_MAX_DISTANCE,
    ):
        self.connect = connect
        self.embedder = embedder
        self.k = k
        self.max_distance = max_distance

    def search(self, question: str) -> list[PolicyPassage]:
        vector = as_pgvector(embed_query(self.embedder, question[:MAX_QUERY_CHARS]))
        try:
            with self.connect() as connection:
                rows = connection.execute(SEARCH, (vector, self.embedder.model, vector, self.k)).fetchall()
        except (psycopg.OperationalError, psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as exc:
            # A dropped connection, or a policy store whose schema is not in place yet,
            # is an outage: the run is requeued and, if it persists, ends dead with a
            # visible failure_class -- never a crashed worker and a run stuck running.
            raise PolicySearchUnavailable(f"policy search could not use the database: {exc}") from exc
        return [
            PolicyPassage(document=document, chunk_index=index, text=text, distance=float(distance))
            for document, index, text, distance in rows
            if distance <= self.max_distance
        ]
