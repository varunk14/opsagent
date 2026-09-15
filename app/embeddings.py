"""
Turning text into vectors with a local embedding model.

A vector that cannot be trusted is refused rather than stored: one of the wrong
length would not fit the vector(768) column, and one from a malformed reply
would quietly rank every later search wrong. The same boundary rules as model
replies apply -- capped response size, strict parsing, no silent defaults.

nomic-embed-text is trained with task prefixes. Passages are embedded as
"search_document: ..." and questions as "search_query: ..."; leaving the
prefixes off measurably degrades retrieval.
"""

import json
import math
import urllib.request
from typing import Protocol

from app.llm import ModelUnavailable, read_capped

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSIONS = 768  # must match policy_chunks.embedding vector(768)
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
OLLAMA_EMBED = "http://localhost:11434/api/embed"
# A 768-number vector is roughly 15 KB of JSON; this allows large batches, not runaway replies.
MAX_EMBED_RESPONSE_BYTES = 20_000_000


class Embedder(Protocol):
    """Anything that turns texts into vectors. Injected, so tests never need Ollama."""

    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def parse_embeddings(body: bytes, expected: int) -> list[list[float]]:
    """An Ollama /api/embed body as vectors, or a refusal saying what was wrong."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ModelUnavailable(f"the embedding response could not be read as JSON: {exc}") from exc

    vectors = payload.get("embeddings") if isinstance(payload, dict) else None
    if not isinstance(vectors, list):
        raise ModelUnavailable("the embedding response carries no embeddings")
    if len(vectors) != expected:
        raise ModelUnavailable(f"expected {expected} embeddings, got {len(vectors)}")

    for vector in vectors:
        size = len(vector) if isinstance(vector, list) else 0
        if size != EMBEDDING_DIMENSIONS:
            raise ModelUnavailable(f"an embedding has {size} dimensions, not {EMBEDDING_DIMENSIONS}")
        if not all(
            isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
            for value in vector
        ):
            raise ModelUnavailable("an embedding contains non-finite numbers")
    return vectors


def embed_documents(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Vectors for policy passages. Nothing to embed means no call at all."""
    if not texts:
        return []
    return embedder.embed([DOCUMENT_PREFIX + text for text in texts])


def embed_query(embedder: Embedder, text: str) -> list[float]:
    """The vector for one search question."""
    return embedder.embed([QUERY_PREFIX + text])[0]


class OllamaEmbedder:
    """The real embedder: nomic-embed-text through the local Ollama server."""

    def __init__(self, model: str = EMBEDDING_MODEL, endpoint: str = OLLAMA_EMBED):
        self.model = model
        self.endpoint = endpoint

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - network
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = read_capped(response, MAX_EMBED_RESPONSE_BYTES)
        except (OSError, TimeoutError) as exc:
            raise ModelUnavailable(f"cannot reach the embedding model at {self.endpoint}: {exc}") from exc
        return parse_embeddings(body, expected=len(texts))
