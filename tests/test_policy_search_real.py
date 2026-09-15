"""
Week 3's done-when, on the real embedding model and the real policy documents.

"charged twice" must find the duplicate-payment policy, whose text contains
neither word (tests/test_policy_chunking.py guards that), among unrelated
policies that could be ranked above it.

Opt-in: CI has no Ollama. Run locally with Ollama and nomic-embed-text:

    OPSAGENT_REAL_MODEL=1 .venv/bin/python -m pytest tests/test_policy_search_real.py --no-cov
"""

import os

import psycopg
import pytest

from app.embeddings import OllamaEmbedder
from app.policies import POLICY_DIR, ingest, load_policies
from app.retrieval import PolicyRetriever

pytestmark = [
    pytest.mark.db,
    pytest.mark.llm,
    pytest.mark.skipif(
        os.environ.get("OPSAGENT_REAL_MODEL") != "1",
        reason="real-model test: set OPSAGENT_REAL_MODEL=1 with Ollama and nomic-embed-text running",
    ),
]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("I was charged twice for my order", "duplicate-payments"),
        ("where is my parcel", "delivery-times"),
        ("the item arrived broken", "damaged-items"),
    ],
)
def test_the_right_policy_is_found_by_meaning(fresh_database, question, expected):
    embedder = OllamaEmbedder()
    with psycopg.connect(fresh_database) as connection:
        ingest(connection, embedder, load_policies(POLICY_DIR))

    hits = PolicyRetriever(lambda: psycopg.connect(fresh_database), embedder).search(question)

    assert hits, f"nothing within the distance cutoff for {question!r}"
    assert hits[0].document == expected
