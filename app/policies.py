"""
Policy documents as retrievable passages, stored with their embeddings.

Each section of a document becomes a passage that carries its document title
and heading, so a passage retrieved on its own still says what it is about.
Long sections split on paragraph boundaries.

Ingest is idempotent. It runs on every deploy, so a passage whose text and
embedding model have not changed is never embedded again. Embeddings are
computed with no transaction open, since model calls take seconds, and all
writes then happen in one short transaction.

Run:  .venv/bin/python -m app.policies
"""

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.pq import TransactionStatus

from app.db import apply_migrations, connect
from app.embeddings import Embedder, OllamaEmbedder, embed_documents

MAX_CHUNK_CHARS = 800
# One key for every ingest. Overlapping deploys serialise on it instead of
# racing to rewrite the same passages.
INGEST_LOCK = 7_318_236_101
POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"

INSERT_CHUNK = """
    INSERT INTO policy_chunks (document, chunk_index, chunk, content_hash, embedding_model, embedding)
    VALUES (%s, %s, %s, %s, %s, %s::vector)
"""


@dataclass(frozen=True)
class Chunk:
    document: str
    index: int
    text: str
    content_hash: str


@dataclass(frozen=True)
class IngestSummary:
    added: int  # passages embedded and stored
    unchanged: int  # documents left exactly as they were
    removed: int  # documents no longer present, and their passages deleted


def load_policies(directory: Path) -> dict[str, str]:
    """
    Every markdown policy in `directory`, keyed by file name, in a stable order.

    Symlinks are refused. The planner treats this text as the rules it must
    follow, so it has to be text that lives in the repository and goes through
    review -- not whatever a link happens to point at on the machine running ingest.
    """
    policies: dict[str, str] = {}
    for path in sorted(directory.glob("*.md")):
        if path.is_symlink():
            raise ValueError(f"{path.name} is a symlink; policy text must live in the repository")
        policies[path.stem] = path.read_text(encoding="utf-8")
    return policies


def _pieces(paragraphs: list[str], max_chars: int) -> list[str]:
    """Pack paragraphs into pieces no longer than max_chars, splitting only oversized paragraphs."""
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for part in [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]:
            candidate = f"{current}\n\n{part}" if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                pieces.append(current)
                current = part
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(document: str, text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """
    Split one markdown policy into titled passages.

    Each section records the title in force when it started, so a later title
    cannot rename earlier sections. Lines inside ``` code blocks are body text,
    never headings.
    """
    title = document
    sections: list[tuple[str, str | None, list[str]]] = [(title, None, [])]
    in_code_block = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            sections[-1][2].append(line)
        elif not in_code_block and line.startswith("# "):
            title = line[2:].strip()
            sections.append((title, None, []))
        elif not in_code_block and line.startswith("## "):
            sections.append((title, line[3:].strip(), []))
        else:
            sections[-1][2].append(line)

    chunks: list[Chunk] = []
    for section_title, heading, lines in sections:
        paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
        label = f"{section_title} — {heading}" if heading else section_title
        for piece in _pieces(paragraphs, max_chars):
            passage = f"{label}\n\n{piece}"
            chunks.append(
                Chunk(
                    document=document,
                    index=len(chunks),
                    text=passage,
                    content_hash=hashlib.sha256(passage.encode()).hexdigest(),
                )
            )
    return chunks


def _as_pgvector(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def ingest(
    connection: psycopg.Connection, embedder: Embedder, documents: Mapping[str, str]
) -> IngestSummary:
    """Bring policy_chunks in line with `documents`, embedding only what changed."""
    if connection.pgconn.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError(
            "ingest commits, so it needs its own transaction: call it on a "
            "connection with no work already open"
        )

    # A session-level lock, held across the read, the embedding and the write.
    # Without it two deploys both see the same documents as changed and one
    # crashes on the unique index. With it the second waits, then finds nothing to do.
    with connection.transaction():
        connection.execute("SELECT pg_advisory_lock(%s)", (INGEST_LOCK,))
    try:
        return _ingest_locked(connection, embedder, documents)
    finally:
        with connection.transaction():
            connection.execute("SELECT pg_advisory_unlock(%s)", (INGEST_LOCK,))


def _ingest_locked(
    connection: psycopg.Connection, embedder: Embedder, documents: Mapping[str, str]
) -> IngestSummary:
    with connection.transaction():
        stored: dict[str, list[tuple[str, str]]] = {}
        for document, content_hash, model in connection.execute(
            "SELECT document, content_hash, embedding_model FROM policy_chunks ORDER BY document, chunk_index"
        ).fetchall():
            stored.setdefault(document, []).append((content_hash, model))

    unchanged = 0
    changed: list[tuple[str, list[Chunk], list[list[float]]]] = []
    for document, text in documents.items():
        chunks = chunk_markdown(document, text)
        if stored.get(document) == [(chunk.content_hash, embedder.model) for chunk in chunks]:
            unchanged += 1
            continue
        changed.append((document, chunks, embed_documents(embedder, [chunk.text for chunk in chunks])))

    gone = sorted(set(stored) - set(documents))
    with connection.transaction():
        for document, chunks, vectors in changed:
            connection.execute("DELETE FROM policy_chunks WHERE document = %s", (document,))
            for chunk, vector in zip(chunks, vectors, strict=True):
                connection.execute(
                    INSERT_CHUNK,
                    (document, chunk.index, chunk.text, chunk.content_hash, embedder.model, _as_pgvector(vector)),
                )
        for document in gone:
            connection.execute("DELETE FROM policy_chunks WHERE document = %s", (document,))

    return IngestSummary(
        added=sum(len(chunks) for _, chunks, _ in changed), unchanged=unchanged, removed=len(gone)
    )


def main() -> int:  # pragma: no cover - the interactive driver
    with connect() as connection:
        apply_migrations(connection)
        summary = ingest(connection, OllamaEmbedder(), load_policies(POLICY_DIR))
    print(f"  passages added {summary.added}, documents unchanged {summary.unchanged}, removed {summary.removed}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
