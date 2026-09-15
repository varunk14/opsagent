"""
Recorded model replies and embeddings, and the stand-ins that record or replay them.

A reply is keyed by a hash of the model's name and the exact prompt; an embedding by a
hash of the embedding model's name and the exact text. So a prompt file that changes, a
case that changes, or a different model finds nothing to replay, and replay says so with
RecordingMissing instead of guessing or calling a model. Vectors are kept as the exact
64-bit floats they came back as, so retrieval ranks passages identically on replay.

The file holds hashes, never prompts or texts: a customer's words stay in the golden
set, where they are reviewed, and the recording diff shows only what the model said.
It is saved sorted, so recording the same things in any order gives the same bytes.
"""

import base64
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from app.embeddings import EMBEDDING_MODEL, Embedder
from app.llm import DEFAULT_MODEL, Model, Reply

RECORD_COMMAND = "python -m evals record"


def reply_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"reply\n{model}\n{prompt}".encode()).hexdigest()


def embedding_key(model: str, text: str) -> str:
    return hashlib.sha256(f"embedding\n{model}\n{text}".encode()).hexdigest()


def task_of(prompt: str) -> str:
    """The TASK line every agent prompt opens with, or 'unknown'."""
    first = prompt.split("\n", 1)[0]
    return first.removeprefix("TASK:").strip() if first.startswith("TASK:") else "unknown"


def pack(vector: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}d", *vector)).decode()


def unpack(packed: str) -> list[float]:
    raw = base64.b64decode(packed)
    return list(struct.unpack(f"<{len(raw) // 8}d", raw))


class RecordingMissing(LookupError):
    """Replay was asked for something that was never recorded. Record again rather than guess."""


@dataclass(frozen=True)
class RecordedReply:
    model: str
    task: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class Recordings:
    """Every recorded reply and vector, in memory, with the file they are kept in."""

    def __init__(self) -> None:
        self.replies: dict[str, RecordedReply] = {}
        self.vectors: dict[str, tuple[str, str]] = {}  # key -> (model, packed vector)

    @classmethod
    def load(cls, path: Path) -> "Recordings":
        """What `path` holds; a file that does not exist yet holds nothing."""
        recordings = cls()
        if not path.exists():
            return recordings
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["kind"] == "reply":
                recordings.replies[entry["key"]] = RecordedReply(
                    model=entry["model"],
                    task=entry["task"],
                    text=entry["text"],
                    prompt_tokens=entry["prompt_tokens"],
                    completion_tokens=entry["completion_tokens"],
                    latency_ms=entry["latency_ms"],
                )
            else:
                recordings.vectors[entry["key"]] = (entry["model"], entry["vector"])
        return recordings

    def save(self, path: Path) -> None:
        entries = [
            ("embedding", key, {"kind": "embedding", "key": key, "model": model, "vector": packed})
            for key, (model, packed) in self.vectors.items()
        ] + [
            (
                "reply",
                key,
                {
                    "kind": "reply",
                    "key": key,
                    "model": reply.model,
                    "task": reply.task,
                    "text": reply.text,
                    "prompt_tokens": reply.prompt_tokens,
                    "completion_tokens": reply.completion_tokens,
                    "latency_ms": reply.latency_ms,
                },
            )
            for key, reply in self.replies.items()
        ]
        entries.sort(key=lambda entry: (entry[0], entry[1]))
        path.write_text("".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for _, _, entry in entries))


class RecordedModel:
    """Answers only from recordings. Never calls a model."""

    def __init__(self, recordings: Recordings, model: str = DEFAULT_MODEL) -> None:
        self.recordings = recordings
        self.model = model

    def generate(self, prompt: str) -> Reply:
        key = reply_key(self.model, prompt)
        found = self.recordings.replies.get(key)
        if found is None:
            raise RecordingMissing(
                f"no recorded {self.model} reply for a {task_of(prompt)} prompt ({key[:12]}): a prompt or a case "
                f"changed since it was recorded; run `{RECORD_COMMAND}` on a machine with the model"
            )
        return Reply(
            text=found.text,
            prompt_tokens=found.prompt_tokens,
            completion_tokens=found.completion_tokens,
            latency_ms=found.latency_ms,
        )


class RecordingModel:
    """Asks the real model, and keeps what it said."""

    def __init__(self, inner: Model, recordings: Recordings, model: str | None = None) -> None:
        self.inner = inner
        self.recordings = recordings
        self.model = model or str(getattr(inner, "model", DEFAULT_MODEL))

    def generate(self, prompt: str) -> Reply:
        reply = self.inner.generate(prompt)
        self.recordings.replies[reply_key(self.model, prompt)] = RecordedReply(
            model=self.model,
            task=task_of(prompt),
            text=reply.text,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            latency_ms=reply.latency_ms,
        )
        return reply


class RecordedEmbedder:
    """Vectors only from recordings. Never calls a model."""

    def __init__(self, recordings: Recordings, model: str = EMBEDDING_MODEL) -> None:
        self.recordings = recordings
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            key = embedding_key(self.model, text)
            found = self.recordings.vectors.get(key)
            if found is None:
                raise RecordingMissing(
                    f"no recorded {self.model} embedding for a text ({key[:12]}): a policy, a prompt or a case "
                    f"changed since it was recorded; run `{RECORD_COMMAND}` on a machine with the model"
                )
            vectors.append(unpack(found[1]))
        return vectors


class RecordingEmbedder:
    """Asks the real embedding model, and keeps every vector."""

    def __init__(self, inner: Embedder, recordings: Recordings) -> None:
        self.inner = inner
        self.recordings = recordings
        self.model = str(getattr(inner, "model", EMBEDDING_MODEL))

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.inner.embed(texts)
        for text, vector in zip(texts, vectors, strict=True):
            self.recordings.vectors[embedding_key(self.model, text)] = (self.model, pack(vector))
        return vectors
