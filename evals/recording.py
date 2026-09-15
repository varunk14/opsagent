"""
Recorded model replies and embeddings, and the stand-ins that record or replay them.

A reply is keyed by a hash of the model's name and the exact prompt; an embedding by a
hash of the embedding model's name and the exact text. So a prompt file that changes, a
case that changes, or a different model finds nothing to replay, and replay says so with
RecordingMissing instead of guessing or calling a model. Vectors are kept as the exact
64-bit floats they came back as, so retrieval ranks passages identically on replay.

What a reply said and what it cost in tokens is kept; how long it took is not. Latency
belongs to the machine that recorded it, would rewrite every line of the file on every
recording, and says nothing true on replay -- so a replayed reply reports none.

Recording can reuse what is already recorded instead of asking again. That is safe
because, at temperature 0 with a fixed seed, the model gives the same reply to the same
prompt byte for byte; it is what lets a long recording that stopped partway carry on.

The file holds hashes of prompts and texts, never the prompts themselves. It does hold
what the model said, and a reply can repeat a customer's words -- which is acceptable
only because every golden case is fictional. It is saved sorted, so recording the same
things in any order gives the same bytes.

A recordings file is an input a pull request can change, so it is read with limits: a
file, a reply or a vector larger than anything real is refused before it is decoded.
And it is trusted, not verified: the key binds a reply to its prompt, not to what the
model really said, so a hand-edited reply would replay as if it were real. The check
for that is re-recording live on a machine with the model, where replies are exact.
"""

import base64
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from app.embeddings import EMBEDDING_MODEL, Embedder
from app.llm import DEFAULT_MODEL, MAX_RESPONSE_BYTES, Model, Reply

RECORD_COMMAND = "python -m evals record"

# Far above 150 cases' worth of replies and vectors, far below what could hurt a CI runner.
MAX_RECORDINGS_BYTES = 50_000_000
# Real embeddings here have 768 dimensions; nothing legitimate comes near this.
MAX_VECTOR_FLOATS = 8_192
# No reply can be longer than the model client would have accepted.
MAX_REPLY_CHARS = MAX_RESPONSE_BYTES


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


def checked_vector(packed: str, where: str) -> str:
    """`packed`, once it is known to be whole 64-bit floats and no longer than any real embedding."""
    if len(packed) * 3 // 4 > MAX_VECTOR_FLOATS * 8 + 2:
        raise ValueError(f"{where}: a vector longer than {MAX_VECTOR_FLOATS} floats is not an embedding")
    raw = base64.b64decode(packed, validate=True)
    if len(raw) % 8 or len(raw) // 8 > MAX_VECTOR_FLOATS:
        raise ValueError(f"{where}: a vector must be whole 64-bit floats, at most {MAX_VECTOR_FLOATS} of them")
    return packed


class RecordingMissing(LookupError):
    """Replay was asked for something that was never recorded. Record again rather than guess."""


@dataclass(frozen=True)
class RecordedReply:
    model: str
    task: str
    text: str
    prompt_tokens: int
    completion_tokens: int


def replayed(found: RecordedReply) -> Reply:
    return Reply(text=found.text, prompt_tokens=found.prompt_tokens, completion_tokens=found.completion_tokens, latency_ms=0)


class Recordings:
    """Every recorded reply and vector, in memory, with the file they are kept in."""

    def __init__(self) -> None:
        self.replies: dict[str, RecordedReply] = {}
        self.vectors: dict[str, tuple[str, str]] = {}  # key -> (model, packed vector)

    @classmethod
    def load(cls, path: Path) -> "Recordings":
        """What `path` holds; a file that does not exist yet holds nothing. Anything larger than real is refused."""
        recordings = cls()
        if not path.exists():
            return recordings
        size = path.stat().st_size
        if size > MAX_RECORDINGS_BYTES:
            raise ValueError(f"{path.name} is too large to be a recording: {size} bytes, at most {MAX_RECORDINGS_BYTES}")
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            where = f"{path.name} line {number}"
            if entry["kind"] == "reply":
                if len(entry["text"]) > MAX_REPLY_CHARS:
                    raise ValueError(f"{where}: a reply longer than {MAX_REPLY_CHARS} characters was never a model's")
                recordings.replies[entry["key"]] = RecordedReply(
                    model=entry["model"],
                    task=entry["task"],
                    text=entry["text"],
                    prompt_tokens=entry["prompt_tokens"],
                    completion_tokens=entry["completion_tokens"],
                )
            else:
                recordings.vectors[entry["key"]] = (entry["model"], checked_vector(entry["vector"], where))
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
        return replayed(found)


class RecordingModel:
    """Asks the real model, and keeps what it said. With `reuse`, a prompt already recorded is not asked again."""

    def __init__(self, inner: Model, recordings: Recordings, model: str | None = None, reuse: bool = False) -> None:
        self.inner = inner
        self.recordings = recordings
        self.model = model or str(getattr(inner, "model", DEFAULT_MODEL))
        self.reuse = reuse

    def generate(self, prompt: str) -> Reply:
        key = reply_key(self.model, prompt)
        found = self.recordings.replies.get(key) if self.reuse else None
        if found is not None:
            return replayed(found)
        reply = self.inner.generate(prompt)
        self.recordings.replies[key] = RecordedReply(
            model=self.model,
            task=task_of(prompt),
            text=reply.text,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
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
    """Asks the real embedding model, and keeps every vector. With `reuse`, only texts not yet recorded are asked."""

    def __init__(self, inner: Embedder, recordings: Recordings, reuse: bool = False) -> None:
        self.inner = inner
        self.recordings = recordings
        self.model = str(getattr(inner, "model", EMBEDDING_MODEL))
        self.reuse = reuse

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [embedding_key(self.model, text) for text in texts]
        wanted = [text for text, key in zip(texts, keys, strict=True) if not (self.reuse and key in self.recordings.vectors)]
        if wanted:
            for text, vector in zip(wanted, self.inner.embed(wanted), strict=True):
                self.recordings.vectors[embedding_key(self.model, text)] = (self.model, pack(vector))
        return [unpack(self.recordings.vectors[key][1]) for key in keys]
