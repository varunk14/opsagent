"""
Asking a local model for something a program can actually use.

Week 0 settled the method and the number. Asked in prose, the model produced
nothing parseable in 5 of 5 attempts. Constrained to JSON and checked against a
schema, it produced usable data in 5 of 5. This module is that result made
reusable, so no node has to rediscover it.

Two things it insists on.

The reply is validated, never trusted. A 3B model asked for JSON will sometimes
return an apology, and sometimes return immaculate JSON that says the refund is
-5 paise. Both are caught in the same place, by the schema.

Every attempt is counted. A prompt that needs three goes cost three calls, and
counting only the one that worked would understate exactly the thing week 9 is
trying to reduce. The replies come back with the answer, and they come back
attached to the exception when there is no answer.
"""

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ValidationError

OLLAMA = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

# Pinned, so a prompt change is the only thing that can move an answer.
DETERMINISTIC = {"temperature": 0, "seed": 0}


@dataclass(frozen=True)
class Reply:
    """One call's worth of output, and what it cost to get it."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class Model(Protocol):
    """Anything that can be asked. Injected everywhere, so tests never dial out."""

    def generate(self, prompt: str) -> Reply: ...


class ModelUnavailable(Exception):
    """The model could not be reached at all."""


class ModelOutputInvalid(Exception):
    """
    Every attempt came back unusable.

    Carries the replies, because the attempts were paid for whether or not they
    worked, and the run should be charged for them.
    """

    def __init__(self, message: str, replies: list[Reply]):
        super().__init__(message)
        self.replies = replies


def complain(original: str, reply: str, error: ValidationError) -> str:
    """
    Ask again, saying what was wrong.

    Re-sending the identical prompt mostly gets the identical answer. The
    complaint is the only thing that makes the next attempt different.
    """
    return (
        f"{original}\n\n"
        "Your previous reply could not be used.\n\n"
        f"You replied:\n{reply}\n\n"
        f"The problem:\n{error}\n\n"
        "Reply again with JSON only. No explanation, no markdown fence."
    )


def structured[T: BaseModel](
    model: Model, prompt: str, schema: type[T], attempts: int = 3
) -> tuple[T, list[Reply]]:
    """
    Ask until the reply fits `schema`, or give up loudly.

    Returns the parsed answer and every reply it took to get there.
    """
    if attempts < 1:
        raise ValueError("structured needs at least one attempt to return anything")

    replies: list[Reply] = []
    asking = prompt

    for _ in range(attempts):
        reply = model.generate(asking)
        replies.append(reply)

        try:
            return schema.model_validate_json(reply.text), replies
        except ValidationError as error:
            asking = complain(prompt, reply.text, error)

    raise ModelOutputInvalid(
        f"{attempts} attempts produced nothing shaped like {schema.__name__}. "
        f"The last reply was: {replies[-1].text!r}",
        replies,
    )


class Ollama:
    """The real thing. Talks to a model running on this machine."""

    def __init__(self, model: str = DEFAULT_MODEL, endpoint: str = OLLAMA):
        self.model = model
        self.endpoint = endpoint

    def generate(self, prompt: str) -> Reply:  # pragma: no cover - network
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    # Ollama's own constraint. Belt and braces with the schema
                    # check, because it guarantees JSON and nothing about shape.
                    "format": "json",
                    "options": DETERMINISTIC,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read()
        except (OSError, TimeoutError) as exc:
            raise ModelUnavailable(f"cannot reach the model at {self.endpoint}: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        payload = json.loads(body)
        return Reply(
            text=payload.get("response", ""),
            prompt_tokens=payload.get("prompt_eval_count", 0),
            completion_tokens=payload.get("eval_count", 0),
            latency_ms=elapsed_ms,
        )
