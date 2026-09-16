"""A model that answers from a script, chosen by the TASK line each prompt opens with."""

import json
from email.message import EmailMessage

from app.llm import ModelUnavailable, Reply

# A scripted reply that means: the model is unreachable for this call.
OUTAGE = "__MODEL_UNAVAILABLE__"


class ScriptedModel:
    """
    Replies per task. A list is consumed in order, and its last entry repeats,
    so a test can script "garbage, then valid" or "garbage forever".

    Keyed on each prompt's first line (TASK: classify, extract, plan). Renaming a
    task in app/graph/prompts.py fails loudly here with a KeyError, not silently.
    """

    def __init__(self, **replies: str | list[str]):
        self.replies = {task: [r] if isinstance(r, str) else list(r) for task, r in replies.items()}
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> Reply:
        self.prompts.append(prompt)
        queue = self.replies[task_of(prompt)]
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        if text == OUTAGE:
            raise ModelUnavailable("scripted outage")
        return Reply(text=text, prompt_tokens=10, completion_tokens=5, latency_ms=1)

    def tasks(self) -> list[str]:
        return [task_of(prompt) for prompt in self.prompts]


def task_of(prompt: str) -> str:
    return prompt.split("\n", 1)[0].removeprefix("TASK:").strip()


CLASSIFIED_DUPLICATE = '{"intent": "duplicate_charge", "confidence": 0.9, "reasoning": "charged twice"}'
CLASSIFIED_STATUS = '{"intent": "order_status", "confidence": 0.9, "reasoning": "asks when"}'
CLASSIFIED_REFUND_REQUEST = '{"intent": "refund_request", "confidence": 0.9, "reasoning": "changed their mind"}'
EXTRACTED_4821 = '{"order_id": "4821", "amount_paise": null, "reason": "charged twice"}'
EXTRACTED_3310 = '{"order_id": "3310", "amount_paise": null, "reason": "charged twice"}'
PROPOSED_LOOKUP_3310 = (
    '{"tool": "get_order", "args": {"order_id": "3310"}, "confidence": 0.8,'
    ' "reasoning": "confirm both charges first"}'
)
PROPOSED_LOOKUP = (
    '{"tool": "get_order", "args": {"order_id": "4821"}, "confidence": 0.8,'
    ' "reasoning": "confirm both charges first"}'
)
PROPOSED_REFUND = (
    '{"tool": "issue_refund", "args": {"order_id": "4821", "amount_paise": 360000,'
    ' "reason": "the ledger shows two charges of 360000"}, "confidence": 0.9,'
    ' "reasoning": "one of the two charges is a duplicate"}'
)


def proposed_refund(amount_paise: int, confidence: str = "0.9", order_id: str = "4821") -> str:
    """A refund for any amount at any confidence, as the model would send it."""
    return json.dumps(
        {
            "tool": "issue_refund",
            "args": {"order_id": order_id, "amount_paise": amount_paise, "reason": "the ledger shows a duplicate"},
            "confidence": float(confidence),
            "reasoning": "one of the two charges is a duplicate",
        }
    )


PROPOSED_ESCALATE = (
    '{"tool": "escalate_to_human", "args": {"reason": "status question"},'
    ' "confidence": 0.7, "reasoning": "no tool answers delivery questions"}'
)


class FakeEmbedder:
    """
    Deterministic vectors from a hash of each text: identical text, identical
    vector. It carries no meaning, so it tests plumbing, never retrieval quality.
    """

    model = "fake-embed"

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [vector_for(text, self.dimensions) for text in texts]


def vector_for(text: str, dimensions: int = 768) -> list[float]:
    import hashlib

    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        counter += 1
    return values[:dimensions]


class FakeRetriever:
    """Returns fixed policy passages and remembers every question it was asked."""

    def __init__(self, passages=None):
        from app.graph.state import PolicyPassage

        self.passages = (
            passages
            if passages is not None
            else [
                PolicyPassage(
                    document="duplicate-payments",
                    chunk_index=1,
                    text="Duplicate payments — What we do\n\nThe duplicate amount is returned in full.",
                    distance=0.33,
                )
            ]
        )
        self.questions: list[str] = []

    def search(self, question: str):
        self.questions.append(question)
        return list(self.passages)


# --- a mailbox, without a mail server -------------------------------------------------------


def an_email(
    *,
    sender: str = "priya@example.com",
    subject: str | None = "Charged twice for order #4821",
    body: str = "Hi, I think I was charged twice for order #4821.",
    message_id: str | None = "<abc123@example.com>",
    date: str = "Tue, 15 Sep 2026 09:15:00 +0000",
) -> bytes:
    """One plain-text email, as bytes off the wire."""
    message = EmailMessage()
    message["From"] = sender
    if subject is not None:
        message["Subject"] = subject
    if message_id is not None:
        message["Message-ID"] = message_id
    message["Date"] = date
    message.set_content(body)
    return message.as_bytes()


class FakeMailbox:
    """
    imaplib's shape, as much of it as the adapter uses.

    No network, because what is under test is our parsing and our ordering, not imaplib -- and a
    test that needs a real mailbox is a test nobody runs. It records what was marked read, which is
    the assertion most of these tests are actually making.
    """

    def __init__(self, messages: dict[bytes, bytes], *, seen: list[bytes] | None = None) -> None:
        self.messages = messages
        self.seen = seen if seen is not None else []
        self.selected: str | None = None
        self.logged_out = False

    def select(self, mailbox: str):
        self.selected = mailbox
        return "OK", [str(len(self.messages)).encode()]

    def search(self, charset, *criteria):
        return "OK", [b" ".join(self.messages)]

    def fetch(self, number: bytes, parts: str):
        return "OK", [(b"", self.messages[number])]

    def store(self, number: bytes, command: str, flags: str):
        self.seen.append(number)
        return "OK", [b""]

    def logout(self):
        self.logged_out = True
        return "BYE", [b""]
