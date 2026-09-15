"""A model that answers from a script, chosen by the TASK line each prompt opens with."""

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
EXTRACTED_4821 = '{"order_id": "4821", "amount_paise": null, "reason": "charged twice"}'
PROPOSED_LOOKUP = (
    '{"tool": "get_order", "args": {"order_id": "4821"}, "confidence": 0.8,'
    ' "reasoning": "confirm both charges first"}'
)
PROPOSED_ESCALATE = (
    '{"tool": "escalate_to_human", "args": {"reason": "status question"},'
    ' "confidence": 0.7, "reasoning": "no tool answers delivery questions"}'
)
