"""
Looking up an order: the model asks, your code decides.

Validation produced a typed object. It still knows nothing about Priya's order:
amount_paise was None, because the email never said a rupee amount.

To find out, the agent needs to LOOK SOMETHING UP. That is tool calling.

The single most important idea in this file:

    THE MODEL NEVER TOUCHES YOUR DATA.
    It emits a request. Your code decides whether to honour it.

Everything about agent safety follows from keeping that boundary.

Also introduced here:
  - the LOOP: think -> act -> observe -> think again
  - a STEP CAP, because a loop without one runs forever (a real failure mode)
  - rejecting INVENTED tools, because the model will ask for ones that don't exist

Run:  .venv/bin/python experiments/order_lookup_agent.py
"""

import json
import sys
import urllib.error
import urllib.request
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

OLLAMA_URL = "http://localhost:11434/api/generate"

# Deliberately NOT llama3.2 (3B), used for one-shot extraction elsewhere.
# 3B is fine at one-shot extraction and poor at multi-step planning.
# Choosing the model by the job is the seed of the routing ladder we build later.
MODEL = "llama3.1:8b"

MAX_STEPS = 5          # the guard against an agent that loops forever
MAX_RETRIES = 2        # per step, when the model emits an unusable tool call


# ---------------------------------------------------------------------------
# THE "DATABASE"
# A dict for now. Postgres arrives when we need durability, not before.
# Money is integer paise. Dates are ISO YYYY-MM-DD.
# ---------------------------------------------------------------------------
ORDERS = {
    "4821": {
        "order_id": "4821",
        "customer_email": "priya@example.com",
        "status": "delivered",
        "charges": [
            {"charge_id": "ch_001", "amount_paise": 360000, "settled_on": "2026-09-08"},
            {"charge_id": "ch_002", "amount_paise": 360000, "settled_on": "2026-09-08"},
        ],
    }
}


# ---------------------------------------------------------------------------
# THE TOOLS
# Note what is missing: there is no refund tool yet. This agent can only LOOK
# and ESCALATE. We add the dangerous one in duplicate_refund_bug.py, broken on purpose, so you
# can watch it fail before we fix it.
# ---------------------------------------------------------------------------
def tool_get_order(order_id: str | None) -> dict:
    if not order_id:
        return {"error": "order_id is required"}
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"no order found with id {order_id}"}
    return order


def tool_escalate(reason: str | None) -> dict:
    return {"escalated": True, "reason": reason or "(no reason given)"}


class ToolCall(BaseModel):
    """What the model is allowed to ask for. Anything else is rejected."""

    model_config = ConfigDict(extra="forbid")

    # A Literal is a whitelist. If the model invents "issue_refund" or
    # "delete_customer", validation fails here and the call never runs.
    tool: Literal["get_order", "escalate_to_human", "finish"]

    # Flat arguments, not nested. Small models handle flat schemas far better.
    order_id: str | None = None
    reason: str | None = None
    summary: str | None = None


SYSTEM = """You are a support agent working on a customer refund request.

You may call exactly one tool per turn. Reply with ONLY a JSON object.

Tools available:
  {"tool": "get_order", "order_id": "<id>"}
      Look up an order and its charges.
  {"tool": "escalate_to_human", "reason": "<why>"}
      Hand off when you cannot safely proceed.
  {"tool": "finish", "summary": "<what you found and what should happen>"}
      Call this when you have enough information. Do not call it before that.

Rules:
- Do not invent tools. Only the three above exist.
- Amounts are in paise. 100 paise = 1 rupee.
- If the order shows two identical charges on the same day, that is a duplicate charge.
"""

TASK = """Customer email:
Hi, I think I was charged twice for order #4821 last Tuesday.
Can you refund one of them? Thanks, Priya

Work out what happened and what should be done."""


def ask_model(prompt: str) -> str:  # pragma: no cover
    body = json.dumps(
        {"model": MODEL, "prompt": prompt, "stream": False, "format": "json"}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"]


def get_tool_call(prompt: str) -> ToolCall | None:  # pragma: no cover
    """Ask for one tool call. On a bad reply, show the model its own error and retry."""
    feedback = ""
    for attempt in range(1, MAX_RETRIES + 1):
        raw = ask_model(prompt + feedback)
        print(f"    model said: {raw.strip()[:160]}")
        try:
            return ToolCall.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            msg = str(e).split("\n")[0]
            print(f"    REJECTED ({attempt}/{MAX_RETRIES}): {msg}")
            # Self-correction: hand the error back so it can try again.
            feedback = (
                f"\n\nYour last reply was invalid: {msg}\n"
                "Reply with valid JSON using only the listed tools."
            )
    return None


def main() -> None:  # pragma: no cover
    history: list[str] = []

    for step in range(1, MAX_STEPS + 1):
        print(f"\n{'=' * 70}\nSTEP {step}\n{'=' * 70}")

        prompt = SYSTEM + "\n\n" + TASK
        if history:
            prompt += "\n\nWhat you have done so far:\n" + "\n".join(history)
        prompt += "\n\nYour next tool call:"

        call = get_tool_call(prompt)
        if call is None:
            print("\n  Model could not produce a valid tool call. Escalating.")
            print("  (In production this is the 'tool_misuse' failure category.)")
            return

        print(f"\n  -> tool: {call.tool}")

        if call.tool == "finish":
            print(f"\n  FINISHED after {step} step(s)")
            print(f"  summary: {call.summary}")
            return

        if call.tool == "escalate_to_human":
            result = tool_escalate(call.reason)
            print(f"  ESCALATED: {result['reason']}")
            return

        # get_order. YOUR code runs it, not the model.
        result = tool_get_order(call.order_id)
        print(f"  result: {json.dumps(result)}")

        history.append(
            f"- called {call.tool}(order_id={call.order_id!r}) -> {json.dumps(result)}"
        )

    # Falling out of the loop is itself a result worth noticing.
    print(f"\n  HIT THE STEP CAP ({MAX_STEPS}) without finishing.")
    print("  Without this cap the agent would run until you killed it.")
    print("  In production this is the 'loop' failure category.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"Could not reach Ollama at {OLLAMA_URL}: {e}")
        print("Is it running?   ollama serve")
        sys.exit(1)
