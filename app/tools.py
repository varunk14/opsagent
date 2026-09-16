"""
What the agent may propose doing, as the model sees it.

A tool here is a name, a description and a JSON schema, and deliberately nothing
executable: proposing a tool can never run it. What runs is decided elsewhere --
app/executor.py holds the implementations, behind idempotency keys, and
app/run_agent.py decides which proposals it executes (get_order,
escalate_to_human) and which the guardrail judges first (issue_refund: paid, or
put to a person to approve).

The descriptions are written for the model, not for us. Every ambiguity a human
would resolve from context is spelled out instead -- most importantly that money
is in paise, because rupees and paise differ by a factor of a hundred and the
model has no way to guess which we meant.
"""

from collections.abc import Collection
from dataclasses import dataclass

# Rs 10,00,000. A sanity ceiling on what could ever be one refund, not a
# refund policy -- the policy limits live in the database (app/guardrails.py).
MAX_AMOUNT_PAISE = 100_000_000

# An order number, not a sentence. Letters, digits and hyphens, starting with a
# letter or digit. Body text copied into order_id fails this, and so does an
# instruction planted there to reach the planning prompt.
ORDER_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9-]{0,31}$"


@dataclass(frozen=True)
class ToolDescription:
    """A tool as the model sees it. Deliberately holds nothing executable."""

    name: str
    description: str
    parameters: dict


TOOLS: tuple[ToolDescription, ...] = (
    ToolDescription(
        name="get_order",
        description=(
            "Look up one order and what was charged for it. Use this before "
            "deciding anything about money: the customer's account of what they "
            "paid is what they remember, not what the ledger says."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "maxLength": 32,
                    "pattern": ORDER_ID_PATTERN,
                    "description": "The order number, e.g. 4821",
                }
            },
            "required": ["order_id"],
        },
    ),
    ToolDescription(
        name="search_policy",
        description=(
            "Search the refund policy in plain language and get back the "
            "passages that apply. Use this before proposing a refund, so the "
            "decision cites a rule instead of an assumption."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "maxLength": 500,
                    "description": "What you need to know, e.g. 'customer charged twice'",
                }
            },
            "required": ["question"],
        },
    ),
    ToolDescription(
        name="issue_refund",
        description=(
            "Pay money back to the customer. This moves real money and cannot "
            "be undone. Only propose it when an order lookup shows the charge "
            "and a policy passage allows it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "maxLength": 32, "pattern": ORDER_ID_PATTERN},
                "amount_paise": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_AMOUNT_PAISE,
                    "description": (
                        "Whole paise, never rupees and never a decimal. "
                        "Rs 3,600 is 360000."
                    ),
                },
                "reason": {
                    "type": "string",
                    "maxLength": 1000,
                    "description": "Why this refund is owed",
                },
            },
            "required": ["order_id", "amount_paise", "reason"],
        },
    ),
    ToolDescription(
        name="escalate_to_human",
        description=(
            "Hand the case to a person and stop. Use this whenever you are not "
            "sure, when the policy does not cover the situation, or when the "
            "customer is asking for something no tool here can do. Escalating "
            "is a correct outcome, not a failure."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "maxLength": 1000,
                    "description": "What a person needs to decide",
                }
            },
            "required": ["reason"],
        },
    ),
)


def described_argument(name: str, schema: dict, required: bool) -> str:
    """One argument on one line: what to call it, what shape it is, and whether it must be there."""
    shape = str(schema.get("type", "value"))
    says = str(schema.get("description", "")).strip()
    return f"    {name} ({shape}{'' if required else ', optional'})" + (f": {says}" if says else "")


def describe_tools(exclude: Collection[str] = ()) -> str:
    """
    The tool list as it appears in a prompt, leaving out tools that would be redundant.

    Written as prose rather than as the raw JSON schema. The schema's maxLength, pattern, minimum
    and maximum are checked in app/contracts.py whatever a prompt says, so sending them to the model
    buys nothing that is not already guaranteed -- and they were 45 % of the planning prompt, which
    is itself 82 % of everything this agent reads. What stays is what the model cannot work out for
    itself: what each tool does, when to reach for it, the arguments' names and shapes, and the few
    facts a reader would otherwise have to guess, such as money being counted in paise.
    """
    described = []
    for tool in TOOLS:
        if tool.name in exclude:
            continue
        properties = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", ()))
        arguments = [described_argument(name, schema, name in required) for name, schema in properties.items()]
        # The convention is stated once by the prompt that uses this, not repeated per tool.
        lines = [f"- {tool.name}: {tool.description}", *arguments]
        described.append("\n".join(lines))
    return "\n\n".join(described)
