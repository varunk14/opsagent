"""
What the agent may propose doing, described but not built.

This week the model is told these four exist and asked which it would use. None
of them are connected to anything, and the guarantee is structural rather than a
promise in prose: a tool here is a name, a description and a JSON schema. There
is no implementation in the registry to call, so proposing a refund cannot issue
one, however wrong the model is.

Week 4 attaches real behaviour, behind idempotency keys.

The descriptions are written for the model, not for us. Every ambiguity a human
would resolve from context is spelled out instead -- most importantly that money
is in paise, because rupees and paise differ by a factor of a hundred and the
model has no way to guess which we meant.
"""

from dataclasses import dataclass

# Rs 10,00,000. A sanity ceiling on what could ever be one refund, not a
# refund policy -- policy limits need the order record and arrive in week 5.
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


def describe_tools() -> str:
    """The tool list as it appears in a prompt."""
    return "\n\n".join(
        f"- {tool.name}: {tool.description}\n  arguments: {tool.parameters}"
        for tool in TOOLS
    )
