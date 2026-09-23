"""
The golden set: labelled cases the agent is scored against.

Each line of golden.jsonl is one case: a customer message exactly as the intake would
receive it, and what should have happened to it. A label states the correct outcome under
the store's policies, the guardrail and the fictional ledger in ledger.json -- never what
a model happened to do. Whether a case escalates is read from its outcome rather than
labelled a second time, so the two can never disagree.

The contract refuses a case whose labels contradict each other -- a category from the
other kind, a refund on a case handed to a person, a refund paid on its own at or over
the limit -- so anything that loads cases gets only ones that can be scored.
"""

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.contracts import IncomingMessage, Intent
from app.tools import ORDER_ID_PATTERN

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN = EVALS_DIR / "golden.jsonl"
LEDGER = EVALS_DIR / "ledger.json"

SMOKE_SIZE = 27
# The automatic-refund limit migration 006 starts with; a test holds the two equal.
DEFAULT_LIMIT_PAISE = 500_000

NORMAL_CATEGORIES = frozenset(
    {
        "duplicate_charge",  # two identical charges, one refunded on its own
        "duplicate_over_limit",  # two identical charges at or over the limit: waits for approval
        "duplicate_not_confirmed",  # a claimed duplicate the ledger does not show
        "cancellation_before_dispatch",
        "cancellation_after_dispatch",
        "change_of_mind",
        "damaged_item",
        "order_status",
        "payment_question",
        "general",
    }
)
ADVERSARIAL_CATEGORIES = frozenset(
    {
        "prompt_injection",  # instructions to the agent inside the message
        "wrong_owner",  # asks about an order someone else placed
        "inflated_amount",  # claims more than was charged
        "unknown_order",  # an order that does not exist
        "multiple_orders",  # several orders in one message
        "fake_policy",  # quotes a rule the store does not have
        "confidence_pressure",  # insists on certainty to push a payment through
        "telegram_no_account",  # a handle, which owns no order
        "garbled",  # too little or too broken to act on
    }
)


class Kind(StrEnum):
    NORMAL = "normal"
    ADVERSARIAL = "adversarial"


class Outcome(StrEnum):
    """Where a correctly handled run comes to rest."""

    REFUNDED = "refunded"  # paid on its own, under the limit
    WAITING_APPROVAL = "waiting_approval"  # a refund is owed, but at or over the limit
    HANDED_OVER = "handed_over"  # no refund the agent may make; a person decides


class Expected(BaseModel):
    """What should have happened."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: Intent
    order_id: str | None = Field(default=None, pattern=ORDER_ID_PATTERN)
    # What the customer says they paid or want back, if they say it; not what is owed.
    stated_amount_paise: int | None = Field(default=None, ge=0, strict=True)
    outcome: Outcome
    # The refund owed, paid or put to approval. None when nothing may be paid.
    refund_paise: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def refund_fits_the_outcome(self) -> Self:
        """The refund and the outcome say the same thing, under the limit the guardrail starts with."""
        refund = self.refund_paise
        if self.outcome is Outcome.HANDED_OVER:
            if refund is not None:
                raise ValueError("a case handed to a person owes no refund, so refund_paise must be null")
        elif refund is None:
            raise ValueError(f"a {self.outcome} case needs the refund_paise it owes")
        elif self.outcome is Outcome.REFUNDED and refund >= DEFAULT_LIMIT_PAISE:
            raise ValueError(f"a refund of {refund} paise is not under the limit, so it is not paid on its own")
        elif self.outcome is Outcome.WAITING_APPROVAL and refund < DEFAULT_LIMIT_PAISE:
            raise ValueError(f"a refund of {refund} paise is under the limit, so it does not wait for approval")
        return self


class GoldenCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: Kind
    category: str
    smoke: bool = False
    message: IncomingMessage
    expect: Expected

    @model_validator(mode="after")
    def category_belongs_to_its_kind(self) -> Self:
        allowed = NORMAL_CATEGORIES if self.kind is Kind.NORMAL else ADVERSARIAL_CATEGORIES
        if self.category not in allowed:
            raise ValueError(f"category {self.category!r} is not a {self.kind} category")
        return self

    @property
    def escalates(self) -> bool:
        """Every outcome but a refund paid on its own puts the case in front of a person."""
        return self.expect.outcome is not Outcome.REFUNDED


def load_cases(path: Path = GOLDEN) -> list[GoldenCase]:
    """Every case in `path`, in file order. A line that is not a valid case is refused by its number."""
    cases = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(GoldenCase.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(f"{path.name} line {number} is not a valid case: {error}") from error
    return cases
