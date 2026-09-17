"""
The fixed wording a customer reads, chosen by how their run turned out.

There is no model here and no template engine: four outcomes, four messages, and
the only things that vary inside one are a validated order number and an amount.
That narrowness is the safety property. A reply cannot echo what a customer wrote
because `render` has no parameter their text could arrive through -- the body and
subject a customer sent never reach this module at all.

`amount_paise` is money, and money is integer paise until the last moment. It is
turned into rupees here by integer division, never by a float: 250000 is exactly
Rs 2,500.00, and 5 is exactly Rs 0.05, with no binary-fraction drift in between.
"""

REFUND_ISSUED = "refund_issued"
HANDED_TO_PERSON = "handed_to_person"
ORDER_NOT_FOUND = "order_not_found"
ENQUIRY = "enquiry"

TEMPLATES = frozenset({REFUND_ISSUED, HANDED_TO_PERSON, ORDER_NOT_FOUND, ENQUIRY})

RUPEE = "₹"


def rupees(amount_paise: int) -> str:
    """Paise as rupees, exactly. Negative money is a caller's bug, not a message."""
    if amount_paise < 0:
        raise ValueError("an amount to refund cannot be negative")
    whole, paise = divmod(amount_paise, 100)
    return f"{RUPEE}{whole:,}.{paise:02d}"


def render(template: str, *, order_id: str | None = None, amount_paise: int | None = None) -> str:
    """
    The fixed message for an outcome, with its one or two safe fields filled in.

    A refund names the order it paid and the amount it paid; the other three carry
    no field at all. A template that needs a field it was not given raises, because
    a blank where the number should be is worse than a message that never sent.
    """
    if template == REFUND_ISSUED:
        if order_id is None or amount_paise is None:
            raise ValueError("a refund reply needs both the order and the amount it paid")
        return (
            f"Good news -- we've refunded {rupees(amount_paise)} for order #{order_id} "
            "to your original payment method. It can take a few days to reach you. "
            "Thanks for your patience.\n\n-- Support"
        )
    if template == HANDED_TO_PERSON:
        return (
            "Thanks for getting in touch. We've passed your message to a member of our "
            "support team, who will follow up with you shortly.\n\n-- Support"
        )
    if template == ORDER_NOT_FOUND:
        return (
            "Thanks for getting in touch. We couldn't match your message to an order on "
            "your account, so a member of our team will take a look and follow up.\n\n-- Support"
        )
    if template == ENQUIRY:
        return (
            "Thanks for your message. A member of our support team will get back to you "
            "shortly.\n\n-- Support"
        )
    raise ValueError(f"no such reply template: {template!r}")
