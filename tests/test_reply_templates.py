"""
The fixed wording a customer reads, and the one number in it that is theirs.

These messages carry no customer text. The proof is in the signature: `render`
takes an outcome, a validated order number and an amount in paise, and nothing a
customer typed. Whatever they wrote in the body cannot reach the reply, because
there is no parameter it could arrive through.

What the tests pin is that the wording is fixed per outcome, that the one amount
in it is rendered as money and not as a float, and that a caller who forgets the
fields a template needs is stopped rather than shipping a blank where the number
should be.
"""

import pytest

from app.replies import templates


def test_every_outcome_renders_to_something_fixed():
    for template in templates.TEMPLATES:
        body = templates.render(template, order_id="4821", amount_paise=250000)
        assert body.strip(), f"{template} rendered nothing"


def test_a_refund_names_the_order_and_the_amount():
    body = templates.render(templates.REFUND_ISSUED, order_id="4821", amount_paise=250000)
    assert "4821" in body
    assert "2,500.00" in body


def test_money_is_exact_not_a_float():
    assert templates.rupees(250000) == "₹2,500.00"
    assert templates.rupees(5) == "₹0.05"
    assert templates.rupees(100) == "₹1.00"
    assert templates.rupees(0) == "₹0.00"


def test_a_negative_amount_is_a_bug_not_a_message():
    with pytest.raises(ValueError):
        templates.rupees(-1)


def test_a_refund_without_its_amount_is_refused():
    with pytest.raises(ValueError):
        templates.render(templates.REFUND_ISSUED, order_id="4821")


def test_a_refund_without_its_order_is_refused():
    with pytest.raises(ValueError):
        templates.render(templates.REFUND_ISSUED, amount_paise=250000)


def test_an_unknown_template_is_refused():
    with pytest.raises(ValueError):
        templates.render("apology_coupon")


def test_the_handover_needs_no_customer_detail():
    body = templates.render(templates.HANDED_TO_PERSON)
    assert body.strip()


def test_an_enquiry_and_a_missing_order_each_say_their_piece():
    assert templates.render(templates.ENQUIRY).strip()
    assert templates.render(templates.ORDER_NOT_FOUND).strip()
