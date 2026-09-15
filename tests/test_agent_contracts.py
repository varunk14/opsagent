"""
The shapes the model's answers must fit before the agent acts on them.

Every test feeds JSON the way the model would send it. Money stays an exact
integer of paise, confidence stays exact, and a proposed tool call must match
the tool's declared arguments -- or it is refused here, not discovered later.
"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.contracts import Classification, ExtractedRefund, Intent, ProposedAction

# --- classification ----------------------------------------------------------


def test_a_classification_parses_from_model_json():
    c = Classification.model_validate_json(
        '{"intent": "duplicate_charge", "confidence": 0.9, "reasoning": "charged twice"}'
    )

    assert c.intent is Intent.DUPLICATE_CHARGE
    assert c.confidence == Decimal("0.9")


def test_confidence_from_json_is_exact_not_binary_float():
    """0.9 as a float is 0.90000000000000002220...; thresholds compare exactly."""
    c = Classification.model_validate_json(
        '{"intent": "other", "confidence": 0.7, "reasoning": "x"}'
    )

    assert c.confidence == Decimal("0.7")


@pytest.mark.parametrize("confidence", ["1.01", "-0.1"])
def test_confidence_outside_zero_to_one_is_refused(confidence):
    with pytest.raises(ValidationError):
        Classification.model_validate_json(
            f'{{"intent": "other", "confidence": {confidence}, "reasoning": "x"}}'
        )


def test_an_intent_outside_the_known_set_is_refused():
    with pytest.raises(ValidationError):
        Classification.model_validate_json(
            '{"intent": "wants_a_pony", "confidence": 0.5, "reasoning": "x"}'
        )


def test_runaway_reasoning_is_refused():
    with pytest.raises(ValidationError):
        Classification(intent=Intent.OTHER, confidence=Decimal("0.5"), reasoning="x" * 1001)


# --- extraction ---------------------------------------------------------------


def test_an_amount_the_customer_never_stated_is_allowed_to_be_missing():
    """Priya never says how much. Guessing would be worse than admitting it."""
    e = ExtractedRefund.model_validate_json(
        '{"order_id": "4821", "amount_paise": null, "reason": "charged twice"}'
    )

    assert e.amount_paise is None


def test_an_amount_in_whole_paise_is_kept():
    e = ExtractedRefund.model_validate_json(
        '{"order_id": "3310", "amount_paise": 720000, "reason": "changed mind"}'
    )

    assert e.amount_paise == 720_000


@pytest.mark.parametrize("amount", ["7200.0", '"720000"', "true", "-1"])
def test_an_amount_that_is_not_whole_non_negative_paise_is_refused(amount):
    """A float, a string, a boolean or a negative is a guess dressed as money."""
    with pytest.raises(ValidationError):
        ExtractedRefund.model_validate_json(
            f'{{"order_id": "3310", "amount_paise": {amount}, "reason": "x"}}'
        )


def test_a_blank_order_id_is_refused_but_a_missing_one_is_not():
    with pytest.raises(ValidationError):
        ExtractedRefund(order_id="  ", amount_paise=None, reason="x")

    assert ExtractedRefund(order_id=None, amount_paise=None, reason="x").order_id is None


# --- proposed action ------------------------------------------------------------


def test_a_well_formed_proposal_parses():
    p = ProposedAction.model_validate_json(
        '{"tool": "get_order", "args": {"order_id": "4821"},'
        ' "confidence": 0.8, "reasoning": "check the charges first"}'
    )

    assert p.tool == "get_order"
    assert p.args == {"order_id": "4821"}


def test_escalating_is_a_valid_proposal():
    p = ProposedAction(
        tool="escalate_to_human", args={"reason": "policy unclear"},
        confidence=Decimal("0.4"), reasoning="not sure",
    )

    assert p.tool == "escalate_to_human"


def test_a_tool_that_does_not_exist_is_refused():
    with pytest.raises(ValidationError, match="unknown tool"):
        ProposedAction(tool="wire_money", args={}, confidence=Decimal("0.9"), reasoning="x")


def test_a_missing_required_argument_is_refused():
    with pytest.raises(ValidationError, match="amount_paise"):
        ProposedAction(
            tool="issue_refund", args={"order_id": "4821", "reason": "dup"},
            confidence=Decimal("0.9"), reasoning="x",
        )


def test_an_argument_the_tool_does_not_take_is_refused():
    with pytest.raises(ValidationError, match="unexpected"):
        ProposedAction(
            tool="get_order", args={"order_id": "4821", "force": True},
            confidence=Decimal("0.9"), reasoning="x",
        )


@pytest.mark.parametrize("amount", ["360000", 3600.0, True])
def test_a_refund_amount_of_the_wrong_type_is_refused(amount):
    """The schema says integer paise. A string, float or boolean is not."""
    with pytest.raises(ValidationError, match="amount_paise"):
        ProposedAction(
            tool="issue_refund",
            args={"order_id": "4821", "amount_paise": amount, "reason": "dup"},
            confidence=Decimal("0.9"), reasoning="x",
        )
