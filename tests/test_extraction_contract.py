"""
The boundary between model output and data the program will act on.

No model is called here. These tests cover the contract itself, which is
deterministic and therefore genuinely unit-testable. Whether the model fills it
in correctly is a question for evaluation, not for pytest.
"""

import pytest
from pydantic import ValidationError

from experiments.validate_refund_details import Extraction


def test_accepts_a_well_formed_extraction():
    extraction = Extraction.model_validate(
        {
            "can_extract": True,
            "order_id": "4821",
            "reason": "double_charge",
            "amount_paise": 360_000,
            "confidence": 0.88,
            "note": None,
        }
    )

    assert extraction.order_id == "4821"
    assert extraction.amount_paise == 360_000


def test_rejects_confidence_above_one():
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        Extraction.model_validate({"can_extract": False, "confidence": 5.0})


def test_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        Extraction.model_validate({"can_extract": False, "confidence": -0.1})


def test_rejects_a_reason_outside_the_allowed_set():
    with pytest.raises(ValidationError):
        Extraction.model_validate(
            {
                "can_extract": True,
                "order_id": "4821",
                "reason": "customer_changed_their_mind",
                "confidence": 0.9,
            }
        )


def test_rejects_invented_fields():
    """A model that adds fields we never asked for is an error, not a shrug."""
    with pytest.raises(ValidationError):
        Extraction.model_validate(
            {
                "can_extract": False,
                "confidence": 0.5,
                "refund_immediately": True,
            }
        )


def test_rejects_a_fractional_amount():
    """Paise are indivisible. A fractional amount means something upstream is wrong."""
    with pytest.raises(ValidationError):
        Extraction.model_validate(
            {
                "can_extract": True,
                "order_id": "4821",
                "reason": "double_charge",
                "amount_paise": 360_000.5,
                "confidence": 0.9,
            }
        )


def test_success_must_come_with_the_goods():
    """can_extract=True while omitting the fields is a contradiction, not a partial result."""
    with pytest.raises(ValidationError, match="missing"):
        Extraction.model_validate({"can_extract": True, "confidence": 0.95})


def test_refusal_does_not_require_the_fields():
    extraction = Extraction.model_validate(
        {"can_extract": False, "confidence": 0.0, "note": "no email content found"}
    )

    assert extraction.can_extract is False
    assert extraction.note == "no email content found"
