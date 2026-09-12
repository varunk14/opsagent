"""
The prompt is an input to the system, so it gets the same scrutiny as any other.
A prompt that silently loses the customer's email produces confident nonsense.
"""

from experiments import extract_refund_details as raw


def test_prompt_contains_the_customer_email():
    """Guards against the failure where the model says 'I see no email'."""
    assert raw.PRIYA_EMAIL in raw.PROMPT
    assert "4821" in raw.PROMPT


def test_prompt_names_every_field_the_schema_requires():
    for field in ("order_id", "reason", "amount_paise", "confidence"):
        assert field in raw.PROMPT, f"prompt never mentions {field}"


def test_prompt_lists_the_allowed_reasons():
    for reason in ("double_charge", "damaged", "not_received", "other"):
        assert reason in raw.PROMPT
