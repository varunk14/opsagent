"""
Critical path: money must move exactly once.

These are the tests that matter most in the project. A refund that fires twice
costs real money and raises no error, so correctness here is not negotiable.

Note the shape of the first two tests: they assert that the BROKEN versions are
still broken. That is deliberate. A regression test for a bug is only meaningful
if it fails when the bug comes back.
"""

import pytest

from experiments import prevent_duplicate_refunds as refunds

ORDER_ID = "4821"
AMOUNT_PAISE = 360_000  # Rs 3,600


@pytest.fixture(autouse=True)
def clean_bank():
    """Every test starts with an empty ledger and a fresh outage simulation."""
    refunds.reset_bank()
    yield
    refunds.reset_bank()


def total_refunded() -> int:
    return sum(entry["amount_paise"] for entry in refunds.BANK_LEDGER)


# --- the fix ---------------------------------------------------------------


def test_same_key_across_retries_refunds_once():
    key = "run_1:step_5:issue_refund"

    refunds.call_with_retry(
        refunds.bank_issue_refund, ORDER_ID, AMOUNT_PAISE, key=key
    )

    assert len(refunds.BANK_LEDGER) == 1
    assert total_refunded() == AMOUNT_PAISE


def test_retry_returns_the_original_refund_id():
    """The caller must end up holding the ID that actually exists at the bank."""
    key = "run_1:step_5:issue_refund"

    result = refunds.call_with_retry(
        refunds.bank_issue_refund, ORDER_ID, AMOUNT_PAISE, key=key
    )

    assert result["refund_id"] == refunds.BANK_LEDGER[0]["refund_id"]


def test_distinct_operations_are_not_deduplicated():
    """
    Guards against over-correcting. Two genuinely different refunds must both
    go through; only repeats of the SAME operation are suppressed.
    """
    refunds.call_with_retry(
        refunds.bank_issue_refund, ORDER_ID, AMOUNT_PAISE,
        key="run_1:step_5:issue_refund",
    )
    refunds.bank_issue_refund(
        ORDER_ID, AMOUNT_PAISE, key="run_1:step_9:issue_refund"
    )

    assert len(refunds.BANK_LEDGER) == 2


# --- the bugs, asserted as still broken ------------------------------------


def test_without_a_key_a_retry_pays_twice():
    refunds.call_with_retry(refunds.bank_issue_refund, ORDER_ID, AMOUNT_PAISE)

    assert len(refunds.BANK_LEDGER) == 2
    assert total_refunded() == AMOUNT_PAISE * 2


def test_key_generated_per_attempt_still_pays_twice():
    """
    The counter-example. Generating the key inside the retry loop looks like a
    fix and changes nothing, because the bank never sees the same key twice.
    """
    refunds.call_with_retry(
        refunds.broken_refund_with_key, ORDER_ID, AMOUNT_PAISE
    )

    assert len(refunds.BANK_LEDGER) == 2


# --- properties of the money itself ----------------------------------------


def test_amounts_are_integers_not_floats():
    """Rupees as floats lose money. Paise as ints do not."""
    refunds.call_with_retry(
        refunds.bank_issue_refund, ORDER_ID, AMOUNT_PAISE, key="run_2:step_1:issue_refund"
    )

    for entry in refunds.BANK_LEDGER:
        assert isinstance(entry["amount_paise"], int)
        assert not isinstance(entry["amount_paise"], bool)
