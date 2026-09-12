"""
Regression guard for the duplicate-refund bug.

These assert that the BROKEN reproduction is still broken. That sounds backwards
until you consider what it protects: the fix in prevent_duplicate_refunds.py is
only meaningful if the bug it fixes is real and still demonstrable. If someone
quietly "tidies up" this file and the bug stops reproducing, the fix loses its
evidence and these tests fail loudly.
"""

import pytest

from experiments import duplicate_refund_bug as bug

ORDER_ID = "4821"
AMOUNT_PAISE = 360_000


@pytest.fixture(autouse=True)
def clean_bank():
    bug.BANK_LEDGER.clear()
    bug._call_count = 0
    yield
    bug.BANK_LEDGER.clear()
    bug._call_count = 0


def test_the_bug_still_reproduces():
    """One refund requested, two refunds issued, no exception raised."""
    result = bug.issue_refund(ORDER_ID, AMOUNT_PAISE)

    assert len(bug.BANK_LEDGER) == 2
    assert sum(e["amount_paise"] for e in bug.BANK_LEDGER) == AMOUNT_PAISE * 2
    assert result["refund_id"] == "rf_002"


def test_the_first_refund_is_orphaned():
    """
    The worst part of the bug: rf_001 exists at the bank and the caller has no
    record of it. Nothing in our code could ever find or reverse it.
    """
    result = bug.issue_refund(ORDER_ID, AMOUNT_PAISE)

    known_to_caller = {result["refund_id"]}
    at_the_bank = {e["refund_id"] for e in bug.BANK_LEDGER}

    assert at_the_bank - known_to_caller == {"rf_001"}


def test_the_bank_itself_is_not_at_fault():
    """A single call behaves correctly. The bug only appears under retry."""
    bug._call_count = 99  # past the simulated outage
    bug.bank_issue_refund(ORDER_ID, AMOUNT_PAISE)

    assert len(bug.BANK_LEDGER) == 1


def test_amounts_are_integer_paise():
    bug.issue_refund(ORDER_ID, AMOUNT_PAISE)

    for entry in bug.BANK_LEDGER:
        assert isinstance(entry["amount_paise"], int)
