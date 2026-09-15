"""
The handbook's done-when for week 5, word for word, as tests.

  "A Rs 7,200 refund pauses for approval and a Rs 900 one does not, and changing
   the threshold changes the behaviour without a code change."

Each case runs the real worker against a scratch database, with a scripted model
proposing the refund. The threshold is changed the way an operator changes it --
app.guardrails.set_limits, the function behind `python -m app.guardrails set` --
never by editing code or restarting anything.
"""

from decimal import Decimal

import psycopg
import pytest

from app.guardrails import set_limits
from tests.test_approval_path import approvals_of, refund_model, refunds, work
from tests.test_run_agent import ledger, queue

pytestmark = pytest.mark.db


def change_threshold(dsn: str, **limits) -> None:
    with psycopg.connect(dsn) as connection:
        set_limits(connection, by="done-when test", **limits)


def test_a_rs_7200_refund_pauses_for_approval(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    outcome = work(fresh_database, refund_model(720_000))

    assert outcome.status == "waiting_approval"
    assert [approval["status"] for approval in approvals_of(fresh_database, run_id)] == ["pending"]
    assert refunds(fresh_database) == []


def test_a_rs_900_refund_does_not(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    outcome = work(fresh_database, refund_model(90_000))

    assert outcome.status == "done"
    assert approvals_of(fresh_database, run_id) == []
    assert refunds(fresh_database) == [("4821", 90_000, run_id)]


def test_raising_the_threshold_lets_the_rs_7200_refund_through_without_a_code_change(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    change_threshold(fresh_database, limit_paise=1_000_000)

    outcome = work(fresh_database, refund_model(720_000))

    assert outcome.status == "done"
    assert refunds(fresh_database) == [("4821", 720_000, run_id)]


def test_lowering_the_threshold_pauses_the_rs_900_refund_without_a_code_change(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    change_threshold(fresh_database, limit_paise=50_000)

    outcome = work(fresh_database, refund_model(90_000))

    assert outcome.status == "waiting_approval"
    assert [approval["reason"] for approval in approvals_of(fresh_database, run_id)] == [
        "Rs 900 is not under the Rs 500 limit for automatic refunds"
    ]
    assert refunds(fresh_database) == []


def test_raising_the_confidence_threshold_pauses_the_rs_900_refund_too(fresh_database):
    """The other threshold the handbook's week 5 names: confidence."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    change_threshold(fresh_database, min_confidence=Decimal("0.95"))

    outcome = work(fresh_database, refund_model(90_000, confidence="0.9"))

    assert outcome.status == "waiting_approval"
    assert [approval["reason"] for approval in approvals_of(fresh_database, run_id)] == [
        "confidence 0.90 is below the 0.95 needed for automatic refunds"
    ]
