"""
The acceptance criterion for guardrails, word for word, as tests.

  "A Rs 7,200 refund pauses for approval and a Rs 900 one does not, and changing
   the threshold changes the behaviour without a code change."

Each case runs the real worker against a scratch database, with a scripted model
proposing the refund. The threshold is changed the way an operator changes it --
app.guardrails.set_limits, the function behind `python -m app.guardrails set` --
never by editing code or restarting anything.
"""

from datetime import UTC, datetime
from decimal import Decimal

import psycopg
import pytest

from app.contracts import Channel, IncomingMessage
from app.guardrails import set_limits
from app.intake import accept
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    CLASSIFIED_REFUND_REQUEST,
    EXTRACTED_3310,
    EXTRACTED_4821,
    PROPOSED_LOOKUP,
    PROPOSED_LOOKUP_3310,
    ScriptedModel,
    proposed_refund,
)
from tests.test_approval_path import (
    approvals_of,
    duplicate_model,
    queue_about,
    refund_model,
    refunds,
    work,
)
from tests.test_run_agent import ledger, queue

pytestmark = pytest.mark.db


def change_of_mind_model(amount_paise: int, confidence: str = "0.9") -> ScriptedModel:
    """Reads the message as a plain refund request, looks 4821 up, then asks to pay anyway."""
    return ScriptedModel(
        classify=CLASSIFIED_REFUND_REQUEST,
        extract=EXTRACTED_4821,
        plan=[PROPOSED_LOOKUP, proposed_refund(amount_paise, confidence)],
    )


def single_charge_model(amount_paise: int, confidence: str = "1.0") -> ScriptedModel:
    """Calls order 3310 a duplicate with complete confidence. The ledger shows one charge."""
    return ScriptedModel(
        classify=CLASSIFIED_DUPLICATE,
        extract=EXTRACTED_3310,
        plan=[PROPOSED_LOOKUP_3310, proposed_refund(amount_paise, confidence, order_id="3310")],
    )


def queue_3310(dsn: str) -> str:
    """A run from the customer who placed order 3310, which was charged once."""
    message = IncomingMessage(
        channel=Channel.EMAIL,
        external_id="3310-once",
        sender="dev@example.com",
        subject="Charged twice for order #3310",
        body="Hi, I think I was charged twice for order #3310 last Tuesday.",
        received_at=datetime(2026, 9, 13, 9, 30, tzinfo=UTC),
    )
    with psycopg.connect(dsn) as connection:
        return str(accept(connection, message).run_id)


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
    """Order 4902 was charged Rs 900 twice, so Rs 900 is what its duplicate owes back."""
    ledger(fresh_database)
    run_id = queue_about(fresh_database, "4902", "rs900")

    outcome = work(fresh_database, duplicate_model("4902", 90_000))

    assert outcome.status == "done"
    assert approvals_of(fresh_database, run_id) == []
    assert refunds(fresh_database) == [("4902", 90_000, run_id)]


def test_raising_the_threshold_lets_the_rs_7200_refund_through_without_a_code_change(fresh_database):
    """Order 4903 was charged Rs 7,200 twice. Only the threshold stands between it and payment."""
    ledger(fresh_database)
    run_id = queue_about(fresh_database, "4903", "rs7200")
    change_threshold(fresh_database, limit_paise=1_000_000)

    outcome = work(fresh_database, duplicate_model("4903", 720_000))

    assert outcome.status == "done"
    assert refunds(fresh_database) == [("4903", 720_000, run_id)]


def test_lowering_the_threshold_pauses_the_rs_900_refund_without_a_code_change(fresh_database):
    ledger(fresh_database)
    run_id = queue_about(fresh_database, "4902", "rs900-lower")
    change_threshold(fresh_database, limit_paise=50_000)

    outcome = work(fresh_database, duplicate_model("4902", 90_000))

    assert outcome.status == "waiting_approval"
    assert [approval["reason"] for approval in approvals_of(fresh_database, run_id)] == [
        "Rs 900 is not under the Rs 500 limit for automatic refunds"
    ]
    assert refunds(fresh_database) == []


def test_raising_the_confidence_threshold_pauses_the_rs_900_refund_too(fresh_database):
    """The other threshold: confidence."""
    ledger(fresh_database)
    run_id = queue(fresh_database)
    change_threshold(fresh_database, min_confidence=Decimal("0.95"))

    outcome = work(fresh_database, refund_model(90_000, confidence="0.9"))

    assert outcome.status == "waiting_approval"
    assert [approval["reason"] for approval in approvals_of(fresh_database, run_id)] == [
        "confidence 0.90 is below the 0.95 needed for automatic refunds"
    ]


# --- an automatic payment needs the conditions, not just a small amount ----------


def test_a_refund_that_is_not_a_confirmed_duplicate_goes_to_a_person_not_an_approval(fresh_database):
    """
    The guardrail judged the amount and the model's own confidence and nothing else, so a
    confident model could have any small refund paid by saying it was owed. An automatic
    payment now needs the conditions to hold in the ledger: a duplicate, charged twice.
    Anything else is a person's to decide, and there is nothing to approve -- the agent has
    no payment it can stand behind, so it hands the case over instead of proposing one.
    """
    ledger(fresh_database)
    run_id = queue(fresh_database)

    outcome = work(fresh_database, change_of_mind_model(90_000))

    assert outcome.status == "waiting_approval"
    assert approvals_of(fresh_database, run_id) == []
    assert refunds(fresh_database) == []


def test_an_order_charged_once_is_no_duplicate_however_confident_the_model_is(fresh_database):
    ledger(fresh_database)
    run_id = queue_3310(fresh_database)

    outcome = work(fresh_database, single_charge_model(90_000))

    assert outcome.status == "waiting_approval"
    assert approvals_of(fresh_database, run_id) == []
    assert refunds(fresh_database) == []


def test_a_confirmed_duplicate_under_the_limit_is_still_paid_automatically(fresh_database):
    ledger(fresh_database)
    run_id = queue(fresh_database)

    outcome = work(fresh_database, refund_model(360_000))

    assert outcome.status == "done"
    assert refunds(fresh_database) == [("4821", 360_000, run_id)]
