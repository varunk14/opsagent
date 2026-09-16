"""
The guardrail: which refunds may run on their own, and which need a person.

Concept 2.10: the limit lives in code and in the database, never in the prompt.
A model told "refunds over Rs 5,000 need approval" can be talked out of it by the
email it is reading; a comparison in Python cannot. So `judge` is pure -- no I/O,
no model -- and the numbers it compares against are a row in Postgres that an
operator changes without touching code.

"Under Rs 5,000" is strict. Exactly Rs 5,000 needs a person, so the limit is the
first amount that is NOT automatic, and a limit of zero makes every refund manual.
"""

from decimal import Decimal

import psycopg
import pytest

from app.contracts import ProposedAction
from app.guardrails import (
    Evidence,
    Guardrails,
    evidence_of,
    judge,
    justified,
    load,
    set_limits,
)

DEFAULTS = Guardrails(auto_refund_limit_paise=500_000, min_confidence=Decimal("0.85"))


def refund(amount_paise: int, confidence: str = "0.90") -> ProposedAction:
    return ProposedAction(
        tool="issue_refund",
        args={"order_id": "4821", "amount_paise": amount_paise, "reason": "charged twice"},
        confidence=Decimal(confidence),
        reasoning="the ledger shows a duplicate charge",
    )


# --- judge: pure ----------------------------------------------------------------


def test_a_small_confident_refund_runs_on_its_own():
    verdict = judge(refund(90_000), DEFAULTS)

    assert verdict.runs is True
    assert verdict.reason is None


def test_a_refund_over_the_limit_needs_a_person():
    verdict = judge(refund(720_000), DEFAULTS)

    assert verdict.runs is False
    assert verdict.reason == "Rs 7,200 is not under the Rs 5,000 limit for automatic refunds"


def test_a_refund_of_exactly_the_limit_needs_a_person():
    """The rule says under Rs 5,000; the boundary belongs to the person, not the agent."""
    assert judge(refund(500_000), DEFAULTS).runs is False


def test_a_refund_one_paisa_under_the_limit_runs():
    assert judge(refund(499_999), DEFAULTS).runs is True


def test_an_unsure_refund_needs_a_person_however_small():
    verdict = judge(refund(90_000, confidence="0.50"), DEFAULTS)

    assert verdict.runs is False
    assert verdict.reason == "confidence 0.50 is below the 0.85 needed for automatic refunds"


def test_a_confidence_just_under_the_threshold_is_shown_unrounded():
    """Rounded, 0.849 would read "0.85 is below the 0.85 needed" and the person could not tell why."""
    verdict = judge(refund(90_000, confidence="0.849"), DEFAULTS)

    assert verdict.reason == "confidence 0.849 is below the 0.85 needed for automatic refunds"


def test_confidence_exactly_at_the_threshold_is_enough():
    assert judge(refund(90_000, confidence="0.85"), DEFAULTS).runs is True


def test_both_reasons_are_given_when_both_apply():
    verdict = judge(refund(720_000, confidence="0.50"), DEFAULTS)

    assert verdict.runs is False
    assert verdict.reason == (
        "Rs 7,200 is not under the Rs 5,000 limit for automatic refunds; "
        "confidence 0.50 is below the 0.85 needed for automatic refunds"
    )


def test_a_limit_of_zero_makes_every_refund_manual():
    """The kill switch: no amount is under zero."""
    switched_off = Guardrails(auto_refund_limit_paise=0, min_confidence=Decimal("0.00"))

    verdict = judge(refund(1, confidence="1.00"), switched_off)

    assert verdict.runs is False
    assert verdict.reason == "Rs 0.01 is not under the Rs 0 limit for automatic refunds"


def test_paise_are_shown_when_an_amount_is_not_whole_rupees():
    verdict = judge(refund(720_050), DEFAULTS)

    assert verdict.reason is not None
    assert verdict.reason.startswith("Rs 7,200.50 is not under")


def test_what_the_order_has_already_had_back_counts_toward_the_limit():
    """Found in the Unit A security review: otherwise two Rs 3,600 refunds pass a Rs 5,000 limit."""
    verdict = judge(refund(360_000), DEFAULTS, already_refunded_paise=360_000)

    assert verdict.runs is False
    assert verdict.reason == (
        "Rs 3,600 would bring refunds on order 4821 to Rs 7,200, not under the Rs 5,000 limit for automatic refunds"
    )


def test_a_further_refund_that_keeps_the_order_under_the_limit_runs():
    assert judge(refund(100_000), DEFAULTS, already_refunded_paise=399_999).runs is True


def test_a_further_refund_that_reaches_the_limit_exactly_needs_a_person():
    assert judge(refund(100_000), DEFAULTS, already_refunded_paise=400_000).runs is False


def test_what_was_already_refunded_cannot_be_negative():
    """A negative total would widen the limit instead of narrowing it."""
    with pytest.raises(ValueError, match="already refunded"):
        judge(refund(90_000), DEFAULTS, already_refunded_paise=-1)


@pytest.mark.parametrize("tool", ["get_order", "search_policy", "escalate_to_human"])
def test_only_refunds_are_judged(tool):
    """
    Judging a lookup would be meaningless, and quietly answering "runs" for it
    would let a caller that forgot the tool groups treat judge() as permission.
    """
    args = {"get_order": {"order_id": "4821"}, "search_policy": {"question": "q"}}.get(tool, {"reason": "r"})
    action = ProposedAction(tool=tool, args=args, confidence=Decimal("0.99"), reasoning="r")

    with pytest.raises(ValueError, match="only issue_refund"):
        judge(action, DEFAULTS)


def test_the_judgement_is_the_same_every_time():
    """No clock, no randomness, no I/O: the same inputs always give the same answer."""
    assert judge(refund(720_000), DEFAULTS) == judge(refund(720_000), DEFAULTS)


# --- load and set: the database ---------------------------------------------------


@pytest.mark.db
def test_load_reads_the_defaults(db):
    guardrails = load(db)

    assert guardrails.auto_refund_limit_paise == 500_000
    assert guardrails.min_confidence == Decimal("0.85")


@pytest.mark.db
def test_a_changed_limit_is_what_the_next_load_sees(db):
    """The second half of the acceptance test: the behaviour changes with no code change."""
    assert judge(refund(720_000), load(db)).runs is False

    set_limits(db, limit_paise=1_000_000, by="asha")

    assert judge(refund(720_000), load(db)).runs is True


@pytest.mark.db
def test_setting_one_value_leaves_the_other_alone(db):
    set_limits(db, min_confidence=Decimal("0.95"), by="asha")

    guardrails = load(db)
    assert guardrails.auto_refund_limit_paise == 500_000
    assert guardrails.min_confidence == Decimal("0.95")


@pytest.mark.db
def test_a_change_records_who_made_it_and_when(db):
    set_limits(db, limit_paise=0, by="asha")

    updated_by, recent = db.execute(
        "SELECT updated_by, updated_at > now() - interval '1 minute' FROM guardrails"
    ).fetchone()
    assert (updated_by, recent) == ("asha", True)


@pytest.mark.db
def test_set_limits_returns_what_is_now_in_force(db):
    assert set_limits(db, limit_paise=250_000, by="asha") == Guardrails(
        auto_refund_limit_paise=250_000, min_confidence=Decimal("0.85")
    )


@pytest.mark.db
def test_set_limits_does_not_commit(migrated_database):
    """The caller owns the transaction, as everywhere else in this codebase."""
    with psycopg.connect(migrated_database) as writer:
        set_limits(writer, limit_paise=1, by="asha")
        with psycopg.connect(migrated_database) as reader:
            assert load(reader).auto_refund_limit_paise == 500_000
        writer.rollback()


@pytest.mark.db
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({}, "nothing to change"),
        ({"limit_paise": 100, "by": "  "}, "who is making"),
        ({"limit_paise": -1}, "at least 0"),
        ({"limit_paise": True}, "whole number of paise"),
        ({"limit_paise": 100.0}, "whole number of paise"),
        ({"min_confidence": 0.9}, "must be a Decimal"),
        ({"min_confidence": True}, "must be a Decimal"),
        ({"min_confidence": Decimal("1.01")}, "between 0 and 1"),
        ({"min_confidence": Decimal("0.855")}, "two decimal places"),
        # Ordered comparisons on these raise InvalidOperation; the finiteness check must come first.
        ({"min_confidence": Decimal("NaN")}, "between 0 and 1"),
        ({"min_confidence": Decimal("sNaN")}, "between 0 and 1"),
        ({"min_confidence": Decimal("Infinity")}, "between 0 and 1"),
    ],
    ids=[
        "nothing", "blank-name", "negative", "bool", "float-limit", "float-confidence", "bool-confidence",
        "over-one", "rounded", "nan", "snan", "infinity",
    ],
)
def test_a_bad_change_is_refused_before_it_reaches_the_database(db, changes, message):
    """
    numeric(3,2) would silently round 0.855 to 0.86, the same trap as a
    fractional paisa in test_schema.py -- so the boundary refuses it instead.
    """
    kwargs = {"by": "asha"} | changes

    with pytest.raises(ValueError, match=message):
        set_limits(db, **kwargs)

    assert load(db) == DEFAULTS


# --- justified: is this refund owed at all? --------------------------------------


def duplicate(*charges_paise: int, intent: str = "duplicate_charge") -> Evidence:
    """What a run established about order 4821 before proposing to pay."""
    return Evidence(intent=intent, charges_paise=charges_paise)


def test_a_confirmed_duplicate_refunded_at_one_of_its_charges_is_owed():

    assert justified(refund(360_000), duplicate(360_000, 360_000)).runs is True


def test_a_refund_for_anything_but_a_duplicate_is_a_persons_decision():

    verdict = justified(refund(90_000), duplicate(360_000, 360_000, intent="refund_request"))

    assert verdict.runs is False
    assert "duplicate" in (verdict.reason or "")


def test_an_order_charged_once_has_no_duplicate_to_refund():

    verdict = justified(refund(360_000), duplicate(360_000))

    assert verdict.runs is False
    assert "4821" in (verdict.reason or "")


def test_a_refund_proposed_before_any_lookup_is_not_owed_by_anything():

    verdict = justified(refund(360_000), duplicate())

    assert verdict.runs is False
    assert "looked up" in (verdict.reason or "")


def test_part_of_a_confirmed_duplicate_is_still_owed():
    """
    The amount is deliberately not a condition. The ledger already refuses a refund larger
    than the order was charged, the limit bounds what runs without a person, and a refund
    split into parts is judged as the total it adds up to -- so requiring the amount to equal
    a charge exactly would refuse legitimate partial refunds while adding no safety.
    """

    assert justified(refund(90_000), duplicate(360_000, 360_000)).runs is True


def test_the_charges_are_read_from_the_lookup_of_that_order_only():

    steps = [
        {"tool": "get_order", "args": {"order_id": "9999"}, "result": {"order_id": "9999", "charges_paise": [10, 10]}},
        {"tool": "get_order", "args": {"order_id": "4821"}, "result": {"order_id": "4821", "charges_paise": [360_000, 360_000]}},
    ]

    found = evidence_of("duplicate_charge", steps, "4821")

    assert found.charges_paise == (360_000, 360_000)
    assert found.intent == "duplicate_charge"


def test_a_lookup_that_returned_nothing_leaves_no_charges():

    steps = [{"tool": "get_order", "args": {"order_id": "4821"}, "result": None}]

    assert evidence_of("duplicate_charge", steps, "4821").charges_paise == ()


def test_a_refund_step_is_not_a_lookup_and_confirms_nothing():
    steps = [
        {"tool": "issue_refund", "args": {"order_id": "4821"}, "result": {"order_id": "4821", "refunded": True}},
    ]

    assert evidence_of("duplicate_charge", steps, "4821").charges_paise == ()
