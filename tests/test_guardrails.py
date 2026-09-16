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

from app.contracts import Intent, ProposedAction
from app.guardrails import (
    Budgets,
    Evidence,
    Guardrails,
    evidence_of,
    judge,
    justified,
    load,
    set_limits,
)

# What migration 009 puts in the row: roughly three times a measured run.
DEFAULT_BUDGETS = Budgets(
    max_tokens_per_run=10_000, max_cost_usd_per_run=Decimal("0.002000"), max_seconds_per_run=180
)
DEFAULTS = Guardrails(
    auto_refund_limit_paise=500_000, min_confidence=Decimal("0.85"), budgets=DEFAULT_BUDGETS
)


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
    switched_off = Guardrails(auto_refund_limit_paise=0, min_confidence=Decimal("0.00"), budgets=DEFAULT_BUDGETS)

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
        auto_refund_limit_paise=250_000, min_confidence=Decimal("0.85"), budgets=DEFAULT_BUDGETS
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


def duplicate(*charges_paise: int, intent: Intent | None = Intent.DUPLICATE_CHARGE) -> Evidence:
    """What a run established about order 4821 before proposing to pay."""
    return Evidence(intent=intent, charges_paise=charges_paise)


def test_a_confirmed_duplicate_refunded_at_one_of_its_charges_is_owed():

    assert justified(refund(360_000), duplicate(360_000, 360_000)).runs is True


def test_a_refund_for_anything_but_a_duplicate_is_a_persons_decision():

    verdict = justified(refund(90_000), duplicate(360_000, 360_000, intent=Intent.REFUND_REQUEST))

    assert verdict.runs is False
    assert "duplicate" in (verdict.reason or "")


def test_an_order_charged_once_has_no_duplicate_to_refund():

    verdict = justified(refund(360_000), duplicate(360_000))

    assert verdict.runs is False
    assert verdict.reason == "order 4821 was charged once, so there is no duplicate to refund"


def test_a_refund_proposed_before_any_lookup_is_not_owed_by_anything():

    verdict = justified(refund(360_000), duplicate())

    assert verdict.runs is False
    assert "looked up" in (verdict.reason or "")


def test_a_refund_for_less_than_the_duplicate_is_a_persons_decision():
    """
    A duplicate charge is owed back at the amount it was charged. Something smaller may well be
    right -- a partial refund can be exactly what a case needs -- but nothing the run established
    says what that smaller number should be, so a person chooses it rather than the model.
    """
    verdict = justified(refund(90_000), duplicate(360_000, 360_000))

    assert verdict.runs is False
    assert "not one of the charges" in (verdict.reason or "")


def test_a_refund_for_more_than_was_ever_charged_is_refused_here_too():
    """The ledger would refuse it as well. Refusing here means it never reaches the ledger."""
    assert justified(refund(400_000), duplicate(360_000, 360_000)).runs is False


def test_the_duplicate_is_owed_back_at_what_it_was_charged():
    assert justified(refund(360_000), duplicate(360_000, 360_000)).runs is True


def test_the_amount_must_match_the_charge_that_was_duplicated():
    """Two amounts, one of them repeated: the repeated one is the duplicate, not the other."""
    assert justified(refund(12_000), duplicate(12_000, 360_000, 360_000)).runs is False
    assert justified(refund(360_000), duplicate(12_000, 360_000, 360_000)).runs is True


def test_the_charges_are_read_from_the_lookup_of_that_order_only():

    steps = [
        {"tool": "get_order", "args": {"order_id": "9999"}, "result": {"order_id": "9999", "charges_paise": [10, 10]}},
        {"tool": "get_order", "args": {"order_id": "4821"}, "result": {"order_id": "4821", "charges_paise": [360_000, 360_000]}},
    ]

    found = evidence_of(Intent.DUPLICATE_CHARGE, steps, "4821")

    assert found.charges_paise == (360_000, 360_000)
    assert found.intent == "duplicate_charge"


def test_a_lookup_that_returned_nothing_leaves_no_charges():

    steps = [{"tool": "get_order", "args": {"order_id": "4821"}, "result": None}]

    assert evidence_of(Intent.DUPLICATE_CHARGE, steps, "4821").charges_paise == ()


def test_a_refund_step_is_not_a_lookup_and_confirms_nothing():
    """Only a lookup reports what the ledger holds. Any other step saying so is not asked."""
    steps = [
        {
            "tool": "issue_refund",
            "args": {"order_id": "4821"},
            "result": {"order_id": "4821", "refunded": True, "charges_paise": [360_000, 360_000]},
        },
    ]

    assert evidence_of(Intent.DUPLICATE_CHARGE, steps, "4821").charges_paise == ()


def test_an_order_charged_twice_for_different_things_holds_no_duplicate():
    """
    Being charged more than once is not being charged twice. An order billed for the item and
    then for shipping has two charges and no duplicate, and the only thing left saying it is a
    duplicate would be the model's reading of an email the customer wrote -- which is the input
    this check exists to stop trusting. Two charges of the same amount is the evidence.
    """
    verdict = justified(refund(90_000), duplicate(360_000, 12_000))

    assert verdict.runs is False
    assert "same amount" in (verdict.reason or "")


def test_a_repeated_charge_among_others_is_still_a_duplicate():
    assert justified(refund(360_000), duplicate(12_000, 360_000, 360_000)).runs is True


def test_an_order_id_is_matched_exactly_and_never_normalised():
    """A lookup of 04821 says nothing about 4821: the ids are compared as they are, not as numbers."""
    steps = [
        {"tool": "get_order", "args": {"order_id": "04821"}, "result": {"order_id": "04821", "charges_paise": [10, 10]}}
    ]

    assert evidence_of(Intent.DUPLICATE_CHARGE, steps, "4821").charges_paise == ()


def test_an_order_with_two_different_duplicates_is_a_persons_decision():
    """
    Charged twice for one thing and twice for another: which of them the customer means is a
    reading of their message, and the message is the thing that cannot be trusted. Nothing in
    the ledger picks between two duplicates, so a person does.
    """
    verdict = justified(refund(50_000), duplicate(10_000, 10_000, 50_000, 50_000))

    assert verdict.runs is False
    assert "more than one" in (verdict.reason or "")


def test_one_duplicate_among_single_charges_is_still_decided_here():
    """Only the repeated amounts count, so other charges on the order do not make it ambiguous."""
    assert justified(refund(50_000), duplicate(10_000, 50_000, 50_000, 70_000)).runs is True


# --- what one run may spend -------------------------------------------------------------------


def test_the_row_carries_ceilings_a_run_may_not_pass():
    """Measured before any of this: 3,037 tokens and 19.7s for a median run, 31.9s at p95."""
    from app.guardrails import Budgets

    assert Budgets(max_tokens_per_run=10_000, max_cost_usd_per_run=Decimal("0.002"), max_seconds_per_run=180)


def test_a_run_inside_every_ceiling_is_not_stopped():
    from app.guardrails import Budgets, over_budget

    ceilings = Budgets(max_tokens_per_run=10_000, max_cost_usd_per_run=Decimal("0.002"), max_seconds_per_run=180)

    assert over_budget(ceilings, tokens=3_000, cost_usd=Decimal("0.0005"), seconds=20) is None


def test_a_run_past_the_token_ceiling_says_which_ceiling_and_by_how_much():
    from app.guardrails import Budgets, over_budget

    ceilings = Budgets(max_tokens_per_run=10_000, max_cost_usd_per_run=Decimal("0.002"), max_seconds_per_run=180)

    reason = over_budget(ceilings, tokens=10_001, cost_usd=Decimal("0.0005"), seconds=20)

    assert reason is not None
    assert "10,001" in reason and "10,000" in reason and "token" in reason


def test_a_run_past_the_cost_ceiling_is_stopped_even_when_its_tokens_are_cheap():
    from app.guardrails import Budgets, over_budget

    ceilings = Budgets(max_tokens_per_run=10_000, max_cost_usd_per_run=Decimal("0.002"), max_seconds_per_run=180)

    reason = over_budget(ceilings, tokens=100, cost_usd=Decimal("0.003"), seconds=20)

    assert reason is not None and "$0.003" in reason


def test_a_run_past_the_time_ceiling_is_stopped():
    from app.guardrails import Budgets, over_budget

    ceilings = Budgets(max_tokens_per_run=10_000, max_cost_usd_per_run=Decimal("0.002"), max_seconds_per_run=180)

    reason = over_budget(ceilings, tokens=100, cost_usd=Decimal("0.0001"), seconds=181)

    assert reason is not None and "181" in reason and "second" in reason


def test_exactly_at_a_ceiling_is_still_inside_it():
    """The ceiling is what a run may spend, not the first amount it may not."""
    from app.guardrails import Budgets, over_budget

    ceilings = Budgets(max_tokens_per_run=10_000, max_cost_usd_per_run=Decimal("0.002"), max_seconds_per_run=180)

    assert over_budget(ceilings, tokens=10_000, cost_usd=Decimal("0.002"), seconds=180) is None


def test_a_ceiling_of_zero_stops_nothing_so_it_can_be_switched_off():
    """
    Unlike the refund limit, where zero is the kill switch, a budget of zero means no budget.

    A ceiling that stopped every run the moment it was set to zero would make the safe way to
    disable a budget indistinguishable from the most aggressive setting possible.
    """
    from app.guardrails import Budgets, over_budget

    off = Budgets(max_tokens_per_run=0, max_cost_usd_per_run=Decimal(0), max_seconds_per_run=0)

    assert over_budget(off, tokens=10_000_000, cost_usd=Decimal("9.99"), seconds=99_999) is None


@pytest.mark.db
def test_the_ceilings_are_read_from_the_row_an_operator_can_change(db):
    from app.guardrails import load, set_limits

    assert load(db).budgets.max_tokens_per_run > 0

    set_limits(db, max_tokens_per_run=2_000, by="an operator")

    assert load(db).budgets.max_tokens_per_run == 2_000


@pytest.mark.db
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"max_tokens_per_run": True}, "whole number"),
        ({"max_tokens_per_run": 1.5}, "whole number"),
        ({"max_seconds_per_run": -1}, "at least 0"),
        ({"max_cost_usd_per_run": 0.002}, "Decimal"),
        ({"max_cost_usd_per_run": Decimal("nan")}, "at least 0"),
        ({"max_cost_usd_per_run": Decimal("-0.001")}, "at least 0"),
    ],
    ids=["tokens true", "tokens fractional", "seconds negative", "cost float", "cost nan", "cost negative"],
)
def test_a_budget_that_is_not_a_budget_is_refused_before_it_reaches_the_database(db, change, message):
    """
    The database would take some of these. numeric(10,6) rounds a float happily, and a ceiling
    silently rounded is a ceiling nobody chose.
    """
    with pytest.raises(ValueError, match=message):
        set_limits(db, by="asha", **change)

    assert load(db).budgets == DEFAULT_BUDGETS


@pytest.mark.db
def test_a_change_must_change_something(db):
    with pytest.raises(ValueError, match="nothing to change"):
        set_limits(db, by="asha")


@pytest.mark.db
@pytest.mark.parametrize("asked", [Decimal("0.0000003"), Decimal("0.0000006"), Decimal("0.00123456789")])
def test_a_cost_ceiling_finer_than_the_column_is_refused(db, asked):
    """
    numeric(10,6) would round it, and here rounding is not a rounding error.

    Zero means no ceiling, so an operator asking for the strictest cost limit there is would have
    it quietly rounded away to the one setting that stops nothing -- and both the page and the
    command line would then say "no ceiling", which is exactly what they would say if that had
    been asked for.
    """
    with pytest.raises(ValueError, match="six decimal places"):
        set_limits(db, max_cost_usd_per_run=asked, by="asha")

    assert load(db).budgets == DEFAULT_BUDGETS


@pytest.mark.db
def test_a_cost_ceiling_the_column_can_hold_exactly_is_taken(db):
    assert set_limits(db, max_cost_usd_per_run=Decimal("0.000001"), by="asha").budgets.max_cost_usd_per_run == Decimal(
        "0.000001"
    )
