"""
The failure taxonomy: every failed run gets one of six named categories, and no run gets two.

"It broke" is not actionable. A run that failed is classified from what it left behind --
its stop reason, its steps and what they returned, what was paid or put to a person -- into
one fixed category, by a fixed priority, so the mix over time says which kind is growing:

- context_overflow: the customer's text was too long for the prompt and was cut;
- loop: the planner repeated a step or used the whole step budget;
- tool_misuse: a tool that does not run here, arguments the ledger refused, a lookup of an
  order the ledger does not know, a refund proposed before any lookup;
- hallucinated_field: an order id or amount in the extraction or the proposal that appears in
  neither the customer's message nor any tool result (rupees in the message count as paise);
- wrong_escalation: a person rejected what the agent proposed (with a label, in evals: paid
  when a person should decide, or escalated something easy);
- drift: only ever assigned by evals, from history.

A run that succeeded, or one that died of infrastructure (an outage, an expired lock), has no
category: those already have failure_class and the dead-letter list. The worker writes the
category when a run rests, in the same transaction as the step; a backfill classifies runs
that rested before the column existed, once.
"""

import json

import psycopg
import pytest

from app.failures import FailureCategory, RunRest, backfill, classify, rest_of
from evals.golden import load_cases
from evals.runner import run_cases
from tests.fakes import CLASSIFIED_DUPLICATE, FakeEmbedder, ScriptedModel
from tests.test_evals_replay import script_for

CASES = {case.id: case for case in load_cases()}
LOOKUP = {"tool": "get_order", "args": {"order_id": "4821"}, "result": {"order_id": "4821", "charges_paise": [360000, 360000]}}
REFUND = {"tool": "issue_refund", "args": {"order_id": "4821", "amount_paise": 360000}, "result": {"refunded": True}}
MESSAGE = "Charged twice for order #4821\n\nHi, I was charged twice for order #4821, Rs 3,600 each time."


def rest(**changes) -> RunRest:
    """A run that paid one duplicate charge correctly; change what the test needs."""
    fields = {
        "status": "done",
        "failure_class": None,
        "failure": None,
        "steps": [LOOKUP, REFUND],
        "proposal": {"tool": "issue_refund", "args": {"order_id": "4821", "amount_paise": 360000}},
        "extraction": {"order_id": "4821", "amount_paise": 360000},
        "refunds_paise": (360000,),
        "approval_paise": None,
        "approval_status": None,
        "message_text": MESSAGE,
    }
    return RunRest(**{**fields, **changes})


def looping_model(case) -> ScriptedModel:
    lookup = json.dumps({"tool": "get_order", "args": {"order_id": case.expect.order_id}, "confidence": 0.9, "reasoning": "look"})
    return ScriptedModel(
        classify=CLASSIFIED_DUPLICATE,
        extract=f'{{"order_id": "{case.expect.order_id}", "amount_paise": null, "reason": "twice"}}',
        plan=[lookup, lookup],
    )


# --- the six categories ------------------------------------------------------------------------


def test_the_taxonomy_is_exactly_the_six_categories():
    assert [category.value for category in FailureCategory] == [
        "hallucinated_field",
        "tool_misuse",
        "loop",
        "context_overflow",
        "wrong_escalation",
        "drift",
    ]


def test_a_run_that_did_what_it_should_has_no_category():
    assert classify(rest()) is None


def test_a_run_waiting_for_a_person_to_approve_is_not_a_failure():
    waiting = rest(status="waiting_approval", refunds_paise=(), approval_paise=360000, approval_status="pending")

    assert classify(waiting) is None


@pytest.mark.parametrize(
    "failure_class", ["model_unavailable", "service_unavailable", "lock_expired", "policy_search_unavailable"]
)
def test_a_run_that_died_of_infrastructure_keeps_its_failure_class_and_gets_no_category(failure_class):
    dead = rest(status="dead", failure_class=failure_class, steps=[], refunds_paise=(), proposal=None)

    assert classify(dead) is None


@pytest.mark.parametrize("failure", ["plan: repeated an earlier step", "plan: step budget of 4 used"])
def test_repeating_a_step_or_using_the_whole_budget_is_a_loop(failure):
    handed_over = rest(status="waiting_approval", failure=failure, refunds_paise=(), steps=[LOOKUP])

    assert classify(handed_over) is FailureCategory.LOOP


def test_being_deferred_by_the_rate_limit_until_a_person_must_take_it_is_a_loop():
    deferred = rest(status="waiting_approval", refunds_paise=(), failure="plan: deferred 3 times by the rate limit")

    assert classify(deferred) is FailureCategory.LOOP


@pytest.mark.parametrize(
    "failure",
    [
        "plan: send_gift_card is not a tool this worker runs",
        "act: the ledger refused the refund: refund of 149700 paise on order 4821 would exceed the 99800 paise charged",
        "act: the approved action could not be read (2 validation errors)",
    ],
)
def test_the_wrong_tool_or_the_wrong_arguments_is_tool_misuse(failure):
    assert classify(rest(status="waiting_approval", failure=failure, refunds_paise=())) is FailureCategory.TOOL_MISUSE


def test_looking_up_an_order_the_ledger_does_not_know_is_tool_misuse():
    unknown = {"tool": "get_order", "args": {"order_id": "9999"}, "result": {"error": "no order 9999"}}
    handed_over = rest(
        status="waiting_approval", steps=[unknown], refunds_paise=(), proposal=None, extraction={"order_id": "9999"}
    )

    assert classify(handed_over) is FailureCategory.TOOL_MISUSE


def test_a_refund_proposed_before_any_lookup_is_tool_misuse():
    blind = rest(status="waiting_approval", steps=[], refunds_paise=(), approval_paise=360000, approval_status="pending")

    assert classify(blind) is FailureCategory.TOOL_MISUSE


def test_an_order_id_that_appears_nowhere_is_a_hallucinated_field():
    invented = rest(
        status="waiting_approval",
        steps=[],
        refunds_paise=(),
        extraction={"order_id": "7777", "amount_paise": None},
        proposal={"tool": "escalate_to_human", "args": {"reason": "unsure"}},
    )

    assert classify(invented) is FailureCategory.HALLUCINATED_FIELD


def test_an_amount_that_appears_nowhere_is_a_hallucinated_field():
    invented = rest(
        status="waiting_approval",
        refunds_paise=(),
        approval_paise=990000,
        approval_status="pending",
        proposal={"tool": "issue_refund", "args": {"order_id": "4821", "amount_paise": 990000}},
    )

    assert classify(invented) is FailureCategory.HALLUCINATED_FIELD


def test_rupees_in_the_message_count_as_paise():
    """Found in Unit 0: "Rs 5,997" in the message is 599700 paise, not an invented number."""
    stated = rest(
        status="waiting_approval",
        refunds_paise=(),
        approval_paise=599700,
        approval_status="pending",
        steps=[LOOKUP],
        message_text="Order #4821 charged twice, Rs 5,997 each. Refund one.",
        proposal={"tool": "issue_refund", "args": {"order_id": "4821", "amount_paise": 599700}},
    )

    assert classify(stated) is None


def test_a_proposal_a_person_rejected_is_a_wrong_escalation():
    rejected = rest(
        status="done", refunds_paise=(), approval_paise=360000, approval_status="rejected", failure_class="rejected"
    )

    assert classify(rejected) is FailureCategory.WRONG_ESCALATION


def test_a_message_too_long_for_the_prompt_is_a_context_overflow():
    from app.graph.prompts import MAX_CUSTOMER_TEXT

    cut = rest(
        status="waiting_approval",
        refunds_paise=(),
        failure="plan: repeated an earlier step",
        message_text="x" * (MAX_CUSTOMER_TEXT + 1),
    )

    assert classify(cut) is FailureCategory.CONTEXT_OVERFLOW


def test_when_two_signals_apply_the_earlier_in_the_priority_wins():
    """A loop that also proposed a refund before any lookup is a loop: the priority is fixed, one category per run."""
    both = rest(status="waiting_approval", failure="plan: repeated an earlier step", steps=[], refunds_paise=())

    assert classify(both) is FailureCategory.LOOP


def test_a_hand_over_the_agent_chose_with_nothing_else_wrong_has_no_category_without_a_label():
    """Whether an escalation was needed is only known with a label; evals decides that, not the worker."""
    chosen = rest(
        status="waiting_approval",
        refunds_paise=(),
        steps=[LOOKUP],
        proposal={"tool": "escalate_to_human", "args": {"reason": "not sure the second charge is a duplicate"}},
    )

    assert classify(chosen) is None


# --- read from a real run, written by the worker, backfilled --------------------------------------


@pytest.mark.db
def test_a_resting_run_is_read_back_as_what_it_left_behind(fresh_database):
    case = CASES["n-001"]
    run_cases(fresh_database, [case], script_for(case.expect.order_id, case.expect.refund_paise), FakeEmbedder())

    with psycopg.connect(fresh_database) as connection:
        (run_id,) = connection.execute("SELECT id FROM runs").fetchone()
        left = rest_of(connection, run_id)

    assert (left.status, left.failure, left.refunds_paise) == ("done", None, (case.expect.refund_paise,))
    assert [step["tool"] for step in left.steps] == ["get_order", "issue_refund"]
    assert case.message.body in left.message_text
    assert left.extraction["order_id"] == case.expect.order_id


@pytest.mark.db
def test_the_worker_writes_the_category_when_a_run_rests(fresh_database):
    looping, paying = CASES["n-002"], CASES["n-001"]

    run_cases(fresh_database, [looping], looping_model(looping), FakeEmbedder())
    run_cases(fresh_database, [paying], script_for(paying.expect.order_id, paying.expect.refund_paise), FakeEmbedder())

    with psycopg.connect(fresh_database) as connection:
        rows = connection.execute("SELECT status, failure_category FROM runs ORDER BY created_at").fetchall()
    assert rows == [("waiting_approval", "loop"), ("done", None)]


@pytest.mark.db
def test_the_category_column_refuses_anything_outside_the_taxonomy(fresh_database):
    case = CASES["n-001"]
    run_cases(fresh_database, [case], script_for(case.expect.order_id, case.expect.refund_paise), FakeEmbedder())

    with psycopg.connect(fresh_database) as connection, pytest.raises(psycopg.errors.CheckViolation):
        connection.execute("UPDATE runs SET failure_category = 'it broke'")


@pytest.mark.db
def test_backfill_classifies_runs_that_rested_before_the_column_existed_once(fresh_database):
    looping = CASES["n-002"]
    run_cases(fresh_database, [looping], looping_model(looping), FakeEmbedder())

    with psycopg.connect(fresh_database) as connection:
        connection.execute("UPDATE runs SET failure_category = NULL")
        first = backfill(connection)
        second = backfill(connection)
        (category,) = connection.execute("SELECT failure_category FROM runs").fetchone()

    assert (first, second, category) == (1, 0, "loop")


@pytest.mark.db
def test_a_run_that_is_not_in_the_database_is_refused_by_id(fresh_database):
    from uuid import uuid4

    missing = uuid4()
    with psycopg.connect(fresh_database) as connection, pytest.raises(LookupError, match=str(missing)):
        rest_of(connection, missing)


@pytest.mark.db
def test_a_person_rejecting_what_the_agent_proposed_is_recorded_as_a_wrong_escalation(fresh_database):
    """Through the real decision, not a hand-made rest: the rejection and the category land in one transaction."""
    from app.approvals import decide

    waiting = CASES["n-021"]
    run_cases(fresh_database, [waiting], script_for(waiting.expect.order_id, waiting.expect.refund_paise), FakeEmbedder())

    with psycopg.connect(fresh_database) as connection:
        (approval_id,) = connection.execute("SELECT id FROM approvals").fetchone()
        assert decide(connection, approval_id, approved=False, by="reviewer")
        (status, category) = connection.execute("SELECT status, failure_category FROM runs").fetchone()

    assert (status, category) == ("done", "wrong_escalation")


@pytest.mark.db
def test_the_mix_is_counted_per_week_and_category(fresh_database):
    from app.failures import mix_by_week

    looping, paying = CASES["n-002"], CASES["n-001"]
    run_cases(fresh_database, [looping], looping_model(looping), FakeEmbedder())
    run_cases(fresh_database, [paying], script_for(paying.expect.order_id, paying.expect.refund_paise), FakeEmbedder())

    with psycopg.connect(fresh_database) as connection:
        rows = mix_by_week(connection)

    assert [(category, count) for _, category, count in rows] == [("loop", 1)]
