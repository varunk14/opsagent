"""
The driver refuses an order tool when there is no order to act on.

`decide()` turns a planner proposal into the action a tick takes. Withholding the order tools from
the plan prompt (app/graph/prompts.py) makes the model less likely to reach for one when no order is
in play, but a live model can still emit `{"tool": "get_order"}` anyway. Left to run, that proposal
is executed, refused by the executor's ownership check, proposed again, and only then handed over --
the "repeated an earlier step" loop seen on real order-status messages. This guard hands the case to
a person on the first such proposal, deterministically, before anything runs.

These are pure tests of `decide()`: no database, no model. The end-to-end behaviour is covered by the
driver tests in test_run_agent.py, whose scenarios all carry an order and so never trip this guard.
"""

from decimal import Decimal
from typing import cast

from app.contracts import ExtractedRefund, ProposedAction
from app.graph.state import AgentState
from app.run_agent import decide


def proposal(tool: str, **args: object) -> ProposedAction:
    return ProposedAction(tool=tool, args=args, confidence=Decimal("0.6"), reasoning="x")


def state_with(prop: ProposedAction, extraction: ExtractedRefund | None) -> AgentState:
    return cast(AgentState, {"proposal": prop, "extraction": extraction})


def test_get_order_with_no_extraction_hands_over_without_running():
    """order_status/other never extracts, so extraction is None: nothing to look up."""
    prop, failure = decide(state_with(proposal("get_order", order_id="9999"), None), steps=[], max_steps=4)

    assert prop.tool == "escalate_to_human"
    assert failure is not None and "order" in failure.lower()


def test_issue_refund_with_no_extraction_hands_over():
    prop, failure = decide(
        state_with(proposal("issue_refund", order_id="9999", amount_paise=250000, reason="x"), None),
        steps=[],
        max_steps=4,
    )

    assert prop.tool == "escalate_to_human"
    assert failure is not None


def test_an_extraction_that_found_no_order_id_hands_over():
    extraction = ExtractedRefund(order_id=None, amount_paise=250000, reason="want it back")
    prop, failure = decide(state_with(proposal("get_order", order_id="9999"), extraction), steps=[], max_steps=4)

    assert prop.tool == "escalate_to_human"


def test_an_order_in_play_still_runs_the_lookup():
    """The guard must not overreach: a real order is looked up as before."""
    extraction = ExtractedRefund(order_id="4821", amount_paise=None, reason="charged twice")
    prop, failure = decide(state_with(proposal("get_order", order_id="4821"), extraction), steps=[], max_steps=4)

    assert (prop.tool, failure) == ("get_order", None)


def test_escalation_is_untouched_by_the_guard():
    prop, failure = decide(state_with(proposal("escalate_to_human", reason="unsure"), None), steps=[], max_steps=4)

    assert (prop.tool, failure) == ("escalate_to_human", None)
