"""
The driver refuses an order tool proposed for an order the customer never wrote.

`decide()` turns a planner proposal into the action a tick takes. Withholding the order tools from
the plan prompt (app/graph/prompts.py) makes the model less likely to reach for one when no order is
in play, but a live model can still emit `{"tool": "get_order", "args": {"order_id": ...}}` for an
order the message never named. Left to run, that proposal is executed, refused by the executor's
mention check, proposed again, and only then handed over -- the "repeated an earlier step" loop seen
on real order-status messages. This guard hands the case to a person on the first such proposal,
deterministically, before anything runs. It asks the same question the executor's `mentioned` does --
is this order number written in the message? -- through the shared `names_order`, so the two agree.

These are pure tests of `decide()`: no database, no model. The end-to-end behaviour is covered by the
driver tests in test_run_agent.py, whose scenarios all name the order they act on.
"""

from decimal import Decimal
from typing import cast

from app.contracts import ProposedAction
from app.graph.state import AgentState
from app.run_agent import decide

SUBJECT = "Charged twice for order #4821"
BODY = "Hi, I think I was charged twice for order #4821 last Tuesday."


def proposal(tool: str, **args: object) -> ProposedAction:
    return ProposedAction(tool=tool, args=args, confidence=Decimal("0.6"), reasoning="x")


def state_with(prop: ProposedAction, *, subject: str | None = SUBJECT, body: str = BODY) -> AgentState:
    return cast(AgentState, {"proposal": prop, "subject": subject, "body": body})


def test_get_order_for_an_order_the_message_never_named_hands_over():
    """The classic loop: a status question naming no order, an invented id proposed anyway."""
    state = state_with(proposal("get_order", order_id="9999"), subject=None, body="where is my order?")
    prop, failure = decide(state, steps=[], max_steps=4)

    assert prop.tool == "escalate_to_human"
    assert failure is not None and "order" in failure.lower()


def test_issue_refund_for_an_unnamed_order_hands_over():
    state = state_with(
        proposal("issue_refund", order_id="9999", amount_paise=250000, reason="x"),
        subject=None,
        body="please refund me",
    )
    prop, failure = decide(state, steps=[], max_steps=4)

    assert prop.tool == "escalate_to_human"
    assert failure is not None


def test_get_order_the_customer_named_still_runs():
    """The guard must not overreach: an order written in the message is looked up as before."""
    prop, failure = decide(state_with(proposal("get_order", order_id="4821")), steps=[], max_steps=4)

    assert (prop.tool, failure) == ("get_order", None)


def test_an_order_named_only_in_the_subject_still_runs():
    state = state_with(proposal("get_order", order_id="4821"), body="Hi, please look into this. Thanks.")
    prop, failure = decide(state, steps=[], max_steps=4)

    assert (prop.tool, failure) == ("get_order", None)


def test_issue_refund_for_a_named_order_is_not_blocked_by_the_guard():
    prop, failure = decide(
        state_with(proposal("issue_refund", order_id="4821", amount_paise=250000, reason="charged twice")),
        steps=[],
        max_steps=4,
    )

    assert (prop.tool, failure) == ("issue_refund", None)


def test_escalation_is_untouched_by_the_guard():
    state = state_with(proposal("escalate_to_human", reason="unsure"), subject=None, body="where is my order?")
    prop, failure = decide(state, steps=[], max_steps=4)

    assert (prop.tool, failure) == ("escalate_to_human", None)
