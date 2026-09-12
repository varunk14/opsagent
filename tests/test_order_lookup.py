"""
The tool layer and the whitelist that constrains what the model may ask for.

Again, no model is called. These cover the parts that must behave the same way
every time: what the tools return, and which tool calls are refused.
"""

import pytest
from pydantic import ValidationError

from experiments.order_lookup_agent import ToolCall, tool_escalate, tool_get_order


def test_get_order_returns_both_charges():
    order = tool_get_order("4821")

    assert order["order_id"] == "4821"
    assert len(order["charges"]) == 2
    assert {c["amount_paise"] for c in order["charges"]} == {360_000}


def test_get_order_reports_unknown_ids_rather_than_raising():
    """
    The agent has to be able to read and reason about the failure, so a missing
    order comes back as data. An exception here would end the run instead.
    """
    result = tool_get_order("does-not-exist")

    assert "error" in result
    assert "does-not-exist" in result["error"]


def test_get_order_requires_an_id():
    assert "error" in tool_get_order(None)
    assert "error" in tool_get_order("")


def test_escalate_always_succeeds():
    """Handing off to a person is the one action that must never fail."""
    result = tool_escalate("amount over the auto-approval limit")

    assert result["escalated"] is True
    assert result["reason"] == "amount over the auto-approval limit"


def test_escalate_without_a_reason_is_still_recorded():
    assert tool_escalate(None)["escalated"] is True


# --- the whitelist ---------------------------------------------------------


@pytest.mark.parametrize("tool", ["get_order", "escalate_to_human", "finish"])
def test_permitted_tools_are_accepted(tool):
    assert ToolCall.model_validate({"tool": tool}).tool == tool


@pytest.mark.parametrize(
    "invented",
    ["issue_refund", "delete_customer", "get_orders", "GET_ORDER", "drop_table"],
)
def test_invented_tools_are_rejected(invented):
    """
    The model will ask for tools that do not exist. The whitelist is what stops
    a plausible-sounding name from reaching an execution path.
    """
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"tool": invented})


def test_unexpected_arguments_are_rejected():
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"tool": "get_order", "amount_paise": 999})
