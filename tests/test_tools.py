"""
What the agent may propose, and the guarantee that it cannot do any of it yet.

This week the model is told these tools exist and asked what it would do. None
of them are wired to anything. That is not an honour-system promise in a
docstring -- there are no implementations in the registry to call, so proposing
a refund cannot issue one even by mistake.
"""

import re

from app.tools import TOOLS, describe_tools, described_argument


def test_the_agent_knows_about_the_tools_it_will_need():
    assert {tool.name for tool in TOOLS} == {
        "get_order",
        "search_policy",
        "issue_refund",
        "escalate_to_human",
    }


def test_nothing_in_the_registry_can_be_called():
    """
    The week's whole promise, enforced structurally.

    A tool here is a name, a description and a schema. If one ever gains a
    callable, this fails, and whoever added it has to say so out loud.
    """
    for tool in TOOLS:
        for value in vars(tool).values():
            assert not callable(value), f"{tool.name} carries something executable"


def test_each_tool_says_what_it_is_for():
    for tool in TOOLS:
        assert len(tool.description) > 20, f"{tool.name} is not described usefully"


def test_each_tool_declares_its_arguments():
    for tool in TOOLS:
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters


def test_no_two_tools_share_a_name():
    """The model picks by name, so a duplicate makes the choice ambiguous."""
    assert len({tool.name for tool in TOOLS}) == len(TOOLS)


def test_the_description_given_to_the_model_names_every_tool():
    described = describe_tools()

    for tool in TOOLS:
        assert tool.name in described


def test_money_is_described_in_paise_so_the_model_does_not_guess():
    """Rupees and paise differ by a factor of a hundred. Say which."""
    refund = next(tool for tool in TOOLS if tool.name == "issue_refund")

    assert "paise" in str(refund.parameters).lower()


def test_a_tool_can_be_left_out_of_the_description():
    described = describe_tools(exclude={"search_policy"})

    assert "search_policy" not in described
    assert "get_order" in described


# --- what the tool list costs to send ---------------------------------------------------------


def test_the_tool_list_says_what_a_tool_does_and_what_it_takes():
    """Everything the model needs to choose a tool and call it correctly."""
    described = describe_tools()

    for tool in TOOLS:
        assert tool.name in described
        assert tool.description in described, "the guidance is the part that earns its tokens"
        for argument in tool.parameters["properties"]:
            assert argument in described


def test_the_prompt_says_once_which_arguments_are_required():
    """Said per tool it was repeated four times; the convention belongs with the prompt, said once."""
    from tests.test_prompts import plan

    asked = plan()

    assert "required unless it says optional" in asked
    assert asked.count("required unless") == 1


def test_the_tool_list_keeps_the_hints_the_model_cannot_guess():
    """
    Paise against rupees is a factor of a hundred, and nothing in a message says which one.

    The example's shape matters as much as its presence. "Rs 1,234 is 123400" was tried and the
    model read "Rs 5,000" as 50000 in two cases; a round figure like the amounts customers write
    got all 30 duplicate-charge cases right.
    """
    described = describe_tools()

    assert "250000" in described and "paise" in described.lower()


def test_the_tool_list_names_no_number_the_model_could_copy_into_a_real_case():
    """
    This used to require the opposite: an example order id, "e.g. 4821", to show the shape.

    The model copied it. Replayed against a check that the customer wrote the order, 19 of the 150
    recorded cases changed -- "Can I pay with UPI?", "hi", "What time does support work until?" --
    because in every one the planner had looked up 4821, and none of those messages mention it.
    The test ledger has no 4821, so each lookup quietly found nothing and all 150 still passed. The
    demo ledger does have one, and it is Priya's.

    So no order number and no amount from either ledger may appear here. An example that happens to
    match a real entry is not an illustration; it is a value waiting to be pasted into a real case.
    """
    import json
    from pathlib import Path

    from app.seed import LEDGER

    described = describe_tools()
    root = Path(__file__).resolve().parent.parent
    for ledger in (LEDGER, root / "evals" / "ledger.json"):
        for order in json.loads(ledger.read_text())["orders"]:
            assert re.search(rf"\b{re.escape(str(order['id']))}\b", described) is None, (
                f"order {order['id']} from {ledger.name} is in the tool list"
            )
            assert str(order["amount_paise"]) not in described, (
                f"{order['amount_paise']}, order {order['id']}'s amount in {ledger.name}, is in the tool list"
            )


def test_the_tool_list_does_not_carry_the_validation_the_code_enforces():
    """
    maxLength, pattern, minimum and maximum are checked in app/contracts.py whatever the prompt
    says, so sending them buys nothing. Measured across the four plan snapshots they were 19% of
    the prompt, the tool list as a whole 49%, and the plan prompt is 82% of all this agent reads.
    """
    described = describe_tools()

    for noise in ("maxLength", "pattern", "minimum", "maximum", "'type': 'object'", "properties"):
        assert noise not in described, f"{noise} is enforced in code, not by asking nicely"


def test_the_tool_list_is_materially_smaller_than_the_schema_it_replaces():
    described = describe_tools()
    raw = "\n\n".join(f"- {tool.name}: {tool.description}\n  arguments: {tool.parameters}" for tool in TOOLS)

    assert len(described) < len(raw) * 0.7, f"{len(described)} against {len(raw)}"


def test_a_tool_left_out_is_left_out():
    assert "search_policy" not in describe_tools(exclude={"search_policy"})


def test_an_argument_that_may_be_left_out_says_so():
    """
    The prompt promises "required unless it says optional", and nothing else keeps that promise.

    Every argument on every tool is required today, so no real schema reaches this branch and no
    snapshot covers it. Were it to break, the first tool with an optional argument would be
    described to the model as demanding one it does not need -- and the prompt's one-line
    convention, which is what the per-tool `required` lists were traded away for, would be a lie.
    """
    assert described_argument("note", {"type": "string"}, required=False) == "    note (string, optional)"
    assert described_argument("note", {"type": "string"}, required=True) == "    note (string)"


def test_an_argument_with_no_description_still_names_its_shape():
    """A schema need not describe every argument; the line must not end on a dangling colon."""
    assert described_argument("count", {"type": "integer"}, required=True) == "    count (integer)"


def test_an_argument_of_no_stated_type_is_not_described_as_nothing():
    assert described_argument("thing", {}, required=True) == "    thing (value)"
