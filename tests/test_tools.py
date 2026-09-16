"""
What the agent may propose, and the guarantee that it cannot do any of it yet.

This week the model is told these tools exist and asked what it would do. None
of them are wired to anything. That is not an honour-system promise in a
docstring -- there are no implementations in the registry to call, so proposing
a refund cannot issue one even by mistake.
"""

from app.tools import TOOLS, describe_tools


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
    """Paise against rupees is a factor of a hundred, and an example order id shows the shape."""
    described = describe_tools()

    assert "360000" in described and "paise" in described.lower()
    assert "4821" in described


def test_the_tool_list_does_not_carry_the_validation_the_code_enforces():
    """
    maxLength, pattern, minimum and maximum are checked in app/contracts.py whatever the prompt
    says, so sending them buys nothing. They were 45% of the plan prompt, and the plan prompt is
    82% of everything this agent reads.
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
