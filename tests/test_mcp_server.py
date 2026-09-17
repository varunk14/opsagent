"""
The tools, described a second way -- over MCP -- and the guarantee that the second way still cannot
do any of it.

`app/tools.py` says these tools exist and carries nothing executable; `tests/test_tools.py` proves a
proposal cannot run. The MCP server is that same catalogue spoken over a standard protocol, and it
must keep the same promise: listing a tool, or being asked to call one, moves no money and touches no
database. Execution stays where it always was -- the durable pipeline in run_agent/executor -- and
there is no path to it from here.

So these tests list the tools and check the schemas are the registry's own, and they ask the server
to `issue_refund` an order that really exists in a loaded ledger and prove no refund and no run
appear. A malformed call comes back as a readable tool error, not an exception, and not a refund.
"""

import asyncio
from pathlib import Path

import mcp.types as mt
import psycopg
import pytest

from app.mcp_server import call_tool, list_tools, mcp_tools
from app.tools import TOOLS
from tests.test_run_agent import ledger

pytestmark = pytest.mark.db


def run(coro):
    return asyncio.run(coro)


def call(name: str, arguments: dict) -> mt.CallToolResult:
    return run(call_tool(None, mt.CallToolRequestParams(name=name, arguments=arguments)))


def counts(dsn: str) -> tuple[int, int]:
    with psycopg.connect(dsn) as connection:
        refunds = connection.execute("SELECT count(*) FROM refunds").fetchone()[0]
        runs = connection.execute("SELECT count(*) FROM runs").fetchone()[0]
    return refunds, runs


# --- the catalogue ------------------------------------------------------------


def test_it_lists_exactly_the_registry_tools():
    listed = run(list_tools(None, None))
    assert {tool.name for tool in listed.tools} == {tool.name for tool in TOOLS} == {
        "get_order",
        "search_policy",
        "issue_refund",
        "escalate_to_human",
    }


def test_each_schema_is_the_registrys_own():
    """One source of truth: the MCP schema is the exact dict app/tools.py already holds."""
    by_name = {tool.name: tool for tool in TOOLS}
    for tool in run(list_tools(None, None)).tools:
        assert tool.input_schema == by_name[tool.name].parameters


def test_the_server_has_no_path_to_execution():
    """
    Structural, like test_tools' no-callables check. Nothing here imports the code that runs a tool
    or opens a database, so there is no path from a call to an effect, whatever a handler does.
    """
    source = Path(__file__).resolve().parent.parent.joinpath("app", "mcp_server.py").read_text()
    for forbidden in ("app.executor", "app.run_agent", "app.db", "from app.db", "connect("):
        assert forbidden not in source, f"mcp_server reaches execution via {forbidden}"


def test_the_server_speaks_no_network_transport():
    """
    stdio only, by construction. The SDK ships SSE and HTTP transports too; wiring one here would
    put the tools on a network, against the whole project's loopback-only stance. Locked structurally
    so a future edit that reaches for a port has to delete this test and say so.
    """
    source = Path(__file__).resolve().parent.parent.joinpath("app", "mcp_server.py").read_text()
    for forbidden in ("sse", "streamable_http", "streamable-http", "uvicorn", "starlette"):
        assert forbidden not in source, f"mcp_server reaches a network transport via {forbidden}"


# --- a call cannot execute ----------------------------------------------------


def test_a_refund_call_moves_no_money_and_makes_no_run(fresh_database):
    """The order exists in the ledger; asking MCP to refund it must still pay nothing."""
    ledger(fresh_database)
    before = counts(fresh_database)

    result = call("issue_refund", {"order_id": "4821", "amount_paise": 250000, "reason": "duplicate"})

    assert counts(fresh_database) == before == (0, 0)
    assert result.is_error is not True  # it is a proposal, not an error


def test_a_wellformed_call_returns_the_proposal_not_a_result():
    result = call("get_order", {"order_id": "4821"})

    assert result.is_error is not True
    assert result.structured_content is not None
    assert result.structured_content.get("tool") == "get_order"
    assert result.structured_content.get("args", {}).get("order_id") == "4821"
    text = " ".join(part.text for part in result.content if isinstance(part, mt.TextContent)).lower()
    assert "propos" in text  # it says this is a proposal


# --- malformed and unknown calls are readable errors, not exceptions ----------


def test_a_malformed_argument_is_a_readable_error():
    """An order id that fails the pattern is refused the same way a proposal would be, not raised."""
    result = call("issue_refund", {"order_id": "not a real id!", "amount_paise": 250000, "reason": "x"})

    assert result.is_error is True
    text = " ".join(part.text for part in result.content if isinstance(part, mt.TextContent))
    assert "order_id" in text


def test_a_missing_argument_is_a_readable_error():
    result = call("issue_refund", {"order_id": "4821"})  # no amount_paise, no reason

    assert result.is_error is True


def test_an_unknown_tool_is_a_readable_error():
    result = call("delete_everything", {})

    assert result.is_error is True


# --- nothing exposed is executable (mirror of test_tools) ---------------------


def test_no_exposed_tool_declares_server_side_execution():
    """A local tool has no `execution` block; that is what keeps calling it a proposal, not a run."""
    for tool in mcp_tools():
        assert tool.execution is None
