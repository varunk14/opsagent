"""
The tool catalogue, spoken over MCP -- and still unable to do any of it.

`app/tools.py` holds the tools as names, descriptions and JSON schemas and nothing executable, and
`tests/test_tools.py` proves a proposal cannot run. This is that same catalogue offered over the
Model Context Protocol, and it keeps the same promise. It is a second *description* of the tools,
never a second way to *run* them.

  tools/list  returns each tool's schema exactly as app/tools.py holds it -- one source of truth.
  tools/call  does not execute. It checks the arguments with the same `validate_tool_call` a run's
              own proposal is checked with, and returns the validated proposal. It opens no database,
              moves no money, and writes no run state. A malformed call comes back as a readable tool
              error, not an exception.

Execution stays exactly where it was: app/run_agent.py decides what runs, app/executor.py runs it,
behind idempotency keys, guardrails and approvals. There is deliberately no import of any of that
here, so there is no path from a call to an effect -- which is the point, because a refund must never
be one unauthenticated JSON-RPC call away.

Transport is stdio only, matching the rest of the project's loopback-only, no-network-exposure stance
(app/web.py binds 127.0.0.1 and refuses other hosts). Over stdio, stdout *is* the wire, so logging
goes to stderr where it cannot corrupt the JSON-RPC stream.

Run:  .venv/bin/python -m app.mcp_server        (a host launches this as a subprocess)
"""

import asyncio
import logging
import sys
from typing import Any

import mcp.types as mt
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

from app.contracts import validate_tool_call
from app.tools import TOOLS

SERVER_NAME = "opsagent"

# Attached to every call's result: this server describes and validates; it does not run anything.
NOT_EXECUTED = (
    "Proposed, not executed. This server validates a tool call against opsagent's own schema and "
    "returns the proposal; refunds and every other effect happen only through the durable pipeline "
    "(run_agent / executor), never over MCP."
)


def mcp_tools() -> list[mt.Tool]:
    """The registry as MCP tools: each tool's input schema is the exact dict app/tools.py holds."""
    return [
        mt.Tool(name=tool.name, description=tool.description, input_schema=tool.parameters)
        for tool in TOOLS
    ]


async def list_tools(
    ctx: ServerRequestContext[Any], params: mt.PaginatedRequestParams | None
) -> mt.ListToolsResult:
    """Advertise the catalogue. Reads nothing, changes nothing."""
    return mt.ListToolsResult(tools=mcp_tools())


async def call_tool(
    ctx: ServerRequestContext[Any], params: mt.CallToolRequestParams
) -> mt.CallToolResult:
    """Validate a proposed call the same way a run's proposal is validated, and return it unrun."""
    arguments = dict(params.arguments or {})
    try:
        validate_tool_call(params.name, arguments)
    except ValueError as refused:
        return mt.CallToolResult(
            content=[mt.TextContent(type="text", text=str(refused))], is_error=True
        )

    proposal: dict[str, Any] = {"tool": params.name, "args": arguments}
    return mt.CallToolResult(
        content=[mt.TextContent(type="text", text=f"{proposal}\n\n{NOT_EXECUTED}")],
        structured_content=proposal,
    )


def build_server() -> Server[None]:
    """The stdio server, its handlers wired in. No capability beyond listing and validating exists."""
    return Server(SERVER_NAME, on_list_tools=list_tools, on_call_tool=call_tool)


async def serve() -> None:  # pragma: no cover - needs a real stdio peer
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> int:  # pragma: no cover - the interactive driver
    # stdout is the wire; send logs to stderr so they cannot corrupt the JSON-RPC stream.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(serve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
