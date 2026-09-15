"""
Click any run and see the full nested trace, with the cost of every step.

A refund run is worked by the real driver with a scripted model. The runs list links
to it; its page shows each tick with classify, extract, retrieve, plan and act nested
beneath it, each model call beneath its step with its own cost, and those costs add
up to what the run recorded. A run that waits for a person and is paid after approval
is one trace: the tick that asked, the tick that paid, and the approval between them.
"""

import re
from decimal import Decimal

import psycopg
import pytest

from tests.test_approval_path import decide_on, refund_model, work
from tests.test_run_agent import happy_model, ledger, queue
from tests.test_web import client_for

pytestmark = pytest.mark.db

ROW = re.compile(
    r'<tr class="span[^"]*" data-depth="(\d+)">\s*'
    r'<td class="name">([^<]+)</td>\s*'
    r'<td class="ms">\d+</td>\s*'
    r'<td class="model">[^<]*</td>\s*'
    r'<td class="version">[^<]*</td>\s*'
    r'<td class="tokens">[^<]*</td>\s*'
    r'<td class="cost">([^<]*)</td>\s*'
    r'<td class="did">([^<]*)</td>'
)
STEPS = ["classify", "extract", "retrieve", "plan", "act"]


def trace_table(html: str) -> list[tuple[int, str, str, str]]:
    """Every row of the trace: depth, span name, cost cell, what it did."""
    return [(int(depth), name, cost, did) for depth, name, cost, did in ROW.findall(html)]


def recorded_cost(dsn: str, run_id: str) -> Decimal:
    with psycopg.connect(dsn) as connection:
        (cost,) = connection.execute("SELECT cost_usd FROM runs WHERE id = %s", (run_id,)).fetchone()
    return cost


def open_from_the_list(dsn: str, run_id: str) -> str:
    client = client_for(dsn)
    listed = client.get("/runs").text
    (link,) = re.findall(rf'href="(/runs/{run_id})"', listed)
    return client.get(link).text


def test_any_run_opens_from_the_list_as_its_nested_trace_with_every_steps_cost(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, happy_model())

    table = trace_table(open_from_the_list(fresh_database, run_id))

    ticks = [index for index, (depth, *_rest) in enumerate(table) if depth == 0]
    assert len(ticks) == 2
    first_tick = table[ticks[0] + 1 : ticks[1]]
    assert [name for depth, name, _cost, _did in first_tick if depth == 1] == STEPS
    calls = [(name, cost) for depth, name, cost, _did in table if depth == 2 and name.endswith(".generate")]
    assert {name for name, _cost in calls} >= {"classify.generate", "extract.generate", "plan.generate"}
    assert all(re.fullmatch(r"\$\d+\.\d+", cost) for _name, cost in calls), calls
    total = sum((Decimal(cost[1:]) for _name, cost in calls), Decimal(0))
    assert total.quantize(Decimal("0.000001")) == recorded_cost(fresh_database, run_id)


def test_a_run_paid_after_approval_is_one_trace_with_the_approval_and_the_payment(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(720_000))
    decide_on(fresh_database, run_id, approved=True)
    work(fresh_database, refund_model(720_000))

    html = open_from_the_list(fresh_database, run_id)

    table = trace_table(html)
    acts = [did for depth, name, _cost, did in table if depth == 1 and name == "act"]
    assert any("approval opened" in did for did in acts), acts
    assert "refunded" in acts[-1]
    assert [depth for depth, *_rest in table].count(0) >= 2
    assert '<p class="reason">Rs 7,200 is not under the Rs 5,000 limit' in html
    assert re.search(r"<dt>Decided by</dt><dd>asha</dd>", html)
    assert "<dt>Paid</dt>" in html
