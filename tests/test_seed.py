"""
The fictional ledger the agent acts on.

Every order in fixtures/inbox.jsonl needs a row behind it, or the first real
get_order would find nothing. Loading must be safe to repeat, like migrations.
"""

import pytest

from app.seed import load_ledger

pytestmark = pytest.mark.db


def test_priyas_order_shows_the_two_charges_she_describes(db):
    load_ledger(db)

    charges = db.execute(
        "SELECT amount_paise FROM charges WHERE order_id = '4821' ORDER BY id"
    ).fetchall()
    customer = db.execute("SELECT customer_email FROM orders WHERE id = '4821'").fetchone()[0]

    assert [amount for (amount,) in charges] == [360_000, 360_000]
    assert customer == "priya@example.com"


def test_every_order_the_inbox_mentions_exists(db):
    load_ledger(db)

    found = {order_id for (order_id,) in db.execute("SELECT id FROM orders").fetchall()}

    assert {"4821", "3310", "5102"} <= found


def test_loading_twice_adds_nothing(db):
    first = load_ledger(db)
    second = load_ledger(db)

    assert first.charges > 0
    assert second.customers == second.orders == second.charges == 0
    assert db.execute("SELECT count(*) FROM charges WHERE order_id = '4821'").fetchone()[0] == 2
