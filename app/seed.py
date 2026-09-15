"""
Loading the fictional ledger the agent acts on.

Every order in fixtures/inbox.jsonl has a row in fixtures/ledger.json, so a real
get_order finds something. Loading is safe to repeat: customers and orders are
inserted with ON CONFLICT DO NOTHING, and an order's charges are written only
when the order itself was new, because a charge has no natural key to conflict on.

Run:  .venv/bin/python -m app.seed
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

from app.db import apply_migrations, connect

LEDGER = Path(__file__).resolve().parent.parent / "fixtures" / "ledger.json"


@dataclass(frozen=True)
class LoadedLedger:
    """How many rows were new. All zero on every load after the first."""

    customers: int
    orders: int
    charges: int


def load_ledger(connection: psycopg.Connection, path: Path = LEDGER) -> LoadedLedger:
    """Insert what is missing. Does not commit; the caller owns the transaction."""
    ledger = json.loads(path.read_text())

    customers = 0
    for customer in ledger["customers"]:
        inserted = connection.execute(
            "INSERT INTO customers (email, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (customer["email"], customer["name"]),
        )
        customers += inserted.rowcount

    orders = charges = 0
    for order in ledger["orders"]:
        inserted = connection.execute(
            "INSERT INTO orders (id, customer_email, amount_paise, status) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (order["id"], order["customer_email"], order["amount_paise"], order["status"]),
        )
        if inserted.rowcount == 0:
            continue
        orders += 1
        for amount in order["charges_paise"]:
            connection.execute(
                "INSERT INTO charges (order_id, amount_paise) VALUES (%s, %s)", (order["id"], amount)
            )
            charges += 1

    return LoadedLedger(customers=customers, orders=orders, charges=charges)


def main() -> int:  # pragma: no cover - the interactive loader
    with connect() as connection:
        apply_migrations(connection)
        loaded = load_ledger(connection)
    print(f"  loaded {loaded.customers} customers, {loaded.orders} orders, {loaded.charges} charges")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
