"""
The bug that costs real money: a retried refund pays the customer twice.

DELIBERATELY BROKEN. Run it, watch Priya get paid twice, then we fix it.

There is no LLM in this file. That is the point: this bug has nothing to do with
AI. It is a plain distributed-systems failure that agents hit constantly, because
agents retry aggressively and call tools that move money.

The setup:
  - issue_refund() talks to a "bank"
  - the bank PROCESSES the refund successfully
  - the acknowledgement is lost on the way back (timeout)
  - our retry logic does exactly what it was written to do, and retries
  - the bank processes it AGAIN

Nobody wrote a bug. Every component behaved correctly. Priya is still 3,600 rupees
up, and no exception was raised to tell you.

Run:  .venv/bin/python experiments/duplicate_refund_bug.py
"""

import time
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# THE "BANK". Synthetic. Money is integer paise. Timestamps are ISO 8601.
# ---------------------------------------------------------------------------
BANK_LEDGER: list[dict] = []

# Controls the simulated network failure. First call is delivered but its reply
# is lost; later calls get through cleanly.
_call_count = 0


class NetworkTimeout(Exception):
    """The request was sent. We do not know whether it arrived."""


def bank_issue_refund(order_id: str, amount_paise: int) -> dict:
    """
    Pretend this is a real payment API over the internet.

    Read this function carefully. It is not buggy. It does exactly one thing
    wrong from our point of view: it succeeds, and then the reply is lost.
    """
    global _call_count
    _call_count += 1

    # The bank does its job. Money moves. THIS ALWAYS HAPPENS.
    refund = {
        "refund_id": f"rf_{len(BANK_LEDGER) + 1:03d}",
        "order_id": order_id,
        "amount_paise": amount_paise,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    BANK_LEDGER.append(refund)
    print(f"      [bank] processed {refund['refund_id']}: {amount_paise} paise")

    # ...and then the response is lost in transit on the first attempt.
    if _call_count == 1:
        print("      [bank] ...reply LOST in transit (we never learn it worked)")
        raise NetworkTimeout("no response from payment gateway after 30s")

    return refund


# ---------------------------------------------------------------------------
# OUR CODE. Also not buggy. Retrying on a timeout is correct and necessary:
# most timeouts genuinely mean the request never arrived.
# ---------------------------------------------------------------------------
def call_with_retry(fn, *args, max_attempts: int = 3, **kwargs):
    """Retry with backoff. Standard, sensible, and here it is a loaded gun."""
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"    attempt {attempt}: calling {fn.__name__}...")
            result = fn(*args, **kwargs)
            print(f"    attempt {attempt}: SUCCESS")
            return result
        except NetworkTimeout as e:
            print(f"    attempt {attempt}: TIMEOUT ({e})")
            if attempt == max_attempts:
                raise
            backoff = 0.2 * attempt
            print(f"    backing off {backoff:.1f}s, then retrying")
            time.sleep(backoff)


def issue_refund(order_id: str, amount_paise: int) -> dict:
    """The tool our agent calls. One duplicate charge on order 4821."""
    return call_with_retry(bank_issue_refund, order_id, amount_paise)


def main() -> None:  # pragma: no cover
    print("=" * 70)
    print("Priya was charged twice for order 4821: 360000 paise (Rs 3,600) each.")
    print("The agent decides to refund ONE of them.")
    print("=" * 70)
    print()

    result = issue_refund("4821", 360_000)
    print(f"\n  agent believes it issued: {result['refund_id']}")

    print(f"\n{'=' * 70}")
    print("THE BANK LEDGER")
    print("=" * 70)
    for entry in BANK_LEDGER:
        rupees = entry["amount_paise"] / 100
        print(
            f"  {entry['refund_id']}  order {entry['order_id']}  "
            f"Rs {rupees:,.2f}  at {entry['at']}"
        )

    total = sum(e["amount_paise"] for e in BANK_LEDGER)
    print(f"\n  refunds issued : {len(BANK_LEDGER)}")
    print(f"  total refunded : Rs {total / 100:,.2f}")
    print(f"  should be      : Rs {360_000 / 100:,.2f}")

    if total > 360_000:
        overpaid = (total - 360_000) / 100
        print(f"\n  >>> OVERPAID BY Rs {overpaid:,.2f} <<<")
        print()
        print("  Note what did NOT happen:")
        print("    - no exception reached the caller")
        print("    - the agent reported success")
        print("    - a log search for ERROR finds nothing")
        print("    - you discover this when accounting does, weeks later")
        print()
        print("  Every component did its job. The bank processed what it received.")
        print("  Our retry did what retries are for. The bug is that neither side")
        print("  could tell 'a new refund' apart from 'the same refund again'.")
        print()
        print("  That distinction is what an idempotency key provides.")
        print("  See prevent_duplicate_refunds.py for the fix.")


if __name__ == "__main__":
    main()
