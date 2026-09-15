"""
Preventing duplicate refunds: the fix, and the fake fix that passes review.

Three scenarios, same simulated outage every time:

  A. no key at all            -> 2 refunds   (the bug, for comparison)
  B. key made INSIDE the loop -> 2 refunds   (looks like a fix. is not.)
  C. key made from the INTENT -> 1 refund    (correct)

B is the one worth staring at. It imports the right concept, uses the right
vocabulary, and would sail through a pull request. It fixes nothing, because a
fresh key on every attempt means the bank has never seen it before.

The rule: an idempotency key identifies THE OPERATION, not THE ATTEMPT.

Run:  .venv/bin/python experiments/prevent_duplicate_refunds.py
"""

import time
import uuid
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# THE BANK. As in the bug reproduction, plus one thing: it remembers keys it honoured.
# ---------------------------------------------------------------------------
BANK_LEDGER: list[dict] = []
SEEN_KEYS: dict[str, dict] = {}
_call_count = 0


class NetworkTimeout(Exception):
    """The request was sent. We do not know whether it arrived."""


def reset_bank() -> None:
    global _call_count
    BANK_LEDGER.clear()
    SEEN_KEYS.clear()
    _call_count = 0


def bank_issue_refund(order_id: str, amount_paise: int, key: str | None = None) -> dict:
    """
    The bank, now idempotency-aware.

    If it has already honoured this key, it does NOT move money again. It returns
    the original result. Repeating the request has the same effect as making it
    once. That is the definition of idempotent.
    """
    global _call_count
    _call_count += 1

    # THE WHOLE FIX IS THESE THREE LINES.
    if key is not None and key in SEEN_KEYS:
        print(f"      [bank] key {key} already honoured -> replaying original result")
        return SEEN_KEYS[key]

    refund = {
        "refund_id": f"rf_{len(BANK_LEDGER) + 1:03d}",
        "order_id": order_id,
        "amount_paise": amount_paise,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    BANK_LEDGER.append(refund)
    if key is not None:
        SEEN_KEYS[key] = refund
    print(
        f"      [bank] processed {refund['refund_id']}: {amount_paise} paise"
        f"{f' (key {key})' if key else ''}"
    )

    # Same simulated outage: first request lands, its reply is lost.
    if _call_count == 1:
        print("      [bank] ...reply LOST in transit")
        raise NetworkTimeout("no response from payment gateway after 30s")

    return refund


def call_with_retry(fn, *args, max_attempts: int = 3, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"    attempt {attempt}:")
            result = fn(*args, **kwargs)
            print(f"    attempt {attempt}: SUCCESS")
            return result
        except NetworkTimeout:
            print(f"    attempt {attempt}: TIMEOUT")
            if attempt == max_attempts:
                raise
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# A. No key. The bug's behaviour, reproduced for comparison.
# ---------------------------------------------------------------------------
def scenario_a() -> None:  # pragma: no cover
    call_with_retry(bank_issue_refund, "4821", 360_000)


# ---------------------------------------------------------------------------
# B. THE FAKE FIX.
#
# A key is generated. It is passed to the bank. The word "idempotency" appears
# in the code. And it does nothing at all, because uuid4() runs again on every
# retry, so the bank sees a key it has never encountered.
# ---------------------------------------------------------------------------
def broken_refund_with_key(order_id: str, amount_paise: int) -> dict:
    key = f"refund:{uuid.uuid4()}"          # <-- generated per ATTEMPT
    print(f"      generated key: {key}")
    return bank_issue_refund(order_id, amount_paise, key=key)


def scenario_b() -> None:  # pragma: no cover
    call_with_retry(broken_refund_with_key, "4821", 360_000)


# ---------------------------------------------------------------------------
# C. THE REAL FIX.
#
# The key is derived from WHAT WE ARE TRYING TO DO, and is computed once, OUTSIDE
# the retry loop. Every attempt carries the same key, so the second one is
# recognised as a repeat.
#
# In the real system this becomes:  f"{run_id}:step_{n}:{tool_name}"
# ---------------------------------------------------------------------------
def scenario_c() -> None:  # pragma: no cover
    run_id, step_no, tool = "run_88", 5, "issue_refund"
    key = f"{run_id}:step_{step_no}:{tool}"     # <-- computed ONCE, from intent
    print(f"      operation key: {key}  (same on every attempt)")
    call_with_retry(bank_issue_refund, "4821", 360_000, key=key)


def report() -> int:  # pragma: no cover
    total = sum(e["amount_paise"] for e in BANK_LEDGER)
    print(
        f"\n  ledger: {len(BANK_LEDGER)} refund(s), total Rs {total / 100:,.2f}",
        end="",
    )
    print("   <-- correct" if total == 360_000 else "   <-- WRONG, overpaid")
    return total


def main() -> None:  # pragma: no cover
    scenarios = [
        ("A. no idempotency key", scenario_a),
        ("B. key generated inside the retry loop (the fake fix)", scenario_b),
        ("C. key derived from the operation (the real fix)", scenario_c),
    ]

    totals = {}
    for label, fn in scenarios:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        reset_bank()
        fn()
        totals[label[0]] = report()

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for letter, total in totals.items():
        verdict = "correct" if total == 360_000 else "OVERPAID"
        print(f"  {letter}:  Rs {total / 100:>8,.2f}   {verdict}")

    print(
        "\nB and C differ by ONE line: where the key is computed.\n"
        "Both compile. Both pass a smoke test that never times out.\n"
        "Only C survives the outage.\n"
        "\nThis is why the test that matters is 'call it twice with the same key\n"
        "and assert one refund' — not 'call it once and assert it worked'."
    )


if __name__ == "__main__":
    main()
