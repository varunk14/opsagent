"""
Extracting refund details from a customer email: the raw truth.

Zero dependencies. urllib + json only. No pydantic, no langchain, no framework.

The point: see with your own eyes that a model returns TEXT, not DATA.
Everything built after this exists because of what you are about to see.

Run:  .venv/bin/python experiments/extract_refund_details.py       (once)
      .venv/bin/python experiments/extract_refund_details.py 5   (five times)
"""

import json
import sys
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

# The running example for the entire project.
PRIYA_EMAIL = (
    "Hi, I think I was charged twice for order #4821 last Tuesday. "
    "Can you refund one of them? Thanks, Priya"
)

# Note how reasonable this prompt looks. That is the trap.
PROMPT = f"""Extract the refund request details from this customer email.

Email:
{PRIYA_EMAIL}

Return JSON with these fields:
- order_id: string
- reason: one of "double_charge", "damaged", "not_received", "other"
- amount_paise: integer, the amount in paise, or null if not stated
- confidence: float between 0.0 and 1.0
"""


def ask_model(prompt: str) -> str:  # pragma: no cover
    """POST to Ollama and return the model's raw response text, untouched."""
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            # DELIBERATELY NOT SET: "format": "json"
            # Ollama can force valid JSON. Turning it on here would hide
            # the exact problem this script exists to show you.
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        envelope = json.loads(resp.read().decode("utf-8"))

    # Ollama's own envelope IS valid JSON. The model's answer, inside it, is not.
    return envelope["response"]


def attempt(n: int) -> bool:  # pragma: no cover
    """One call. Show the raw text, then try to parse it. Return True if it parsed."""
    print(f"\n{'=' * 70}")
    print(f"ATTEMPT {n}")
    print("=" * 70)

    try:
        raw = ask_model(PROMPT)
    except urllib.error.URLError as e:
        print(f"\n  Could not reach Ollama at {OLLAMA_URL}")
        print(f"  {e}")
        print("\n  Is it running?   ollama serve")
        sys.exit(1)

    # repr() not print() — so whitespace, newlines and fences are visible.
    print("\n--- RAW, as repr() ---")
    print(repr(raw))

    print("\n--- RAW, as printed ---")
    print(raw)

    print("\n--- json.loads() on that exact string ---")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  FAILED: {e.__class__.__name__}: {e}")
        return False

    print(f"  OK, parsed to {type(parsed).__name__}: {parsed}")

    # Parsing is not the same as being correct. Check the shape too.
    problems = []
    if not isinstance(parsed, dict):
        problems.append(f"expected an object, got {type(parsed).__name__}")
    else:
        for field in ("order_id", "reason", "amount_paise", "confidence"):
            if field not in parsed:
                problems.append(f"missing field: {field}")
        if isinstance(parsed.get("amount_paise"), float):
            problems.append("amount_paise is a float — money must never be a float")

    if problems:
        print("  ...but the SHAPE is wrong:")
        for p in problems:
            print(f"    - {p}")
        return False

    return True


def main() -> None:  # pragma: no cover
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    results = [attempt(i) for i in range(1, runs + 1)]

    good = sum(results)
    print(f"\n{'=' * 70}")
    print(f"SCOREBOARD: {good}/{runs} usable")
    print("=" * 70)
    if good == runs:
        print("All clean this time. Run it again with a higher number — the failure")
        print("rate is what matters, and it is never zero.")
    else:
        print("There it is. That is why validate_refund_details.py exists.")


if __name__ == "__main__":
    main()
