"""
Validating refund details: stop the model from lying to your code.

The unvalidated version scored 0/5 usable. Three different problems were tangled together.
This file separates them and fixes each one with a different tool.

  problem                          fix
  -------------------------------  --------------------------------------------
  prose, ``` fences, bullet lists  constrained decoding  (Ollama format="json")
  valid JSON but wrong shape       schema validation     (Pydantic)
  model cannot answer at all       an explicit REFUSAL path, not an exception

That last row is the one people miss. Unvalidated, attempt 4 claimed there was no
email. That is not a parsing bug and no amount of regex fixes it. The model needs
a legitimate way to say "I could not do this" that your code can act on.

Run:  .venv/bin/python experiments/validate_refund_details.py 5
"""

import json
import sys
import urllib.error
import urllib.request
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

PRIYA_EMAIL = (
    "Hi, I think I was charged twice for order #4821 last Tuesday. "
    "Can you refund one of them? Thanks, Priya"
)


# ---------------------------------------------------------------------------
# THE CONTRACT
#
# This class is the boundary between "text a model produced" and "data my
# program is allowed to act on". Nothing crosses without passing through here.
# ---------------------------------------------------------------------------
class Extraction(BaseModel):
    # extra="forbid" means: if the model invents a field we did not ask for,
    # that is an ERROR, not something we silently ignore. Loud beats quiet.
    model_config = ConfigDict(extra="forbid")

    can_extract: bool
    order_id: str | None = None
    reason: Literal["double_charge", "damaged", "not_received", "other"] | None = None

    # Money is an INTEGER, in paise. Never a float.
    # 0.1 + 0.2 != 0.3 in floating point. Rupees as floats lose money.
    amount_paise: int | None = None

    # ge/le are real constraints. A model returning confidence 5.0 fails here
    # instead of sailing into a threshold check that silently passes.
    confidence: float = Field(ge=0.0, le=1.0)

    note: str | None = None  # why it could not extract, when can_extract is False

    @model_validator(mode="after")
    def check_consistency(self):
        """A claim of success must come with the goods."""
        if self.can_extract:
            missing = [f for f in ("order_id", "reason") if getattr(self, f) is None]
            if missing:
                raise ValueError(
                    f"can_extract=True but these are missing: {', '.join(missing)}"
                )
        return self


PROMPT = f"""Extract the refund request details from this customer email.

Email:
{PRIYA_EMAIL}

Reply with a JSON object containing exactly these fields:
- can_extract: true if you can read the email and extract details, false if you cannot
- order_id: string, or null
- reason: one of "double_charge", "damaged", "not_received", "other", or null
- amount_paise: integer amount in paise, or null if not stated
- confidence: float between 0.0 and 1.0
- note: if can_extract is false, a short reason why. Otherwise null.
"""


def ask_model(prompt: str) -> str:  # pragma: no cover
    """The same POST as before, with ONE addition: format="json"."""
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            # THE FIX for failure modes 2, 3 and 5.
            # This is not a polite request. Ollama constrains token sampling to a
            # JSON grammar, so the model is structurally unable to emit prose or
            # ``` fences. It is enforcement, not persuasion.
            "format": "json",
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"]


# Three outcomes, not two. This is the important idea in this file.
OK, REFUSED, INVALID = "ok", "refused", "invalid"


def attempt(n: int) -> str:  # pragma: no cover
    print(f"\n{'=' * 70}\nATTEMPT {n}\n{'=' * 70}")

    try:
        raw = ask_model(PROMPT)
    except urllib.error.URLError as e:
        print(f"  Could not reach Ollama at {OLLAMA_URL}: {e}")
        print("  Is it running?   ollama serve")
        sys.exit(1)

    print("\n--- RAW ---")
    print(repr(raw))

    # Gate 1: is it even JSON? With format="json" this should now always pass.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n  STILL not JSON: {e}")
        return INVALID

    # Gate 2: is it the RIGHT SHAPE? Valid JSON is not the same as valid data.
    try:
        extraction = Extraction.model_validate(data)
    except ValidationError as e:
        print("\n  Valid JSON, but it failed the contract:")
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            print(f"    - {loc}: {err['msg']}")
        return INVALID

    # Gate 3: did the model succeed, or legitimately refuse?
    if not extraction.can_extract:
        print("\n  REFUSED (this is a valid outcome, not a crash)")
        print(f"    note: {extraction.note}")
        return REFUSED

    print("\n  USABLE — typed object your code can act on:")
    print(f"    order_id     = {extraction.order_id!r}")
    print(f"    reason       = {extraction.reason!r}")
    print(
        f"    amount_paise = {extraction.amount_paise!r}"
        f"  ({type(extraction.amount_paise).__name__})"
    )
    print(f"    confidence   = {extraction.confidence}")
    return OK


def main() -> None:  # pragma: no cover
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    results = [attempt(i) for i in range(1, runs + 1)]

    print(f"\n{'=' * 70}\nSCOREBOARD over {runs} runs\n{'=' * 70}")
    for label, name in (
        (OK, "usable"),
        (REFUSED, "refused (handled)"),
        (INVALID, "invalid (bug)"),
    ):
        print(f"  {name:<20} {results.count(label)}")

    print(
        "\nCompare with extract_refund_details.py, which scored 0/5.\n"
        "Note that 'refused' is NOT in the failure column. An agent that can say\n"
        "'I could not do this' is safer than one that always produces an answer."
    )


if __name__ == "__main__":
    main()
