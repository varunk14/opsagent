"""
What one run costs, before anything has been optimised.

Week 9 is supposed to end with a sentence like "cost per 100 runs fell from A to
B, with task completion unchanged". That sentence is worth nothing unless A was
recorded before any of the work that produced B, using the same arithmetic. It
cannot be reconstructed later, so it is taken now, at the naive settings: one
model, no routing, no caching, no prompt trimming.

On the dollar figure, plainly: inference here runs on a local model, so the cash
cost is nil. Reporting $0.00 and then "improving" it would be a lie. What is
actually measured is tokens and wall-clock -- both real, both attributable --
and those are converted with a rate written down below.

That rate is not a quote from any vendor and is not claimed to be one. It does
not need to be. Both the before and the after use it, so it cancels, and what
survives the comparison is the ratio, which was the claim in the first place.
Anyone who wants real dollars can substitute a real rate and re-derive both ends
from the token counts, which are kept.

Run:  .venv/bin/python -m app.baseline
"""

import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from app.adapters.fixture import read_messages

OLLAMA = "http://localhost:11434/api/generate"
NAIVE_MODEL = "llama3.2"

# The prompt as it would be written by someone who has not yet been burned. No
# schema, no constraint, no examples. Week 2 replaces it; this is what it is
# being replaced FROM.
NAIVE_PROMPT = """You are a customer support agent. Read this email and work out
what the customer wants. Say what you would do about it.

From: {sender}
Subject: {subject}

{body}
"""


@dataclass(frozen=True)
class ReferenceRate:
    """
    USD per million tokens. An accounting convention, not a price list.

    Named so that a figure can never be quoted without saying which rate made it.
    """

    name: str
    input_per_million: Decimal
    output_per_million: Decimal


REFERENCE_RATE = ReferenceRate(
    name="reference-2026-09 (arbitrary, fixed, used on both sides of every comparison)",
    input_per_million=Decimal("0.15"),
    output_per_million=Decimal("0.60"),
)


@dataclass(frozen=True)
class StepMeasurement:
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


def percentile(values: Sequence[int], q: int) -> int:
    """
    Nearest-rank, which for the handful of samples taken here is the honest
    choice: every number reported is one that was actually observed, rather than
    an interpolation between two that were.
    """
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class Baseline:
    runs: int
    model: str
    rate: ReferenceRate
    prompt_tokens: int
    completion_tokens: int
    latencies_ms: tuple[int, ...]

    @property
    def cost_usd(self) -> Decimal:
        million = Decimal(1_000_000)
        return (
            Decimal(self.prompt_tokens) / million * self.rate.input_per_million
            + Decimal(self.completion_tokens) / million * self.rate.output_per_million
        )

    @property
    def cost_per_100_runs(self) -> Decimal:
        return self.cost_usd / self.runs * 100

    @property
    def tokens_per_run(self) -> int:
        return (self.prompt_tokens + self.completion_tokens) // self.runs

    @property
    def p50_ms(self) -> int:
        return percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> int:
        return percentile(self.latencies_ms, 95)


def summarise(measurements: Sequence[StepMeasurement], rate: ReferenceRate) -> Baseline:
    """
    Add up what was observed.

    Refuses an empty sequence. A baseline of nothing reports zero cost and zero
    latency, and week 9 would compare against it without complaint.
    """
    if not measurements:
        raise ValueError("cannot summarise no measurements")

    return Baseline(
        runs=len(measurements),
        model=measurements[0].model,
        rate=rate,
        prompt_tokens=sum(m.prompt_tokens for m in measurements),
        completion_tokens=sum(m.completion_tokens for m in measurements),
        latencies_ms=tuple(m.latency_ms for m in measurements),
    )


def measure_one(prompt: str, model: str = NAIVE_MODEL) -> StepMeasurement:  # pragma: no cover
    """One call to the local model, timed. Network, so not exercised by the suite."""
    request = urllib.request.Request(
        OLLAMA,
        data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read())
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return StepMeasurement(
        model=model,
        prompt_tokens=payload.get("prompt_eval_count", 0),
        completion_tokens=payload.get("eval_count", 0),
        latency_ms=elapsed_ms,
    )


def report(baseline: Baseline, inbox: Path) -> str:
    """The markdown committed as BASELINE.md."""
    return f"""# Baseline — before any optimisation

**Recorded:** {datetime.now(timezone.utc).date().isoformat()}
**Model:** `{baseline.model}`, run locally through Ollama
**Sample:** {baseline.runs} runs over `{inbox}`
**Settings:** one model for everything. No routing, no caching, no prompt
trimming, no structured output. The naive version, deliberately.

## The numbers week 9 will be compared against

| | |
|---|---|
| Tokens per run | {baseline.tokens_per_run:,} |
| Prompt tokens, total | {baseline.prompt_tokens:,} |
| Completion tokens, total | {baseline.completion_tokens:,} |
| Cost per 100 runs | ${baseline.cost_per_100_runs:.4f} |
| Latency p50 | {baseline.p50_ms:,} ms |
| Latency p95 | {baseline.p95_ms:,} ms |

## About that dollar figure

Inference is local, so nothing was actually spent. The cost above is token counts
converted at a fixed rate defined in `app/baseline.py`:

> {baseline.rate.name}
> ${baseline.rate.input_per_million}/M input, ${baseline.rate.output_per_million}/M output

The rate is arbitrary and is not a quote from anyone. It is the same rate on both
sides of every comparison, so it cancels; what survives is the ratio, which is
the thing being claimed. The token counts are kept, so substituting a real rate
later re-derives both ends.

## Reproducing this

    ollama serve &
    .venv/bin/python -m app.baseline

Latency depends on the machine. The token counts do not.
"""


def main(argv: list[str]) -> int:  # pragma: no cover - the interactive driver
    inbox = Path(argv[1] if len(argv) > 1 else "fixtures/inbox.jsonl")
    messages = list(read_messages(inbox))

    print(f"Measuring {len(messages)} naive runs on {NAIVE_MODEL}. This is slow by design.\n")
    measurements = []
    for number, message in enumerate(messages, start=1):
        prompt = NAIVE_PROMPT.format(
            sender=message.sender, subject=message.subject or "", body=message.body
        )
        try:
            step = measure_one(prompt)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  cannot reach Ollama at {OLLAMA}: {exc}")
            print("  start it with `ollama serve` and try again.")
            return 1

        measurements.append(step)
        print(
            f"  {number}/{len(messages)}  {step.prompt_tokens:>5} in  "
            f"{step.completion_tokens:>4} out  {step.latency_ms:>6} ms"
        )

    baseline = summarise(measurements, REFERENCE_RATE)
    Path("BASELINE.md").write_text(report(baseline, inbox), encoding="utf-8")

    print(f"\n  tokens/run       {baseline.tokens_per_run:,}")
    print(f"  cost/100 runs    ${baseline.cost_per_100_runs:.4f}")
    print(f"  latency p50/p95  {baseline.p50_ms:,} / {baseline.p95_ms:,} ms")
    print("\n  written to BASELINE.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
