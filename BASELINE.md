# Baseline — before any optimisation

**Recorded:** 2026-09-13
**Model:** `llama3.2`, run locally through Ollama
**Sample:** 4 runs over `fixtures/inbox.jsonl`
**Settings:** one model for everything. No routing, no caching, no prompt
trimming, no structured output. The naive version, deliberately.

## The numbers week 9 will be compared against

| | |
|---|---|
| Tokens per run | 342 |
| Prompt tokens, total | 344 |
| Completion tokens, total | 1,027 |
| Cost per 100 runs | $0.0167 |
| Latency p50 | 7,185 ms |
| Latency p95 | 9,213 ms |

## About that dollar figure

Inference is local, so nothing was actually spent. The cost above is token counts
converted at a fixed rate defined in `app/baseline.py`:

> reference-2026-09 (arbitrary, fixed, used on both sides of every comparison)
> $0.15/M input, $0.60/M output

The rate is arbitrary and is not a quote from anyone. It is the same rate on both
sides of every comparison, so it cancels; what survives is the ratio, which is
the thing being claimed. The token counts are kept, so substituting a real rate
later re-derives both ends.

## Reproducing this

    ollama serve &
    .venv/bin/python -m app.baseline

Latency depends on the machine. The token counts do not.
