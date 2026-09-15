# Baseline — before any optimisation

**Recorded:** 2026-09-13
**Model:** `llama3.2`, run locally through Ollama
**Sample:** 12 runs — 3 passes over the 4 messages in `fixtures/inbox.jsonl`
**Settings:** one model for everything. No routing, no caching, no prompt
trimming, no structured output. The naive version, deliberately.

## The numbers later optimisation will be compared against

| | |
|---|---|
| Tokens per run | 338.5 |
| Prompt tokens, total | 1,032 |
| Completion tokens, total | 3,030 |
| Cost per 100 runs | $0.0164 |
| Latency p50 | 5,936 ms |
| Latency p95 | 11,642 ms |

Read the p95 with care. Percentiles here are nearest-rank: sort the
12 runs from fastest to slowest and take the 12th. With a sample
this small that is the slowest run observed, not an estimate of a tail. It is
reported because the later comparison needs the same statistic computed the same way, not
because 12 runs can characterise a distribution.

A warm-up run is made and discarded before measuring, so a cold model load does
not land in the sample.

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

Sampling is pinned (`temperature 0`, `seed 0`), so the token counts are a
property of the prompt and should reproduce. Latency depends on the machine and
will not.

Every run behind the table above is in `baseline-measurements.json`, so a real
price list can be substituted later and both ends of the before-and-after comparison
re-derived without measuring anything again.
