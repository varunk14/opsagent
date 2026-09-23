# Current stack baseline

**Recorded:** 2026-09-23
**Model:** `llama3.1:8b` for planning; `nomic-embed-text` for policy retrieval,
run locally through Ollama. The whole agent: classify, extract, retrieve, plan, tools,
guardrail, reply -- as it runs in production.

**Sample:** 12 runs -- 3 passes over the 4
messages in `fixtures/inbox.jsonl`, the same inbox the naive baseline used.

## The numbers, next to the naive baseline

| | Naive (BASELINE.md) | Current stack |
|---|---|---|
| Tokens per run | 338.5 | 2,473.00 |
| Cost per 100 runs | $0.0164 | $0.0460 |
| Latency p50 | 5,936 ms | 24,515 ms |
| Latency p95 | 11,642 ms | 43,525 ms |

Percentiles are nearest-rank over 12 runs, exactly the way the naive baseline is
computed. That is not enough to characterise a tail; it is enough that both sides
compare like for like.

The reference rate is the same on both sides (see `BASELINE.md`), so it cancels. What
survives the comparison is the ratio. The raw per-run token counts are in
`current-measurements.json` alongside the naive `baseline-measurements.json`, so a
different rate can rederive both without measuring again.
