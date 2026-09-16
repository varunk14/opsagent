# Reliability — what this agent gets right, what it gets wrong, and how that is known

Every number here is measured, not estimated, and every one of them can be reproduced from
this repository. Where a number is bad it is printed as it is: the point of measuring is to
know where the work goes next, and an evaluation that only ever says "fine" measures nothing.

**Measured:** 2026-09-16 · `llama3.1:8b`, run locally through Ollama, temperature 0, seed 0
· golden set `sha256 af4048c0885c`, 150 cases.

## How it is measured

150 golden cases, each a customer message with a label saying what should happen to it: the
intent, the order, the amount owed, and whether a person should decide. A case is **complete**
when the run comes to rest in the right place with the right money — refunded, waiting for
approval, or handed to a person — and incomplete otherwise. A right outcome reached for the
wrong reason is still a wrong reason: intent and extraction are scored separately.

Running 150 cases against a local model takes about an hour, and CI has no GPU. So the model's
replies are **recorded on the machine that has one and replayed in CI**, keyed by the hash of
the model name and the exact prompt. A replay cannot invent a reply: a prompt with no recording
fails rather than guesses. Because a recording can go stale, `python -m evals verify` re-runs
the whole set live and reports every reply that has changed. That is the check a replay cannot
make for itself.

Three layers, deliberately:

1. **Exact scoring** — outcome, money, intent, extraction. Deterministic, and the only layer
   that gates.
2. **A local judge** — scores whether a reply was grounded and appropriate. Published below,
   **not** gated, because it is not yet good enough to gate on (see "The judge").
3. **Live verification** — `verify`, above.

```bash
python -m evals record     # on a machine with the model; resumable
python -m evals gate       # replay + compare against the committed baseline (this runs in CI)
python -m evals verify     # live re-run, reports drift against the recordings
```

## Results

| Measure | Value |
|---|---|
| Cases | 150 |
| Task completion | 0.7200 (108 of 150) |
| Intent accuracy | 0.8600 |
| Extraction accuracy | 0.7800 |
| Escalation precision | 0.9118 |
| Escalation recall | 1.0000 |
| False-positive rate | 0.4615 |
| Safety violations | 0 |
| Unresolved runs | 0 |
| Model calls | 560 |
| Reference cost | $0.062746 |

Completion is 72 %, and nothing unsafe is paid. The largest remaining cause is the planner asking
for a lookup it has already been shown; the run then exhausts its step budget and goes to a
person. That is a safe failure — nothing is paid — but it is a failure, and it is what the next
change addresses.

Weakest categories, from `evals/scoreboard.md`: `confidence_pressure` 0.00, `inflated_amount`
0.00, `change_of_mind` 0.42, `damaged_item` 0.50. Strongest: `duplicate_not_confirmed`,
`multiple_orders`, `order_status`, `payment_question`, `unknown_order`, `wrong_owner`, `garbled`,
`general`, all 1.00.

### The nineteen unsafe payments, and how they were closed

The first measured baseline paid nineteen refunds automatically that a person should have
decided. They were **pinned by id in `evals/baseline.json`**, each with its own violations, so
the gate would fail if a new case became unsafe, a pinned case got worse, or the total rose. They
were published rather than hidden, because they were one defect, not nineteen:

> The guardrail judged a refund on the **amount** and the model's **self-reported confidence**
> only. It never checked whether the refund was owed. A confident model asking to refund a
> change-of-mind order under the limit was paid.

They are now zero. An automatic payment has to satisfy conditions that hold in the ledger rather
than in the model's opinion of itself: the message read as a duplicate charge, and the order
charged at least twice. Anything else is handed to a person — handed over, not queued for
approval, because there is no payment the agent can stand behind for someone to approve.

The amount is deliberately not one of those conditions. The ledger already refuses a refund
larger than the order was charged, the limit bounds what runs without a person, and refunds split
into parts are judged as the total they add up to — so requiring the amount to equal one charge
exactly would refuse legitimate partial refunds and add no safety.

What that change moved, measured the same way on the same 150 cases: task completion 0.5933 →
0.7200, safety violations 19 → 0, escalation recall 0.8468 → 1.0000, `wrong_escalation` 41 → 22,
and `duplicate_not_confirmed` from 0.20 to 1.00. Nothing got worse.

## The failure taxonomy

Every failed run is classified into exactly one of six fixed categories, by deterministic rules
over what the run left behind — its stop reason, its steps and what they returned, what was paid
or put to a person. No model judges this: a classifier you cannot trust cannot tell you where to
look. A run that did what it should has no category, and neither does one that died of an
outage or an expired lock — those are already named by `failure_class` and the dead-letter list.

| Category | Cases | What it means | What we would fix |
|---|---|---|---|
| `wrong_escalation` | 22 | A person rejected what the agent proposed, or it paid where a person should decide | Decide from policy conditions in code, not from confidence: a duplicate needs two ledger charges; change-of-mind and damaged items go to a person. |
| `loop` | 17 | The planner repeated a step, spent the whole step budget, or was deferred until a person had to take it | Give the planner a way to decide: when it repeats a lookup whose result is shown, ask once more with that result marked, then hand over. |
| `tool_misuse` | 3 | A tool that does not run here, arguments the ledger refused, a lookup of an unknown order, a refund for an order never looked up | Check arguments before proposing: a refund must be for an order this run looked up; the ledger already refuses more than was charged. |
| `hallucinated_field` | 0 | An order id or amount in neither the message, the policy it was shown, nor any tool result | Tighten extraction: an order id or amount must be quoted from the message or a lookup; refuse the proposal otherwise. |
| `context_overflow` | 0 | The customer's text was too long for the prompt and was cut | Chunk or summarise a long message before the prompt; today anything over the cap is cut. |
| `drift` | 0 | The same case, the same prompts, a different reply than last time | Pin the model version and the prompt hashes; record again and compare each case against the last accepted baseline. |

Drift is measured differently from the rest, and honestly: a replay is deterministic by
construction, so a replay can never show drift. Only a live run can, which is what `verify`
reports.

**Known limit of the classifier.** The hallucination rule trusts every number in the customer's
message as a possible source, so a message padded with numbers can hide an invented amount from
it. The category is a diagnostic; the exact scoring still marks the outcome wrong either way.

### The mix over time

`python -m evals accept` appends one line to `evals/history.jsonl` for every accepted baseline —
the date, the commit, the model and prompt versions, the completion, the violations, the failure
mix and the golden set's hash. The gate checks that the file ends with the baseline it is
comparing against, so a history edited on its own, or a baseline accepted without its line, fails.
The local screen charts it at `/failures`, beside the mix of real runs per week, so a category
that a change made worse is visible before anything is deployed.

| Accepted | Commit | Complete | hallucinated_field | tool_misuse | loop | context_overflow | wrong_escalation | drift |
|---|---|---|---|---|---|---|---|---|
| 2026-09-15 | `9303535` | 89 / 150 | 0 | 3 | 17 | 0 | 41 | 0 |
| 2026-09-16 | `cb9dd23` | 108 / 150 | 0 | 3 | 17 | 0 | 22 | 0 |

## The judge

| Measure | Value |
|---|---|
| Cases judged | 25 (0 unjudged) |
| Judged grounded | 0.0800 |
| Judged appropriate | 0.0000 |
| Agreement with layer 1 | 0.2000 |

A local model scoring another local model's work agrees with exact scoring on a fifth of cases
and calls nothing appropriate. It is published because the number is the finding: **this
judge is not fit to gate on**, and a gate built on it would have been a gate built on noise. It
stays in the harness, unwired from the gate, so the next model can be measured against the same
cases.

## Cost and latency

`BASELINE.md` holds the first measurement — one small model for everything, no routing, no cache,
no prompt trimming: 338.5 tokens per run, $0.0164 per 100 runs at the reference rate, p50 5,936 ms,
p95 11,642 ms over 12 runs. Those percentiles are nearest-rank over a sample far too small to
characterise a tail, and are reported only so the later comparison can compute the same statistic
the same way. The before-and-after against today's stack is measured in a later milestone and will
be published here with it; no number is quoted for it until it has been measured.

Every run's cost is charged from its own token counts at a fixed reference rate and stored on the
run, and every step writes a span, so cost is attributable per step rather than estimated in
aggregate. Embedding tokens are traced but not yet counted in a run's cost; that gap is named on
the `/runs` page rather than rounded away.

## What this does not measure

- **Real customers.** The 150 cases are written, not sampled from production traffic; there is
  none yet. The distribution of real messages will differ, and the numbers will move when it does.
- **One machine.** Every latency here is this laptop's. They are comparable to each other and to
  nothing else.
- **The adversarial set is small.** Prompt injection and fake-policy cases are present but few;
  a determined attacker is not represented by a handful of labelled cases.
- **The judge is weak**, as above.
