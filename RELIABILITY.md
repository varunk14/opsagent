# Reliability — what this agent gets right, what it gets wrong, and how that is known

Every number here is measured, not estimated, and every one of them can be reproduced from
this repository. Where a number is bad it is printed as it is: the point of measuring is to
know where the work goes next, and an evaluation that only ever says "fine" measures nothing.

**Measured:** 2026-09-23 · `llama3.1:8b`, run locally through Ollama, temperature 0, seed 0
· golden set `sha256 222ed8697b18`, 159 cases.

## How it is measured

159 golden cases, each a customer message with a label saying what should happen to it: the
intent, the order, the amount owed, and whether a person should decide. A case is **complete**
when the run comes to rest in the right place with the right money — refunded, waiting for
approval, or handed to a person — and incomplete otherwise. A right outcome reached for the
wrong reason is still a wrong reason: intent and extraction are scored separately.

Running 159 cases against a local model takes about an hour, and CI has no GPU. So the model's
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
| Cases | 159 |
| Task completion | 0.9748 (155 of 159) |
| Intent accuracy | 0.8679 |
| Extraction accuracy | 0.7673 |
| Escalation precision | 0.9706 |
| Escalation recall | 1.0000 |
| False-positive rate | 0.1481 |
| Safety violations | 0 |
| Unresolved runs | 0 |
| Model calls | 671 |
| Reference cost | $0.068331 |

Completion is 97 %, and nothing unsafe is paid.

**Every one of the four remaining failures is the same category.** `inflated_amount` sits at 0.00;
all eighteen other categories are at 1.00. Those four are customers asking for more than the order
was ever charged, and the agent proposes the inflated figure rather than the real one. The ledger
refuses to pay it and the case goes to a person, so no money is lost — but the agent is wrong before
the ledger catches it, and that is a specific next thing to fix rather than anything diffuse about
accuracy.

Intent and extraction accuracy have not moved, and that is worth saying plainly: completion rose
because cases already destined for a person now reach them the right way, not because the model
reads messages any better than it did.

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

The amount is a condition too: it must equal a charge the order was duplicated at. A duplicate is
owed back at what it was charged, and nothing the run established says what a smaller number
should be, so a person chooses that instead of the model. Measured, this cost no completion —
every golden case that expects a refund expects one of the order's charges — and it removed
`tool_misuse` entirely, because a refund the ledger would have refused is now refused before it
ever reaches the ledger. Two consequences worth stating: an automatic refund can no longer exceed
what the order took, so the ledger's cap is only reachable by paying the same duplicate back more
times than it was charged; and a prompt telling the model to double every refund can no longer
get anything paid, only fail to complete.

What that change moved, measured the same way on the same 150 cases: task completion 0.5933 →
0.7200, safety violations 19 → 0, escalation recall 0.8468 → 1.0000, `wrong_escalation` 41 → 22,
and `duplicate_not_confirmed` from 0.20 to 1.00. Nothing got worse.

### Asking again before waking anyone

The next largest cause of incomplete runs was the planner asking for a lookup whose result was
already in front of it. The run then spent its step budget and went to a person — a safe failure,
since nothing is paid, but a failure. That reads as not having noticed the answer rather than as
being stuck, so the run is now asked once more with that result marked, and only a second repeat
hands the case over. Every earlier step is already on record, so asking again costs one model
call, not four, and the run is charged for both.

Measured: task completion 0.7200 → 0.7467, `loop` 17 → 13, `confidence_pressure` 0.00 → 1.00,
`duplicate_over_limit` 0.40 → 0.50, still nothing unsafe. It is a real gain and a partial one:
four of the seventeen looping runs recovered, and the other thirteen asked for the same thing
twice. The price is honest too — model calls rose from 560 to 644 and the reference cost from
$0.0627 to $0.0771 for the set, because runs that used to stop now carry on.

### The guard that only ran on refunds about to pay themselves

The largest single gain, thirty-four cases, came from a change that was not planned and was found by
the gate refusing a release.

Shortening the planning prompt made the model more decisive. Every looping run disappeared and
duplicate-charge cases went to 1.00 — but ten cases got *worse* in a very specific way: the agent now
proposed a refund where it used to escalate, and each of those proposals was queued as a **human
approval** instead of the case being **handed over**. The per-category gate caught it even though the
headline completion had gone up.

Both outcomes end with a person, so nothing unsafe was paid either way, and it would have been easy
to wave through. They are not the same thing though. Handing over gives someone the case to decide;
an approval gives them a filled-in refund and one button — and people approve what is put in front
of them far more readily than they would have proposed it. For a customer citing a policy that does
not exist, that difference is the whole control.

The cause was that the conditions guard ran **only** where a refund would otherwise have paid itself.
A refund already bound for a person skipped it, on the reasoning that one going to a person is a
person's to judge. Asking one of those conditions — *was this read as a duplicate charge at all?* —
of every refund recovered the ten regressions and eighteen more that had been arriving as approvals
since long before this work. It refuses only: it can take a payment away, never grant one, or
satisfying it would become a route around the confidence floor rather than a check on top of it.

Measured: task completion 0.7467 → 0.9733, `wrong_escalation` 25 → 4, `loop` 13 → 0, false-positive
rate 0.4615 → 0.1538, safety violations 0 → 0, and the reference cost *down* from $0.077111 to
$0.066294 because fewer runs go round again.

A wider version of the same fix — applying every condition to every refund — was written first and
thrown away: it broke twenty-four tests across eight files, including the acceptance test that a
refund over the limit pauses for approval. It was quietly redefining the over-limit approval path,
which is a different decision from the one being made here.

### Offering only the tools that apply, and what it did not move

The planner used to be shown every tool on every run. Given `get_order` with no order named, the
real model reached for it anyway — copying the example id out of the description — and a message that
mentioned no order looped on looking one up until it was handed over. The plan prompt now withholds
`get_order` and `issue_refund` when no order is in play, the same way it already withholds
`search_policy` once the policy is in the prompt.

Measured on the golden set, this moved the completion number **barely** — 0.9733 → 0.9740 — and it is
worth being exact about why, because the honest answer is that the golden set could not show the
problem this fixes. Its orderless cases already reached a person, and `loop` was already 0; there was
no golden run stuck on a phantom lookup for this to rescue. The looping was seen on *real* messages,
which named no order and are not in the hand-written set. So the fix ships with the evidence for it:
the four demo messages that exercised the loop are now measured cases (`n-151`–`n-154`), and the set
grew from 150 to 154. All four land where their labels say — a duplicate charge refunded, a
change-of-mind and an order-status question and a channel enquiry each handed to a person — which is
the whole of the completion change.

One of the four costs a little accuracy, and it should be shown rather than smoothed over. The
channel enquiry is a duplicate-charge complaint from a sender who owns no account, so its label says
there is no order for the agent to act on; the model, reading a message that names one, extracts it
anyway. That is scored as an extraction miss — extraction accuracy edges from 0.7800 to 0.7727 — and
it is the right thing to count: the case still hands over, but the model did reach for an order that
was never the sender's to claim.

Cost did not fall; it rose, from 638 model calls to 675 and $0.066 to $0.069. That is the four added
cases running their full multi-step flows, not the change, which is close to call-neutral on the
existing set. What the change buys is not on this scoreboard: a message that names no order now goes
to a person in one step instead of after a loop, and a tool that moves money is no longer offered
when there is no order to move it against — a narrowing of what the model can even propose, on top of
the code guards that would refuse it regardless.

### Refusing the loop before it starts, not after

Withholding the tools from the prompt is a nudge, not a fence: a live model can still ask for
`get_order` on an order the customer never wrote — an id it invented, or copied from somewhere it
should not have. Run, that lookup is refused by the ownership check, proposed again, and only then
handed over: the "repeated an earlier step" loop, safe but wasteful, and it was seen on real
order-status messages the moment the agent read a live channel. The driver now asks the same question
the executor asks — *is this order number written in the message?* — before it runs the tool, and
hands the case over on the first ask rather than after the round trip. The check is the executor's own,
shared so the two can never drift.

Measured, it moved no completion (0.9740) and nothing unsafe (0), and it took the reference cost from
$0.066294 to $0.066311 — flat, because the golden set rarely triggers it; the handful of cases where
the recorded model reached for an unwritten order now rest a step sooner, 675 model calls down to 650.
Its worth is not on this scoreboard either: on a live channel it turns a visible loop into a clean
hand-over, and a lookup of an order the customer never named can no longer run at all.

## The failure taxonomy

Every failed run is classified into exactly one of six fixed categories, by deterministic rules
over what the run left behind — its stop reason, its steps and what they returned, what was paid
or put to a person. No model judges this: a classifier you cannot trust cannot tell you where to
look. A run that did what it should has no category, and neither does one that died of an
outage or an expired lock — those are already named by `failure_class` and the dead-letter list.

| Category | Cases | What it means | What we would fix |
|---|---|---|---|
| `wrong_escalation` | 4 | A person rejected what the agent proposed, or it paid where a person should decide | Refuse an amount larger than the order was charged at extraction, not at the ledger: all four are a customer asking for more than they paid. |
| `loop` | 0 | The planner repeated a step, spent the whole step budget, or was deferred until a person had to take it | Closed. A repeated lookup is now asked once more with the result marked, and a shorter planning prompt stopped the rest. |
| `tool_misuse` | 0 | A tool that does not run here, arguments the ledger refused, a lookup of an unknown order, a refund for an order never looked up | Check arguments before proposing: a refund must be for an order this run looked up; the ledger already refuses more than was charged. |
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
| 2026-09-16 | `34ac5fe` | 112 / 150 | 0 | 3 | 13 | 0 | 22 | 0 |
| 2026-09-16 | `4551a40` | 112 / 150 | 0 | 0 | 13 | 0 | 25 | 0 |
| 2026-09-16 | `3ed692f` | 146 / 150 | 0 | 0 | 0 | 0 | 4 | 0 |
| 2026-09-17 | `ff52501` | 150 / 154 | 0 | 0 | 0 | 0 | 4 | 0 |

## The judge

| Measure | Value |
|---|---|
| Cases judged | 25 (0 unjudged) |
| Judged grounded | 0.0800 |
| Judged appropriate | 0.0000 |
| Agreement with layer 1 | 0.1600 |

A local model scoring another local model's work agrees with exact scoring on a sixth of cases
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

- **Real customers.** The agent now reads real channels — an IMAP mailbox, a Telegram bot, and a
  voice channel — but the 159 cases are still written by hand, not sampled from production traffic,
  of which there is none yet. The distribution of real messages will differ, and the numbers will
  move when it does.
- **The voice cases carry a transcript, not a real audio clip.** Five voice-channel cases
  (`n-124`–`n-128`) sit in the golden set with bodies written to look the way Sarvam's speech-to-
  text tends to write them — lowercase, thin punctuation, filler words. They exercise the intake,
  the guardrail, and the reply outbox against the voice channel; every one is handed to a person
  because a voicemail sender cannot be tied to an account. What Sarvam gets wrong on real audio,
  and how a misheard order number reads through the same pipeline, is a separate thing to measure.
  It will be once there are enough real recordings to sit alongside these.
- **One machine.** Every latency here is this laptop's. They are comparable to each other and to
  nothing else.
- **The adversarial set is small.** Prompt injection and fake-policy cases are present but few;
  a determined attacker is not represented by a handful of labelled cases.
- **The judge is weak**, as above.
