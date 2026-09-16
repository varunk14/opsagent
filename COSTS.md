# What a run costs

Measured before and after, on the same machine, by the same method. The live burn-down is on the
screen at `/costs`.

## The headline

| | before | after | |
|---|---|---|---|
| Tokens per run | 3,036.8 | **2,564.2** | −15.6 % |
| Cost per 100 runs | $0.0541 | **$0.0473** | −12.6 % |
| Task completion | 112 / 150 | **146 / 150** | +34 cases |
| Safety violations | 0 | **0** | unchanged |

A run got cheaper and more of them finished. Neither number came from the change that was supposed
to deliver it.

## Two measurements, not one

Cost is measured twice, by methods with different weaknesses, because a single figure from a laptop
is not evidence.

**Live, 12 runs.** Four fixture messages, three passes, the warm-up discarded, each run worked to
rest through the real graph on `llama3.1:8b`. This is the method the project's first baseline used,
so the two are comparable. It gives the table above.

**Replayed, 150 cases.** The whole golden set scored from recorded model replies. Replay is
deterministic and calls no model, so there is no sampling error and no thermal noise at all:

| | before | after |
|---|---|---|
| Reference cost, 150 cases | $0.077111 | **$0.066294** (−14.0 %) |
| Model calls | 644 | 639 |
| Completed | 112 | 146 |
| Safety violations | 0 | 0 |

−12.6 % and −14.0 %, from twelve live runs and from a hundred and fifty replayed ones. They agree
within a point and a half, which is about as much confidence as this hardware can supply.

## Where the saving came from

**The tool list stopped being JSON schema.** The planning prompt is 82 % of everything this agent
reads, and half of that was the tool block — mostly `maxLength`, `pattern`, `minimum` and `maximum`
punctuation. `app/contracts.py` enforces every one of those in code whatever the prompt says, so
sending them to the model guaranteed nothing that was not already guaranteed. Rendered as prose
instead: **37 % off the tool block, 14 % off all prompt text.** Nothing a model cannot work out for
itself was dropped — what each tool does, when to reach for it, and the facts a reader would
otherwise guess, such as money being counted in paise.

Two supporting changes made the number measurable rather than smaller. **Per-model rates**, because
pricing every token identically would have shown a saving of exactly zero for any routing change.
And **run budgets** — a ceiling on tokens, money and model time per run — because `MAX_STEPS` bounded
a count of tools, which says nothing about what a run may spend.

## Where the completions came from

Not from the trimming, which was worth six of the thirty-four.

The re-record showed the trimming had also made the model more decisive: every loop disappeared, and
duplicate-charge cases went to 100 %. But ten cases regressed, all in the same direction — the agent
now *proposed* a refund where it used to escalate, and those proposals queued for a human approval
instead of handing the case over.

The cause was a guard that only ran on refunds about to pay themselves. A refund already bound for a
person skipped the conditions entirely, by an explicit decision: one already going to a person is a
person's to judge. That reasoning does not hold, because the two outcomes ask different things of
the person. Handing over gives them the case; an approval gives them a filled-in refund and one
button, and people approve what is put in front of them far more readily than they would have
proposed it.

Asking one condition — *was this even read as a duplicate charge?* — of every refund rather than only
of the automatic ones recovered those ten, and eighteen more that had been arriving as approvals all
along, before any of this work. Refusal only: it can take a payment away, never give one back, or satisfying it
would become a way around the confidence floor instead of a check standing on top of it.

## What was measured away

Two of the four things planned for this work were never built, because measuring them first showed
what they were worth.

**A routing ladder** sending the cheap steps to a 3B model could only touch the classify and extract
prompts — **18.1 %** of prompt text. At a third the rate, its ceiling was about **9.5 %**, before any
loss from a smaller model failing schemas.

**A reply cache** would have served **0.15 %** of prompts: one repeat in 669, counted over a full
replay. That is structural, not a tuning problem. Every prompt embeds the customer's message, so two
prompts collide only when two customers write the same words in the same situation. No TTL, store or
key changes it. A cache would pay for *re-runs* — an outage requeue, a redelivered message — and
there is no such traffic to protect yet.

Both measurements cost an afternoon. Building either would have cost more and delivered a headline
number that was mostly the prompt trimming anyway.

## Latency: not attributable on this machine

| | before | after |
|---|---|---|
| p50 | 19,749 ms | 30,636 ms |
| p95 | 31,909 ms | 43,625 ms |

Read as a result, that says the agent got slower. It is not a result. Here is the after measurement
in the order it ran, same four messages cycling:

```
16,281   22,167   17,576   24,579     <- runs 1-4
32,435   43,625   30,636   35,240     <- runs 5-8
30,095   37,738   30,923   31,806     <- runs 9-12
```

The machine throttles about a third of the way in and never recovers within the sample, so **8 of 12
runs** sit in the slow regime and the percentiles mostly report *when* that happened. The before
measurement did the same thing — its first four runs took 15.0, 18.4, 15.8 and 18.9 s, its last four
27.1, 30.9, 26.5 and 29.7 s.

Meanwhile the token counts per message are byte-identical between passes and between the two
measurements' shared cases. The agent does measurably *less* work per run and takes longer in wall
clock, on a laptop that is thermally limited before it is compute limited.

Twelve runs cannot separate the code from the hardware. The honest statement is that this project has
not measured a latency change, and that a number claiming one from this sample would be a number
about a laptop.

## What these figures leave out

**Embedding tokens are not counted.** Every policy search embeds its question with
`nomic-embed-text`, and those tokens are in neither a run's cost nor the table above. It is one short
call against several long ones — well under 1 % of a run — but the figures are low by that much.
Counting it properly means changing the recordings format and re-recording the whole golden set, and
that was judged not worth an hour of a laptop's time for a sub-1 % correction. The `/costs` page says
the same thing where the numbers are shown.

Costs are in **reference dollars**: these models run locally on one Mac and cost electricity, not
API fees. The rates price each model as a comparable hosted model would, so the figures mean
something to a reader and move correctly when the work changes.

## Conditions

Both measurements: the same MacBook, `llama3.1:8b` through a local Ollama, self-hosted tracing
stopped, nothing else pulling models, four fixture messages × three passes with a discarded warm-up,
nearest-rank percentiles.

Charging state is recorded because it matters on this hardware: the before ran **charging from 41 %**,
the after **charging from 37 %**.

Reproduce the deterministic half with `python -m evals accept` — it needs no model and no network.
