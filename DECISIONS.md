# Decisions

Why this is built the way it is, including the parts that were built and then thrown away, and the
things that were planned, measured, and never built at all.

---

## The model proposes. Code decides.

`app/tools.py` holds names, descriptions and JSON schemas and **nothing executable**. A test walks
every tool and fails if one ever gains a callable attribute, so proposing a refund cannot issue one
even by mistake. What runs is decided in `app/run_agent.py` against three frozen sets: tools that
run immediately, tools the guardrail judges first, and tools this worker does not run at all. A new
tool added to the registry cannot execute by default; it has to be put in one of those sets on
purpose.

This is the single most important line in the system. Everything else is a consequence of it.

## Money is integers, in paise

`amount_paise: int`, `strict=True`, never a float and never rupees. Rupees and paise differ by a
factor of a hundred, and a model reading an email has no way to guess which was meant — so the unit
is stated in the tool description, validated at the boundary, and a `bool` is explicitly rejected
because `True` is a subclass of `int` and is not an amount of money.

A floating-point refund is a rounding error someone eventually pays for.

## Durability is hand-rolled, not Temporal

A run is a row in Postgres. Workers claim with `FOR UPDATE SKIP LOCKED`, commit after every step,
and reclaim a run whose worker has gone quiet for five minutes.

Temporal would do all of this properly. Its cloud tier costs money this project does not have, and
the self-hosted server wants the memory the local models need. But the honest reason is that the
failure modes **are** the interesting part, and hiding them behind a framework would have hidden the
work. Each one is pinned by a test: a SIGKILL mid-run resumed by another worker, two workers racing
one idempotency key, a worker that lost its claim executing nothing.

The cost of this decision is real: roughly a thousand lines that a framework would have given away,
and every one of them is ours to maintain.

## Postgres is the queue, and the vector store

**The queue**: `SKIP LOCKED` gives a multi-worker queue with no new moving part, and — more
importantly — the claim and the work land in the same database, so "claimed but not recorded" is not
a state that can exist. A Redis queue beside a Postgres database would have made it one.

**The vectors**: pgvector, not Pinecone or Qdrant. Fifteen policy chunks do not need a dedicated
vector database, and a hosted one would mean the policy text leaves the machine. The embedding and
the row it belongs to are written in the same transaction.

Redis is in the compose file and deliberately unused — see *the cache that was not built*, below.

## The transactions are short, and separate on purpose

| Step | What it holds |
|---|---|
| claim | one `UPDATE`, committed at once |
| graph | **no transaction open** |
| act | one transaction: re-confirm the claim, execute, append the step, add the cost |

The graph step is where model calls happen and they take tens of seconds. Holding a row lock across
them would block every other worker for the duration. So the lock is taken, released, and taken
again — which means a worker must re-confirm it still holds the claim before acting, and one that
lost it does nothing.

## Confidence is not evidence

The first guardrail judged a refund on its amount and the model's **self-reported confidence**. That
is circular: a model reading a customer's email is exactly the thing an email can talk round, and a
confident model could have any small refund paid by asserting it was owed. The first measured
baseline paid nineteen refunds automatically that a person should have decided.

The guard now reads the **ledger**: for an automatic refund, the message must have been read as a
duplicate charge, and the order must show exactly one amount charged twice. Being charged more than
once is not being charged twice — an order billed for the item and then for shipping has two charges
and no duplicate.

Confidence still has a job. It decides whether a person is asked. It no longer decides whether
something is true.

## Handing over and queueing an approval are different things

Both end with a person, which made them look interchangeable for a long time. They are not.

Handing a case over gives someone the case. Queueing an approval gives them a **filled-in refund and
one button** — and people approve what is put in front of them far more readily than they would have
proposed it. For a customer citing a policy that does not exist, that difference is the entire
control.

So the conditions guard is asked of every refund, not only of those about to pay themselves. It
refuses only: it can take a payment away, never grant one, or satisfying it would become a route
around the confidence floor instead of a check standing on top of it.

This was found by the evaluation gate **refusing a change whose headline number had improved**.

## Zero means two different things, and that is deliberate

`auto_refund_limit_paise = 0` is a **kill switch**: nothing pays automatically.

`max_tokens_per_run = 0` means **no ceiling**.

The asymmetry is uncomfortable and it is the right way round. A budget that stopped every run the
moment it was set to zero would make the safe way to switch a budget off indistinguishable from the
harshest setting there is — and an operator reaching for "turn this off" during an incident would
stop the agent dead. For the refund limit, "stop paying" is exactly what zero should mean.

A budget's seconds are the run's **own model time**, never wall-clock since the message arrived. A
run waiting for a person can sit for days, and charging that against a budget would hand over every
refund somebody took a lunch break over.

## Evaluation records on one machine and replays on another

150 labelled cases. Model replies are recorded here, where the GPU is, and replayed in CI, which has
no GPU and no Ollama. Recordings are keyed by a hash of **model + prompt**, so changing a prompt
makes them stop matching and the gate fails loudly rather than scoring against stale replies.

A recording is keyed by what the model was **asked**, not by what it said, so a hand-edited reply
still has to be found under the prompt that produced it.

The consequence worth stating: replay is deterministic, so replay **can never show drift**. Only a
live re-run can. That is what `python -m evals verify` is for, and any release that skips it has not
proven its recordings still match the model.

## The judge is published, not gated

A local model scoring the agent's decisions sounded good and measured badly: it rated a double
refund "appropriate", scoring 0.00 on appropriateness with 0.32–0.41 agreement with the
deterministic scorer.

It is recorded and shown, and it gates nothing. **A classifier you cannot trust cannot tell you
where to look.** The same reasoning is why the failure taxonomy is deterministic rules over what a
run left behind — its stop reason, its steps, what was paid — and not a model's opinion.

## Two things measured, then not built

Both were on the plan. Both were measured first.

**A routing ladder** sending cheap steps to a 3B model could only touch the classify and extract
prompts — 18.1 % of all prompt text, because the planning prompt is 82 % of it and decides money. At
a third the rate, the ceiling was about **9.5 %**, before any loss from a smaller model failing
schemas.

**A reply cache** would have served **0.15 %** of prompts: one repeat in 669, counted over a full
replay. Every prompt embeds the customer's message, so two prompts collide only when two customers
write the same words in the same situation. No TTL, store or key changes that.

Instead, the tool list stopped being JSON schema. `maxLength`, `pattern`, `minimum` and `maximum`
are enforced in `app/contracts.py` whatever the prompt says, so sending them to the model guaranteed
nothing that was not already guaranteed. That was **14 % off every prompt** — more than the ladder's
ceiling, for less code.

The measurement that killed each idea cost an afternoon. Building either would have cost more and
published a number that was mostly the prompt trimming anyway.

## Costs are reference costs, and say so

The models run locally and cost electricity, not API fees. Each is priced as a comparable hosted
model would be, so the figures mean something to a reader and move correctly when the work changes.
The rate is arbitrary, it is the same on both sides of every comparison, and the ratio is what
survives.

**Embedding tokens are not counted.** Every policy search embeds its question, and those tokens are
in no run's cost — under 1 %, but the published figures are low by that much and both the page and
the cost document say so. Counting them properly means changing the recordings format and
re-recording the whole set, which was not worth an hour of a laptop's time for a sub-1 % correction.

## Latency is published as not attributable

The measured p50 went up. It is not reported as a result, because the machine throttles a third of
the way into any sample while token counts stay byte-identical between passes.

Twelve runs on a thermally limited laptop cannot separate the code from the hardware. The run-by-run
timings are published so a reader can see that for themselves. A number claiming a latency
improvement from that sample would be a number about a laptop.

## The screen is loopback-only and has no login

It binds to `127.0.0.1`, refuses any request naming another host, sets `default-src 'none'`, and
takes the operator's name from an environment variable. There is no authentication because there is
no network path to it.

That is a decision with an expiry date. The moment this is exposed, authentication has to come
first: the pages carry customer emails and a button that approves real refunds.

## What is still simulated

Stated plainly, because a demo can make this ambiguous:

- **Intake reads a JSONL file.** There is no mailbox and no chat channel.
- **The ledger is a local table** seeded from a fixture. The customers and orders are invented.
- **A paid refund is a row** in a local table. No money moves anywhere.

The agent, the retrieval, the durability, the guardrails and the evaluation are real. The edges are
not, and the numbers in this repository describe the agent, not a deployment.
