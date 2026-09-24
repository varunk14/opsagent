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

## The planner is offered only the tools that apply

A tool offered is a tool the small model will reach for. With policy already retrieved and
`search_policy` still in the list, every run proposed searching for the policy it had just been
handed; with no order named and `get_order` still in the list, every real message looped on looking
up an order that was never mentioned — copying the example id out of the description — until the step
budget ran out and it was handed over. Both are the same mistake: a redundant tool reads as an
instruction to use it.

So the plan prompt withholds what cannot apply. Once policy is in the prompt, `search_policy` is
dropped. When no order is in play — extraction found none, or the case is not about a refund at all —
`get_order` and `issue_refund` are dropped and the prompt says plainly that no order was identified.
Nothing about execution changes: the money guardrails live in code and judge a proposal whether or
not the prompt offered the tool. Withholding a tool only makes the model less likely to propose one
that had nowhere to go, which is why a message naming no order now goes to a person in one step
instead of after a loop.

The list the versions are computed over is still the whole list, so a prompt hash does not shift with
what one message happens to hide; only the rendered prompt for that message is shorter.

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

## MCP is a second description of the tools, not a second way to run them

`app/mcp_server.py` offers the tool catalogue over the Model Context Protocol, so the project can
honestly say the tools are MCP-compatible. It is deliberately a description, not an execution path.
`tools/list` returns each tool's schema exactly as `app/tools.py` holds it; `tools/call` checks the
arguments with the same `validate_tool_call` a run's own proposal is checked with and returns the
validated proposal -- it opens no database, moves no money, and writes no run state. Nothing in the
module imports the execution code, so there is no path from a call to an effect.

That restraint is the point. The single most important line in this system is that the model
proposes and code decides; an MCP `tools/call` wired to real execution would be a second decider,
one that could move money over an unauthenticated channel. Execution stays in run_agent/executor,
behind idempotency keys, guardrails and approvals, reached only by the durable pipeline.

The transport is stdio only, matching the screen's loopback-only stance: no port, no network, no
SSE (which the spec has superseded). A host launches the server as a subprocess and speaks over its
stdin and stdout -- and because stdout is the wire there, logs go to stderr.

## A reply is a row before it is a message

A run that rests owes the customer a word, and that word is written to an `outbox` in the **same
transaction as the outcome** (`app/replies/outbox.py`, called from `app/run_agent.py`). A refund
that committed cannot leave without the reply that says so committing with it, and an outcome that
rolled back takes its reply down too, so a customer's money never moves in silence.

The message is chosen by outcome from a fixed set -- refund issued, handed to a person, order not
found, an enquiry answered -- rendered and frozen into the row. No model writes it, and the only
customer detail it can carry is the order number the agent acted on and the amount it paid: the
render function has no parameter a customer's own words could arrive through, so echoing their text
back to them is impossible by construction rather than by care.

A separate drain (`app/replies/send.py`) sends them, and its ordering is the poll ordering seen from
the other side. A poll records first and confirms the channel afterwards; a drain sends first and
marks the row sent afterwards -- for the same reason, since marking first would let a crash in the
gap drop a reply. Delivery is at-least-once: a failed send keeps its row pending with the error
written down and is retried a few times before it rests as failed for a person.

## Refuse to reply to an address that will only bounce

The mailbox is read over IMAP and written over SMTP with the same account, so a bounce lands back
in the same inbox the poll reads next. Left to happen, the sequence is a loop: the drain sends,
the receiver refuses, the bounce arrives, a fresh run answers it, that reply bounces too. One dead
address would spin the whole inbox.

So each reply is checked before the SMTP session opens (`app/replies/deliverability.py`). The
recipient's domain is refused if it is a reserved test name -- `example.com`, `.invalid`, `.test`,
`.localhost` (RFC 2606), which is what the fixture messages seeded this project with -- and
otherwise looked up for an MX record. A domain that publishes no way to receive mail is one whose
replies would only bounce; `Undeliverable` is raised at the sender, the drain marks the row
`failed` on that pass, and the attempt counter is never touched. Counting attempts on a permanent
no would waste four more drains on a row that will never succeed.

Bounces coming the other way are dropped at intake by a header sniff on the raw bytes
(`app/adapters/mailbox.py`). A `mailer-daemon` `From`, an `Auto-Submitted: auto-` line, an empty
`Return-Path`, or a `multipart/report` `Content-Type` all mark the message `ignored` before it can
become a run. The sniff scans only the header block -- a customer who quotes a bounce in their own
body is still read as a message, which is the failure this whole project is about.

The MX lookup is DNS, which fails in ways that are not the domain's answer. `NXDOMAIN` and
`NoAnswer` come back as `LookupError` (the standard's no); a `Timeout` or `NoNameservers` is left
to propagate and lands in the drain's ordinary retry path -- one flaky DNS query must not silently
mark a real address undeliverable.

## Replay is a new run, never an edit

When a prompt, a policy, or a line of code changes, the question is whether the case that went wrong
before goes right now. Replay (`app/replay.py`) answers it by making a **new** run that carries the
old one's message, worked under whatever is true now; the original row is never modified. The whole
value is the comparison, and a comparison against something that was itself edited proves nothing.

The fresh run copies only what the customer wrote, gets its own id and its own idempotency key
(`replay:<id>`, never the original's, which is unique and would read as a re-delivery), and is
threaded back by a `replay_of` column. It re-enters the pipeline exactly where a new message would,
so it meets every guardrail the original did: replaying a run whose order was already refunded
proposes the refund again and is held for a person, not paid twice.

## The deployment is one file, and only the proxy faces the internet

The whole system runs from `compose.deploy.yml`: Postgres, Redis, Ollama, the screen, the worker,
and Caddy in front. **Only Caddy publishes ports** (80/443). The database, cache, model server,
screen and worker talk over an internal network and are never reachable from outside -- an exposed
Postgres or an open Ollama would each be their own incident. Caddy gets a TLS certificate on its
own, puts the screen behind a password, and proxies the rest through.

It is a manual deploy on purpose: one script run on the box over SSH, no key in GitHub and no
pipeline that can reach production. The thing that deploys is a person already on the machine. Two
small things had to become configurable for the app to run in a container -- where the model server
is (`OPSAGENT_OLLAMA_URL`) and where the screen binds and which host it answers to
(`OPSAGENT_WEB_HOST`, `OPSAGENT_ALLOWED_HOSTS`) -- and both default to the safe local values, so
nothing changed for local use.

## The screen is loopback-only, and public only behind a proxy that authenticates

Locally it binds `127.0.0.1`, refuses any request naming another host, sets `default-src 'none'`,
and takes the operator's name from an environment variable. There is no login because there is no
network path to it.

That was a decision with an expiry date, and the deployment is the expiry. There the screen binds
its container's interface -- reachable only over the internal network, never published -- and Caddy
sits in front, terminating TLS and putting the whole screen behind a password before a single page
is served. Authentication is added at the proxy, exactly where the network path first appears,
rather than grafted as a login into an app that never had users. The pages carry customer emails and
a button that approves real refunds, so nothing there is served before the password.

## What is still simulated

Stated plainly, because a demo can make this ambiguous.

What was simulated and now is not:

- **Intake reads real channels.** A real IMAP mailbox and a real Telegram bot, each adapter
  refusing far more than it parses; the JSONL file remains for offline demos.
- **Replies go back out.** A rested run writes a fixed-template reply and a drain sends it by SMTP
  or Telegram.

What is still simulated:

- **The ledger is a local table** seeded from a fixture. The customers and orders are invented.
- **A paid refund is a row** in a local table. No money moves anywhere.

The agent, the channels, the replies, the retrieval, the durability, the guardrails and the
evaluation are real. The money is not, and the numbers in this repository describe the agent, not a
business.
