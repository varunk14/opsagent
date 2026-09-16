# opsagent

An AI agent that handles customer refund requests end to end, and the tooling needed to
run it safely in production.

**Status: in active development.**

## The problem

A customer writes in:

> "Hi, I think I was charged twice for order #4821 last Tuesday. Can you refund one of
> them? Thanks, Priya"

Handling that automatically means reading the email, checking whether there really were
two charges, deciding whether a refund is allowed, issuing it, and knowing when to hand
the case to a person instead. Every one of those steps can fail in a way that costs money
without raising an error.

## What is here

Two things, kept apart on purpose.

`app/` is the system. `experiments/` is the record of what was learned before it
was written, each script isolating one failure and its fix. Several are written
to fail first, because the failure is the part worth seeing.

### The system

| Module | What it does |
|---|---|
| `app/contracts.py` | The typed boundary. Untrusted input and every model answer become one validated shape, or are refused. |
| `app/db.py` | Connecting, and a migration runner that is safe to run on every start-up. |
| `app/intake.py`, `app/poll.py` | A message becomes a run exactly once; a bounded pass owns its transaction. A message that reuses another's key with different text is quarantined, not lost. |
| `app/adapters/fixture.py` | The first intake adapter. Reads JSONL; Gmail will be the second. |
| `app/llm.py` | Asks a local model for JSON that fits a schema, retries with the error fenced as data, counts every attempt's tokens. |
| `app/tools.py` | What the agent may propose, as the model sees it: names, descriptions and argument limits, and nothing executable. |
| `app/graph/` | The LangGraph agent: classify → extract → retrieve → plan. Proposes one checked action; never touches the database. A run picked back up starts at planning with what its earlier steps found. |
| `app/embeddings.py`, `app/policies.py` | Policy documents chunked, embedded locally with `nomic-embed-text`, stored in pgvector. Reloading unchanged documents embeds nothing. |
| `app/retrieval.py` | Search by meaning: nearest passages within a distance cutoff, same embedding model only. |
| `app/executor.py` | Runs a tool for real, exactly once per operation: every call is keyed `run:step:tool`, and a repeat replays the stored result. An order is reachable only by the customer who placed it. |
| `app/run_agent.py` | Claims a run with `FOR UPDATE SKIP LOCKED` and works it one committed step at a time. A lookup runs and the agent plans again with the result; a refund is judged by the guardrail, then paid once through the executor or put to a person with the reason, and an approved refund is paid exactly as approved. A run whose worker died is reclaimed once its lock expires and continues from its last committed step. An outage is retried with backoff; one sender's lookups are rate limited; worker sessions carry timeouts so a hung connection cannot hold a run. |
| `app/guardrails.py` | Which refunds run without a person: under a limit on the order's total refunds, and at or above a confidence. Both are a row in Postgres, read at the moment of deciding, so `python -m app.guardrails set` changes behaviour with no code change; a limit of Rs 0 stops automatic refunds. |
| `app/approvals.py`, `app/web.py` | The approval queue, and a local screen on `127.0.0.1:8055` to approve or reject with the evidence in front of you. It also lists every run waiting for a person with nothing to approve. Everything shown is escaped, every decision needs the page's token, and the screen cannot pay anything itself. |
| `app/tracing.py`, `app/traces.py` | Every run as one OpenTelemetry trace: spans written with the steps they describe, read back as a tree for the `/runs` pages, and copied to a Langfuse on this machine when one is configured. `python -m app.tracing env` writes that Langfuse's secrets. |
| `app/dead_letters.py` | Runs that ran out of attempts, and quarantined messages, each with the reason. `python -m app.dead_letters` lists them and requeues a run. |
| `app/seed.py` | Loads a small fictional ledger: customers, orders, and the charges behind them. |
| `app/baseline.py` | What a run costs before any optimisation, so later cost work has something to compare against. |
| `evals/` | The golden set, recorded model replies, and the gate that scores every pull request against a committed baseline. |

The tables are defined in `migrations/`, including a refund ledger that refuses, inside Postgres, to
pay back more than an order was charged. `policies/` holds six short fictional store policies.

### Running it

    docker compose up -d
    ollama pull llama3.1:8b && ollama pull nomic-embed-text
    .venv/bin/python -m app.policies                      # migrate, then load policies
    .venv/bin/python -m app.seed                          # the fictional ledger
    .venv/bin/python -m app.poll fixtures/inbox.jsonl     # messages become runs
    .venv/bin/python -m app.run_agent                     # work each run until it is done or waits
    .venv/bin/python -m app.guardrails show               # the limits in force
    OPSAGENT_OPERATOR=yourname .venv/bin/python -m app.web  # approve or reject on http://127.0.0.1:8055/approvals

Priya's email ("charged twice for order #4821") is classified `duplicate_charge` and retrieves the
duplicate-payment policy -- which never uses the words "charged" or "twice". The agent then looks
order 4821 up for real and plans again with what the ledger says. Its Rs 3,600 refund is under the
default guardrail (Rs 5,000, confidence 0.85), so it is paid, once. Lower the limit and the same
refund waits for a person instead, with no code change:

    .venv/bin/python -m app.guardrails set --limit-rupees 1000 --by yourname

Approved on the screen, the next worker pays exactly what was approved; rejected, nothing is paid.

### Tracing

Every run is one OpenTelemetry trace, and its trace id is the run's own id. However many ticks a run
takes, by whichever workers, and however long it waits for an approval, it is one trace. Each tick,
step, model call, policy search, guardrail decision and payment is a span. Spans are written to the
`spans` table in the same transaction as the step they describe, so a step that never committed
leaves no span.

    OPSAGENT_OPERATOR=yourname .venv/bin/python -m app.web  # then open http://127.0.0.1:8055/runs

`/runs` lists the newest runs. Each opens as its trace: every step under its tick, every model call
under its step, each with its model, prompt version, tokens and cost. Below that, the calls' total
sits beside the cost recorded on the run, then the approvals the run waited on. Prompts live in
`prompts/`, and every run records the hash of the prompts it ran under.

To read the same traces in Langfuse, self-hosted on this machine:

    .venv/bin/python -m app.tracing env                  # fresh secrets into .env; prints none of them
    docker compose -f compose.tracing.yml up -d          # Langfuse on http://127.0.0.1:3000
    set -a; . ./.env; set +a                             # the worker's and the screen's settings
    .venv/bin/python -m app.run_agent                    # spans now also go to Langfuse

No login is seeded; create one in the Langfuse UI. Spans are only sent to 127.0.0.1 or localhost.
A worker given any other endpoint refuses to start, and it ignores proxy settings and redirects.
Langfuse takes about 2.3 GB of memory next to the local model.

Costs are reference costs: tokens priced at the fixed rate in `BASELINE.md`, so runs can be
compared. The models run locally and cost nothing. Embedding a search query is traced, but its
tokens are not yet counted in a run's cost; the page says "not counted".

### Evaluation

`evals/golden.jsonl` holds 150 fictional cases, each a customer message and what should happen to
it: 120 ordinary requests of ten kinds, and 30 written to break the agent -- instructions hidden in
the message, someone else's order, more money than was charged, an order that does not exist, a
policy quoted that does not exist. Every case has an order of its own in `evals/ledger.json`, and a
test checks every label against that ledger and the default guardrail.

CI has no GPU, so the real model cannot run there. Instead every case is run once on a machine with
Ollama, and each model reply and embedding is recorded, keyed by the model and a hash of the exact
prompt. CI replays the recordings through the real worker and Postgres, and scores what happened.

    .venv/bin/python -m evals record    # with Ollama: run all 150 cases, keep every reply
    .venv/bin/python -m evals accept    # after a deliberate change: write the baseline and scoreboard
    .venv/bin/python -m evals gate      # anywhere, no model: replay, score, compare
    .venv/bin/python -m evals verify    # with Ollama: record again, report any reply that differs

The scores are task completion; intent and extraction accuracy; precision, recall and false-positive
rate for handing a case to a person; and safety violations -- paying when a person should have
decided, paying a different amount, or paying twice. `gate` runs on every pull request. It fails
when a score falls below `evals/baseline.json`, when anything is unsafe, when the golden set
changed, or when `evals/scoreboard.md` no longer says what the recordings score. A prompt edited
without recording again fails as well, because nothing was ever recorded for it.
`tests/test_evals_bad_prompt.py` checks both ways a bad prompt can reach a pull request.

A label cannot say whether a decision was grounded in the policy and the ledger, so the 25 smoke
cases are also judged by the local model (`prompts/judge.txt`). It is shown what the run saw and did
-- the message, the policy passages, what the lookups returned, where the run came to rest -- and
never the label. Its verdicts are recorded and replayed like every other reply and printed on the
scoreboard, with how often the judge agrees with the exact checks beside its score. It gates
nothing: small local models are poor judges, and this one has not earned it yet.
`python -m evals full` judges every case before a release and writes `evals/full.md`.

The first recording, on llama3.1:8b, is the baseline in `evals/scoreboard.md`, and it is not flattering.
The agent completes 89 of the 150 cases (59%). When a case should reach a person it does 85% of the
time, and 90% of the cases it hands over should be. It also made 19 unsafe payments: refunds under the
limit, paid with no person involved, that a person should have decided -- a claimed double charge
with one charge on the ledger, change-of-mind and damaged-item refunds paid without checking the
policy's conditions, one prompt injection. The guardrail checks the amount and the model's confidence,
not whether the policy allows the refund. Those 19 cases are named on the scoreboard and pinned: a
case that is safe today becoming unsafe fails the gate, and fixing the 19 is the next piece of work.

A recording is keyed by what the model was asked, not by what it said, so a hand-edited reply would
replay as real. At temperature 0 with a fixed seed the model's replies are byte-identical from run
to run, so `verify` records everything again and any difference is the edit.

### The experiments

| Script | Question it answers | Outcome |
|---|---|---|
| `extract_refund_details.py` | Can a local model read the email and return usable data? | No. 0 of 5 attempts produced anything a program could parse. |
| `validate_refund_details.py` | What makes it reliable? | Constrained output plus a schema check at the boundary. 5 of 5. |
| `order_lookup_agent.py` | How does the agent learn how much was charged? | It calls a lookup tool and decides from the result, with a limit on how long it may keep going. |
| `duplicate_refund_bug.py` | What happens when the payment API times out? | The retry pays the customer twice. Rs 3,600 lost, and no error is raised. |
| `prevent_duplicate_refunds.py` | How is that fixed? | An idempotency key derived from the operation rather than the attempt. |

## Tests

    docker compose up -d db
    .venv/bin/python -m pytest

Over 900 tests with the run failing below 80% coverage. Tests that need Postgres are marked `db`
and **fail rather than skip** when it is absent, because a skipped test that reads as green is the
failure this project is about. CI runs the same checks on every push to `main` and every pull
request, and a pull request cannot merge into `main` until they pass.

Tests against the real local models are opt-in, since CI has no Ollama:

    OPSAGENT_REAL_MODEL=1 .venv/bin/python -m pytest tests/test_policy_search_real.py --no-cov

To run only what needs no database:

    .venv/bin/python -m pytest -m "not db" --no-cov

Several tests record findings that contradict a reasonable assumption: a `bigint` column rounds a
fractional paisa instead of refusing it; a migration runner that does not commit reports success
against an empty database; an outage partway through a run used to lose the cost of the steps that
had finished.

Crash recovery is tested with a real crash. `tests/test_resume.py` starts a worker process, kills it
with SIGKILL the moment its first step commits, and checks that a second worker finishes the run from
that step without repeating it.

## Cost

`BASELINE.md` holds what one run costs before any optimisation — tokens, latency,
and a cost derived from a fixed reference rate. Cost optimisation is measured against it.
The rate is arbitrary and says so; it is the same on both sides, so the ratio is
what survives.

## Reliability

`RELIABILITY.md` holds what the agent gets right and wrong, measured over 150 labelled cases:
task completion, escalation precision and recall, the failure mix across six fixed categories,
and the nineteen refunds an earlier version paid that a person should have decided — how they
were found, and how they were closed. The model's replies are recorded on a machine with a GPU
and replayed in CI, so the gate runs without one; `python -m evals verify` re-runs the set live
and reports what has changed since.

Completion is 72 % with nothing unsafe paid, and the judge is too weak to gate on; both numbers
are published rather than buried. `/failures` on the local screen charts the mix of real runs per
week beside the mix of every accepted baseline, so a category a change made worse is visible
before anything ships.

## Why the durability is hand-rolled

A run is a row in Postgres, not a workflow in Temporal. Workers claim runs with
`FOR UPDATE SKIP LOCKED`, commit after every step, and take back a run whose worker has been silent
for five minutes. Every tool call carries an idempotency key built from the operation -- run, step,
tool -- so a retry replays the first result instead of acting twice.

Temporal would handle all of this. Its cloud version costs money this project does not have, and the
self-hosted server wants memory the local models need. Building it by hand was also the point: the
failure modes are the interesting part, and each one is pinned by a test.

* Calling refund twice with one key produces one refund, including two workers racing on the key.
* A worker killed with SIGKILL mid-run is resumed by another from its last committed step.
* A worker that lost its claim executes nothing.
* An outage waits 30 seconds, then twice as long each time up to an hour; a run out of attempts is
  dead-lettered with its reason in the same statement that marks it dead.
* Two workers racing for a sender's last allowed lookup use it once.
* A refund split in two, or two runs refunding one order at the same moment, is judged as the total.
* An approval is paid once: a stale copy of it, a run requeued by hand, or a worker that lost its
  claim pays nothing more.

## Planned

A live deployment, and cost routing measured against `BASELINE.md`.

## Running the experiments

    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python -r requirements-dev.txt
    ollama serve &
    .venv/bin/python experiments/extract_refund_details.py 5

`duplicate_refund_bug.py` and `prevent_duplicate_refunds.py` use no model and run
instantly. The others need Ollama with `llama3.2` and `llama3.1:8b` pulled.
