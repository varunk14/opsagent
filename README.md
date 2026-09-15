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
| `app/intake.py`, `app/poll.py` | A message becomes a run exactly once; a bounded pass owns its transaction. |
| `app/adapters/fixture.py` | The first intake adapter. Reads JSONL; Gmail will be the second. |
| `app/llm.py` | Asks a local model for JSON that fits a schema, retries with the error fenced as data, counts every attempt's tokens. |
| `app/tools.py` | What the agent may propose, with argument limits. No implementations: nothing can execute yet. |
| `app/graph/` | The LangGraph agent: classify → extract → retrieve → plan. Proposes one checked action; never touches the database. |
| `app/embeddings.py`, `app/policies.py` | Policy documents chunked, embedded locally with `nomic-embed-text`, stored in pgvector. Reloading unchanged documents embeds nothing. |
| `app/retrieval.py` | Search by meaning: nearest passages within a distance cutoff, same embedding model only. |
| `app/run_agent.py` | Claims a queued run with `FOR UPDATE SKIP LOCKED`, walks the graph, records the proposal, its policy sources and its exact cost. |
| `app/baseline.py` | What a run costs before any optimisation, so week 9 has something to compare against. |

The six tables are defined in `migrations/`; `policies/` holds six short fictional store policies.

### Running it

    docker compose up -d
    ollama pull llama3.1:8b && ollama pull nomic-embed-text
    .venv/bin/python -m app.policies                      # migrate, then load policies
    .venv/bin/python -m app.poll fixtures/inbox.jsonl     # messages become runs
    .venv/bin/python -m app.run_agent                     # each run gets one proposal

Priya's email ("charged twice for order #4821") is classified `duplicate_charge`, retrieves the
duplicate-payment policy -- which never uses the words "charged" or "twice" -- and ends waiting for
a decision with the proposal `get_order(4821)`. Nothing executes: that is week 4.

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

Over 300 tests with the run failing below 80% coverage. Tests that need Postgres are marked `db`
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

## Cost

`BASELINE.md` holds what one run costs before any optimisation — tokens, latency,
and a cost derived from a fixed reference rate. Week 9 is measured against it.
The rate is arbitrary and says so; it is the same on both sides, so the ratio is
what survives.

## Planned

Real tools behind idempotency keys with retries and a dead-letter queue (week 4), a human approval
queue and policy limits in code (week 5), tracing and a live deployment (week 6), an evaluation
suite gating every pull request (week 7), a failure taxonomy (week 8), and cost routing measured
against `BASELINE.md` (week 9).

## Running the experiments

    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python -r requirements-dev.txt
    ollama serve &
    .venv/bin/python experiments/extract_refund_details.py 5

`duplicate_refund_bug.py` and `prevent_duplicate_refunds.py` use no model and run
instantly. The others need Ollama with `llama3.2` and `llama3.1:8b` pulled.
