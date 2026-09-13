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
| `app/contracts.py` | The typed boundary. Untrusted input becomes one validated shape, or is refused. |
| `app/db.py` | Connecting, and a migration runner that is safe to run on every start-up. |
| `app/intake.py` | A message becomes a run, exactly once, with the database arbitrating. |
| `app/adapters/fixture.py` | The first intake adapter. Reads JSONL; Gmail will be the second. |
| `app/poll.py` | One pass over an inbox. Owns its transaction, bounded, all or nothing. |
| `app/baseline.py` | What a run costs before any optimisation, recorded so week 9 has something to compare against. |

Six tables in `migrations/001_schema.sql`: `runs` is the durable spine, one row
per request; then `customers`, `orders`, `approvals`, `tool_calls` and
`policy_chunks`.

### Running it

    docker compose up -d db
    .venv/bin/python -c "from app.db import connect, apply_migrations; apply_migrations(connect())"
    .venv/bin/python -m app.poll fixtures/inbox.jsonl

Four rows appear. Run it again and none do — the point of the whole module.

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

139 tests, 99% statement coverage, with the run failing below 80%. Tests that
need Postgres are marked `db` and **fail rather than skip** when it is absent,
because a skipped test that reads as green is the failure this project is about.
To run only what needs no database:

    .venv/bin/python -m pytest -m "not db" --no-cov

The tests worth reading first are `tests/test_intake.py` and
`tests/test_refund_idempotency.py`. Several assert that *broken* versions are
still broken, because a fix only means something while the bug it fixes remains
demonstrable.

Two of them record findings that contradict a reasonable assumption: a `bigint`
column does not reject a fractional paisa, it silently rounds it; and a migration
runner that does not commit reports success against an empty database.

## Cost

`BASELINE.md` holds what one run costs before any optimisation — tokens, latency,
and a cost derived from a fixed reference rate. Week 9 is measured against it.
The rate is arbitrary and says so; it is the same on both sides, so the ratio is
what survives.

## Planned

A durable worker that survives a restart, policy retrieval with pgvector, a human
approval queue for refunds above a threshold, per-step cost and latency tracing,
an evaluation suite gating every pull request, and a named taxonomy of failure
modes.

## Running the experiments

    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python pydantic httpx "psycopg[binary]"
    ollama serve &
    .venv/bin/python experiments/extract_refund_details.py 5

`duplicate_refund_bug.py` and `prevent_duplicate_refunds.py` use no model and run
instantly. The others need Ollama with `llama3.2` and `llama3.1:8b` pulled.
