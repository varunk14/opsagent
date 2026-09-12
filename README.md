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

## What is here so far

`experiments/` holds standalone scripts, each isolating one problem and its fix. Several
are written to fail first, because the failure is the part worth seeing.

| Script | Question it answers | Outcome |
|---|---|---|
| `extract_refund_details.py` | Can a local model read the email and return usable data? | No. 0 of 5 attempts produced anything a program could parse. |
| `validate_refund_details.py` | What makes it reliable? | Constrained output plus a schema check at the boundary. 5 of 5. |
| `order_lookup_agent.py` | How does the agent learn how much was charged? | It calls a lookup tool and decides from the result, with a limit on how long it may keep going. |
| `duplicate_refund_bug.py` | What happens when the payment API times out? | The retry pays the customer twice. Rs 3,600 lost, and no error is raised. |
| `prevent_duplicate_refunds.py` | How is that fixed? | An idempotency key derived from the operation rather than the attempt. |

## Tests

    uv pip install --python .venv/bin/python pytest pytest-cov
    .venv/bin/python -m pytest

37 tests, 99% statement coverage, with the run failing below 80%. Network calls
and interactive drivers are marked `# pragma: no cover`; business logic is not
excluded.

The tests worth reading first are in `tests/test_refund_idempotency.py`. Two of
them assert that the *broken* versions are still broken, because a fix is only
meaningful while the bug it fixes remains demonstrable.

## Planned

Durable runs on Postgres so work survives a restart, policy retrieval with pgvector, a
human approval queue for refunds above a threshold, per-step cost and latency tracing,
an evaluation suite gating every pull request, and a named taxonomy of failure modes.

## Running the experiments

    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python pydantic httpx
    ollama serve &
    .venv/bin/python experiments/extract_refund_details.py 5

`duplicate_refund_bug.py` and `prevent_duplicate_refunds.py` use no model and run
instantly. The others need Ollama with `llama3.2` and `llama3.1:8b` pulled.
