# TDD evidence — Week 1, Unit 1: schema and typed contracts

**Date:** 2026-09-13
**Branch:** main
**Commits:** `ae2e3f1` (RED) → `eac6880` (GREEN) → `3217740` (review fixes)

## Source plan

No `*.plan.md` artifact. The plan was produced inline by ECC `/plan` in the
session of 2026-09-13 and confirmed by the user before any code was written. It
covers Week 1 of `../jobs/P1_HANDBOOK.md` Part 5, "Skeleton and intake". The
schema originates from that handbook, lines 928–997.

Two decisions were taken by the user at plan time and shaped what follows:

* Intake is channel-agnostic, with a fixture adapter first and Gmail later as a
  second adapter. Nothing in Unit 1 needs a credential.
* The full ECC chain runs per unit: tests first, then implementation, then
  `python-reviewer` and `security-reviewer`, then fixes, then commit.

## User journeys

1. As the intake layer, I want every incoming message validated at the boundary,
   so malformed input never reaches the database.
2. As an operator, I want a run's status confined to the six known states, so an
   unknown state cannot silently exist.
3. As the person who pays for mistakes, I want cost stored as exact decimal and
   the idempotency key enforced by the database itself, so a duplicate cannot be
   created by a race that Python loses.

## Task report

### Task 1 — the six tables

`migrations/001_schema.sql` creates `runs`, `customers`, `orders`, `approvals`,
`tool_calls` and `policy_chunks`, plus the pgvector extension and the index the
worker will poll on.

Two departures from the handbook's sketch, both promoting a comment into an
enforced constraint: `CHECK` constraints on the status columns, and explicit
foreign keys.

* **RED:** `ModuleNotFoundError: No module named 'app'` — tests referenced an
  implementation that did not exist.
* **GREEN:** `.venv/bin/python -m pytest` → 73 passed.

### Task 2 — typed contracts

`app/contracts.py` defines `Channel`, `RunStatus`, `IncomingMessage` and
`RunRecord`. The idempotency key is a property of the message, derived from
channel and message id only, and `RunRecord.from_message` takes it rather than
recomputing it, so there is exactly one definition of it.

### Task 2a — findings from review, fixed in `3217740`

`python-reviewer` returned **block** on one CRITICAL and one HIGH.

**CRITICAL — `apply_migrations` committed nothing.** It executed every
statement, recorded them, and returned the list of filenames it had applied.
Closing the connection then rolled all of it back. psycopg does that silently:
no exception, nothing logged. The return value read exactly like success against
an empty database.

**HIGH — the suite could not have caught it.** Every test ran against a database
that an earlier session had already migrated, so the tests proved the schema was
present without ever proving this code put it there. On a reused database, the
branch that actually executes migration SQL never ran at all, while coverage
stayed above the gate and the suite read green.

`security-reviewer` returned two HIGH findings, both fixed:

* No length bound on anything the sender controls. An unbounded body is stored
  whole in jsonb and later paid for by the token.
* Untrusted text was stored in `state` indistinguishable from fields we write
  ourselves, immediately before prompt construction lands.

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | The idempotency key is derived from message identity and nothing else | `test_contracts.py::test_key_ignores_everything_except_the_message_identity` | unit | PASS |
| 2 | The same id on a different channel is a different message | `test_contracts.py::test_the_same_id_on_a_different_channel_is_a_different_message` | unit | PASS |
| 3 | A run takes its key from the message rather than recomputing it | `test_contracts.py::test_a_run_is_built_from_a_message_without_restating_the_key` | unit | PASS |
| 4 | A cost given as a float is refused | `test_contracts.py::test_cost_given_as_a_float_is_rejected` | unit | PASS |
| 5 | A status outside the six is refused by pydantic | `test_contracts.py::test_an_unknown_status_is_rejected` | unit | PASS |
| 6 | Blank id, sender or body is refused | `test_contracts.py` (3 tests) | unit | PASS |
| 7 | A naive timestamp is refused | `test_contracts.py::test_a_timestamp_without_a_timezone_is_rejected` | unit | PASS |
| 8 | Oversized body, subject, sender and id are refused | `test_contracts.py` (4 tests) | unit | PASS |
| 9 | Customer-written text is confined to `state["untrusted"]` | `test_contracts.py` (2 tests) | unit | PASS |
| 10 | The database refuses a repeated idempotency key | `test_schema.py::test_the_same_idempotency_key_cannot_be_inserted_twice` | integration | PASS |
| 11 | Two genuinely different messages both get a row | `test_schema.py::test_two_different_messages_both_get_a_row` | integration | PASS |
| 12 | A status outside the six is refused by the database | `test_schema.py::test_a_status_outside_the_state_machine_is_refused` | integration | PASS |
| 13 | All six real statuses are accepted | `test_schema.py` (6 parametrised) | integration | PASS |
| 14 | An order cannot belong to a customer who does not exist | `test_schema.py::test_an_order_cannot_belong_to_a_customer_who_does_not_exist` | integration | PASS |
| 15 | A tool-call key can only be recorded once | `test_schema.py::test_a_tool_call_key_can_only_be_recorded_once` | integration | PASS |
| 16 | Cost is `numeric(10,6)`, amounts are `bigint`, embeddings are `vector(768)` | `test_schema.py` (3 tests) | integration | PASS |
| 17 | The worker can find due runs without a sequential scan | `test_schema.py::test_the_worker_can_find_due_runs_without_a_sequential_scan` | integration | PASS |
| 18 | **A fractional paisa is rounded, not refused** (finding, recorded) | `test_schema.py::test_a_fractional_paisa_is_silently_rounded_not_refused` | integration | PASS |
| 19 | Migrating an empty database applies every migration | `test_migrations.py::test_migrating_an_empty_database_applies_every_migration` | integration | PASS |
| 20 | **The schema survives the connection that created it closing** | `test_migrations.py::test_the_schema_survives_the_connection_that_created_it` | integration | PASS |
| 21 | A second run applies nothing | `test_migrations.py::test_a_second_run_applies_nothing` | integration | PASS |
| 22 | A migration filename that would sort wrongly is refused | `test_migrations.py::test_a_migration_that_would_sort_wrongly_is_refused` | unit | PASS |
| 23 | The database URL comes from the environment | `test_migrations.py` (2 tests) | unit | PASS |

Row 18 is the one whose meaning changed while it was being written. The test was
drafted asserting that a `bigint` column rejects `360000.5`. It does not:
Postgres rounds it to `360001` and says nothing. Verified directly:

```
INSERT INTO t VALUES (360000.5), (360000.4), (1.9);
 360001
 360000
      2
```

The column type is therefore not the defence. The defence is the typed boundary,
which refuses a float outright. The test now records that, so anyone later
assuming the database guards this meets a contradiction in writing.

Row 20 is the regression test for the CRITICAL finding. It failed before
`3217740` and passes after.

## Coverage

```
.venv/bin/python -m pytest
88 passed
app/contracts.py   63 stmts   0 missed   100%
app/db.py          28 stmts   0 missed   100%
TOTAL             224 stmts   1 missed    99%
Required test coverage of 80% reached. Total coverage: 99.55%
```

The single uncovered statement is in `experiments/duplicate_refund_bug.py`, which
predates this unit.

## Known gaps, carried forward deliberately

* **A Message-ID is chosen by the sender.** The idempotency key is derived from
  it, so it can be forged. Someone who learns a customer's Message-ID could
  pre-empt the key and have a genuine complaint silently treated as a duplicate.
  The answer belongs with the intake adapters, not here.
* **No content checksum on applied migrations.** `schema_migrations` records the
  filename only, so a migration edited after it was applied is never detected —
  silent schema drift between environments.
* **`CREATE EXTENSION` needs elevated privileges.** Fine for the local stack; a
  real deployment should separate a one-time bootstrap role from the runtime one.

## Running this

```bash
docker compose up -d db
.venv/bin/python -m pytest
```

Tests needing Postgres are marked `db` and **fail rather than skip** when it is
absent. A skipped test that reads as green is the failure mode this project
exists to demonstrate. To run only what needs no database:

```bash
.venv/bin/python -m pytest -m "not db" --no-cov
```
