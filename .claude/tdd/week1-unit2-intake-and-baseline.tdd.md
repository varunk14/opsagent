# TDD evidence — Week 1, Units 2 and 3: intake, and the baseline

**Date:** 2026-09-13
**Branch:** main
**Commits:** `4ecf1cd` (RED) → `683eba2` (GREEN) → `f5fa7ab` (review fixes)
→ `e335432` (baseline) → `2592ecf` (Redis) → `b5e5ca5` (baseline review fixes)

## Source plan

The inline ECC `/plan` of 2026-09-13, Tasks 3 and 4. Continues
`week1-unit1-schema-and-contracts.tdd.md`.

## User journeys

4. As a poller that restarts mid-batch, I want re-reading a message I already
   handled to change nothing, so a restart cannot become a second refund.
5. As whoever reads the inbox, I want a malformed message to stop the pass
   loudly, so a customer's email is never skipped in silence.
6. As the person who will claim in week 9 that cost fell, I want the before-number
   recorded now by the same arithmetic, so the claim can be checked.

## Task report

### Task 3 — intake and the first adapter

`app/intake.py` records a message as a run using `INSERT ... ON CONFLICT
(idempotency_key) DO NOTHING`, so the duplicate check happens inside Postgres
while the row is written. `app/adapters/fixture.py` reads JSONL and refuses
malformed lines by number. `app/poll.py` runs one bounded pass.

* **RED:** `ModuleNotFoundError: No module named 'app.intake'`
* **GREEN:** 113 passed
* **Week 1 done-when, demonstrated:** `python -m app.poll fixtures/inbox.jsonl`
  → 4 accepted, 0 duplicate. Run again → 0 accepted, 4 duplicate. Four rows.

### Task 3a — findings from review, fixed in `f5fa7ab`

**HIGH, transaction ownership.** `poll_once` called `connection.commit()` after
a `connection.transaction()` block. Dead on every path that existed, but on a
connection with work already open it either raised from inside psycopg or
committed the caller's unrelated statements as a side effect of polling. Both
reproduced by the reviewer. It refuses a non-idle connection now.

**HIGH, unbounded pass.** One pass was one transaction over an inbox of any size.
Bounded now.

**Found by the tests while fixing that.** The bound counts runs written, not
messages read. An adapter has no memory and re-offers the whole inbox from the
start, so a limit counting duplicates would spend its whole budget recognising
handled messages and stop in the same place forever. This was caught by
`test_the_rest_is_taken_on_the_following_pass` failing.

**HIGH, silent loss on a forged key.** A Message-ID is chosen by the sender.
`DO NOTHING` was discarding the second message's text and returning as though
nothing had happened, so a genuine refund request arriving after someone claimed
its key simply ceased to exist. The conflict path now compares stored against
incoming and reports a collision. Counted rather than raised, so a forged key
cannot be used to stop the poller.

### Task 4 — the baseline

`app/baseline.py` measures naive runs and writes `BASELINE.md`.

### Task 4a — findings from review, fixed in `b5e5ca5`

**HIGH.** Token counts were read with a default of zero, so any reply missing
them became a run that used no tokens and cost nothing — the exact failure the
module's `summarise` guard was written to prevent, reintroduced behind the
network pragma. Extracted as `read_token_counts`, which refuses, and tested.

**HIGH.** The error net missed malformed replies and dropped connections.

**HIGH.** `BASELINE.md` did not say that a nearest-rank p95 over a small sample
is just the slowest run. The commit message claimed it did. Corrected in both.

**MEDIUM.** Not reproducible: no pinned sampling, no warm-up, sample size
inherited from a shared fixture. All three fixed. Two consecutive measurements
now produce identical token counts.

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 24 | A message becomes exactly one run | `test_intake.py::test_a_message_becomes_a_run` | integration | PASS |
| 25 | The same message twice produces one run | `test_intake.py::test_the_same_message_twice_produces_one_run` | integration | PASS |
| 26 | The second caller is handed the existing run | `test_intake.py::test_the_second_caller_is_handed_the_existing_run` | integration | PASS |
| 27 | **A repeat does not disturb a run already underway** | `test_intake.py::test_a_repeat_does_not_disturb_a_run_already_underway` | integration | PASS |
| 28 | **A collision carrying different text is reported, not swallowed** | `test_intake.py::test_a_collision_carrying_different_text_is_reported` | integration | PASS |
| 29 | An honest re-delivery is not reported as a collision | `test_intake.py::test_an_honest_redelivery_is_not_reported_as_a_collision` | integration | PASS |
| 30 | The row matches every field of the contract, not the column defaults | `test_intake.py::test_the_row_matches_the_record_the_contract_describes` | integration | PASS |
| 31 | `accept` leaves the commit to its caller | `test_intake.py::test_accept_leaves_the_commit_to_its_caller` | integration | PASS |
| 32 | A malformed line names itself and stops the read | `test_fixture_adapter.py` (2 tests) | unit | PASS |
| 33 | Reading the same file twice gives the same keys | `test_fixture_adapter.py::test_reading_the_same_file_twice_gives_the_same_keys` | unit | PASS |
| 34 | **A bad line leaves no half-finished pass** | `test_poll.py::test_a_bad_line_leaves_no_half_finished_pass` | integration | PASS |
| 35 | **A pass refuses a connection that is not idle** | `test_poll.py::test_polling_on_a_connection_already_in_a_transaction_is_refused` | integration | PASS |
| 36 | A pass stops at its limit and says more is waiting | `test_poll.py::test_a_pass_stops_at_its_limit` | integration | PASS |
| 37 | **The rest is taken on the following pass** (the livelock guard) | `test_poll.py::test_the_rest_is_taken_on_the_following_pass` | integration | PASS |
| 38 | Collisions are counted rather than stopping the pass | `test_poll.py::test_collisions_are_counted_rather_than_stopping_the_pass` | integration | PASS |
| 39 | The runs outlive the connection that made them | `test_poll.py::test_the_runs_outlive_the_connection_that_made_them` | integration | PASS |
| 40 | **A response without a token count is refused** | `test_baseline.py` (2 parametrised) | unit | PASS |
| 41 | A run that genuinely produced nothing is still readable | `test_baseline.py::test_a_run_that_genuinely_produced_nothing_is_still_readable` | unit | PASS |
| 42 | Cost is exact decimal arithmetic | `test_baseline.py::test_cost_is_exact_decimal_arithmetic` | unit | PASS |
| 43 | **Changing the rate moves both sides by the same factor** | `test_baseline.py::test_changing_the_rate_moves_both_sides_by_the_same_factor` | unit | PASS |
| 44 | Percentiles use nearest rank | `test_baseline.py` (3 tests) | unit | PASS |
| 45 | A baseline of no runs cannot be built at all | `test_baseline.py::test_a_baseline_of_no_runs_cannot_be_built_at_all` | unit | PASS |
| 46 | Tokens per run is the exact mean | `test_baseline.py::test_tokens_per_run_is_the_exact_mean` | unit | PASS |

Row 43 is the one that justifies using an invented price. Double the rate and
every cost doubles, so the before-and-after ratio is untouched — and the ratio
is the claim.

## Coverage

```
.venv/bin/python -m pytest
139 passed
TOTAL  370 stmts  4 missed  99%
Required test coverage of 80% reached. Total coverage: 98.92%
```

## Week 1 acceptance, against the handbook

| Handbook requirement | State |
|---|---|
| Compose with Postgres + pgvector + Redis | Done. Redis present, nothing consumes it yet, and the compose comment says so. |
| The six tables | Done, `migrations/001_schema.sql`. |
| Intake inserting a typed `runs` row | Done, via a channel-agnostic core and a JSONL adapter. |
| Done when: a message arrives and a row appears with the right fields | Demonstrated above. |
| Do not skip: a naive baseline cost per run | Done, `BASELINE.md` + `baseline-measurements.json`. |

Deviation, agreed at plan time: intake is channel-agnostic with a fixture
adapter rather than a Gmail poller. Gmail becomes a second adapter and needs an
App Password, which only the repository owner can create.

## Known gaps, carried forward

* A Message-ID is sender-chosen, so the idempotency key is forgeable. Collisions
  are now detected and counted; the colliding payload is still discarded rather
  than quarantined, which needs the dead-letter table in week 4.
* `IntakeResult.created` is an existence oracle if ever echoed back to a message
  source. Not reachable today; matters when a webhook adapter appears.
* No content checksum on applied migrations.
* `CREATE EXTENSION` needs elevated privileges; a real deployment should separate
  the bootstrap role from the runtime one.
