"""
Tools that act for real, each call keyed so a repeat replays instead of re-running.

The test the handbook asks for comes first: issue_refund twice with one key, one
refund. The rest keep that guarantee honest -- two workers racing on the same
key, a key reused for a different operation, and a ledger that cannot pay back
more than was actually taken.
"""

import threading
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

from app import executor as executor_module
from app.contracts import ProposedAction
from app.executor import KeyReused, execute, operation_key

pytestmark = pytest.mark.db

PRIYA = "priya@example.com"


def insert_run(
    connection: psycopg.Connection, key: str | None = None, sender: str | None = PRIYA
) -> UUID:
    run_id = uuid4()
    state = {"untrusted": {"sender": sender}} if sender is not None else {}
    connection.execute(
        """
        INSERT INTO runs (id, channel, status, current_node, state, idempotency_key)
        VALUES (%s, 'email', 'running', 'act', %s, %s)
        """,
        (run_id, Jsonb(state), key or f"email_msg_{run_id}"),
    )
    return run_id


def insert_order(
    connection: psycopg.Connection, order_id: str = "4821", charges: tuple[int, ...] = (360_000, 360_000)
) -> None:
    connection.execute(
        "INSERT INTO customers (email, name) VALUES (%s, 'Priya') ON CONFLICT DO NOTHING", (PRIYA,)
    )
    connection.execute(
        "INSERT INTO orders (id, customer_email, amount_paise, status) VALUES (%s, %s, %s, 'paid')",
        (order_id, PRIYA, charges[0]),
    )
    for amount in charges:
        connection.execute(
            "INSERT INTO charges (order_id, amount_paise) VALUES (%s, %s)", (order_id, amount)
        )


def refund(order_id: str = "4821", amount_paise: int = 360_000) -> ProposedAction:
    return ProposedAction(
        tool="issue_refund",
        args={"order_id": order_id, "amount_paise": amount_paise, "reason": "charged twice"},
        confidence="0.9",
        reasoning="the ledger shows two charges",
    )


def lookup(order_id: str = "4821") -> ProposedAction:
    return ProposedAction(
        tool="get_order", args={"order_id": order_id}, confidence="0.9", reasoning="check the ledger"
    )


def refunds_for(connection: psycopg.Connection, order_id: str = "4821") -> list[tuple]:
    return connection.execute(
        "SELECT amount_paise, idempotency_key FROM refunds WHERE order_id = %s", (order_id,)
    ).fetchall()


# --- the handbook's done-when ---------------------------------------------------


def test_refund_called_twice_with_one_key_refunds_once(db):
    run_id = insert_run(db)
    insert_order(db)

    first = execute(db, run_id, 5, refund())
    second = execute(db, run_id, 5, refund())

    assert len(refunds_for(db)) == 1
    assert first.replayed is False
    assert second.replayed is True


def test_the_repeat_returns_the_original_refund(db):
    """The caller must end up holding the refund that actually exists."""
    run_id = insert_run(db)
    insert_order(db)

    first = execute(db, run_id, 5, refund())
    second = execute(db, run_id, 5, refund())

    assert second.result == first.result
    assert first.result["refunded"] is True
    stored_id = db.execute("SELECT id FROM refunds WHERE order_id = '4821'").fetchone()[0]
    assert first.result["refund_id"] == stored_id


def test_the_key_names_the_operation_not_the_attempt():
    run_id = UUID("00000000-0000-0000-0000-000000000088")

    assert operation_key(run_id, 5, "issue_refund") == operation_key(run_id, 5, "issue_refund")
    assert operation_key(run_id, 5, "issue_refund") == f"{run_id}:step_5:issue_refund"


def test_distinct_steps_are_not_deduplicated(db):
    """Guards against over-correcting: two different operations both happen."""
    run_id = insert_run(db)
    insert_order(db)

    execute(db, run_id, 5, refund(amount_paise=100_000))
    execute(db, run_id, 6, refund(amount_paise=100_000))

    assert len(refunds_for(db)) == 2


def test_every_call_is_recorded_with_its_result(db):
    run_id = insert_run(db)
    insert_order(db)

    result = execute(db, run_id, 5, refund())

    stored = db.execute(
        "SELECT run_id, tool, args, result FROM tool_calls WHERE idempotency_key = %s",
        (operation_key(run_id, 5, "issue_refund"),),
    ).fetchone()
    assert stored[0] == run_id
    assert stored[1] == "issue_refund"
    assert stored[2] == {"order_id": "4821", "amount_paise": 360_000, "reason": "charged twice"}
    assert stored[3] == result.result


# --- a key cannot be borrowed ----------------------------------------------------


def test_a_key_reused_with_different_arguments_is_refused(db):
    """Replaying the old result for a different request would report a refund that never happened."""
    run_id = insert_run(db)
    insert_order(db)
    execute(db, run_id, 5, refund(amount_paise=100_000))

    with pytest.raises(KeyReused):
        execute(db, run_id, 5, refund(amount_paise=200_000))

    assert len(refunds_for(db)) == 1


def test_a_key_from_another_run_is_refused(db):
    first_run = insert_run(db)
    other_run = insert_run(db)
    insert_order(db)
    execute(db, first_run, 5, refund())
    borrowed = operation_key(first_run, 5, "issue_refund")
    db.execute(
        "UPDATE tool_calls SET run_id = %s WHERE idempotency_key = %s", (other_run, borrowed)
    )

    with pytest.raises(KeyReused):
        execute(db, first_run, 5, refund())


# --- races -----------------------------------------------------------------------


def test_two_workers_racing_on_one_key_refund_once(fresh_database):
    with psycopg.connect(fresh_database) as setup:
        run_id = insert_run(setup)
        insert_order(setup)

    barrier = threading.Barrier(2)
    results = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with psycopg.connect(fresh_database) as connection:
                barrier.wait()
                results.append(execute(connection, run_id, 5, refund()))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    with psycopg.connect(fresh_database) as check:
        assert len(refunds_for(check)) == 1
    assert sorted(result.replayed for result in results) == [False, True]
    assert results[0].result == results[1].result


def test_two_different_refunds_cannot_together_exceed_what_was_charged(fresh_database):
    """The cap is checked under a lock on the order, so two keys racing cannot both slip under it."""
    with psycopg.connect(fresh_database) as setup:
        first_run = insert_run(setup)
        second_run = insert_run(setup)
        insert_order(setup, charges=(360_000,))

    barrier = threading.Barrier(2)
    results = []

    def worker(run_id: UUID) -> None:
        with psycopg.connect(fresh_database) as connection:
            barrier.wait()
            results.append(execute(connection, run_id, 5, refund()))

    threads = [threading.Thread(target=worker, args=(run,)) for run in (first_run, second_run)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    with psycopg.connect(fresh_database) as check:
        assert len(refunds_for(check)) == 1
    assert sorted(result.result["refunded"] for result in results) == [False, True]


# --- the ledger's limits -----------------------------------------------------------


def test_a_refund_beyond_what_was_charged_is_refused_as_data(db):
    run_id = insert_run(db)
    insert_order(db, charges=(360_000,))

    outcome = execute(db, run_id, 5, refund(amount_paise=360_001))

    assert outcome.result["refunded"] is False
    assert "charged" in outcome.result["error"]
    assert refunds_for(db) == []


def test_refunds_add_up_against_the_cap(db):
    run_id = insert_run(db)
    insert_order(db, charges=(360_000, 360_000))

    assert execute(db, run_id, 5, refund(amount_paise=360_000)).result["refunded"] is True
    assert execute(db, run_id, 6, refund(amount_paise=360_000)).result["refunded"] is True
    assert execute(db, run_id, 7, refund(amount_paise=1)).result["refunded"] is False


def test_a_refund_for_an_unknown_order_is_refused_as_data(db):
    run_id = insert_run(db)

    outcome = execute(db, run_id, 5, refund(order_id="9999"))

    assert outcome.result["refunded"] is False
    assert "9999" in outcome.result["error"]


def test_a_refused_refund_replays_as_refused(db):
    """Same operation, same answer: a refusal is not retried into a success by accident."""
    run_id = insert_run(db)
    insert_order(db, charges=(100,))

    execute(db, run_id, 5, refund(amount_paise=360_000))
    db.execute("INSERT INTO charges (order_id, amount_paise) VALUES ('4821', 1000000)")
    again = execute(db, run_id, 5, refund(amount_paise=360_000))

    assert again.replayed is True
    assert again.result["refunded"] is False
    assert refunds_for(db) == []


def test_the_database_refuses_an_over_refund_written_around_the_executor(db):
    """The cap lives in Postgres, so code that skips the executor is still stopped."""
    run_id = insert_run(db)
    insert_order(db, charges=(360_000,))
    key = operation_key(run_id, 5, "issue_refund")
    db.execute(
        "INSERT INTO tool_calls (idempotency_key, run_id, tool, args) VALUES (%s, %s, 'issue_refund', '{}')",
        (key, run_id),
    )

    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "INSERT INTO refunds (order_id, amount_paise, reason, run_id, idempotency_key) "
            "VALUES ('4821', 360001, 'bypass', %s, %s)",
            (run_id, key),
        )


def test_a_refund_amount_must_be_positive_in_the_database(db):
    run_id = insert_run(db)
    insert_order(db)
    key = operation_key(run_id, 5, "issue_refund")
    db.execute(
        "INSERT INTO tool_calls (idempotency_key, run_id, tool, args) VALUES (%s, %s, 'issue_refund', '{}')",
        (key, run_id),
    )

    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "INSERT INTO refunds (order_id, amount_paise, reason, run_id, idempotency_key) "
            "VALUES ('4821', 0, 'nothing', %s, %s)",
            (run_id, key),
        )


# --- only the sender's own orders ---------------------------------------------------


def test_another_customers_order_looks_exactly_like_no_order(db):
    """Anyone can email in with a guessed order number; the reply must not confirm it exists."""
    run_id = insert_run(db, sender="dev@example.com")
    insert_order(db)

    order = execute(db, run_id, 1, lookup()).result

    assert order == {"error": "no order 4821"}


def test_a_refund_on_another_customers_order_is_refused(db):
    run_id = insert_run(db, sender="dev@example.com")
    insert_order(db)

    outcome = execute(db, run_id, 5, refund())

    assert outcome.result == {"refunded": False, "error": "no order 4821"}
    assert refunds_for(db) == []


def test_a_run_with_no_sender_reaches_no_order(db):
    run_id = insert_run(db, sender=None)
    insert_order(db)

    assert execute(db, run_id, 1, lookup()).result == {"error": "no order 4821"}
    assert execute(db, run_id, 5, refund()).result["refunded"] is False
    assert refunds_for(db) == []


def test_the_sender_matches_regardless_of_letter_case(db):
    """Mail systems treat Priya@Example.com and priya@example.com as the same person."""
    run_id = insert_run(db, sender="Priya@Example.com")
    insert_order(db)

    assert execute(db, run_id, 1, lookup()).result["order_id"] == "4821"


# --- the other tools ---------------------------------------------------------------


def test_get_order_returns_the_order_its_charges_and_what_was_refunded(db):
    run_id = insert_run(db)
    insert_order(db)
    execute(db, run_id, 5, refund(amount_paise=360_000))

    order = execute(db, run_id, 6, lookup()).result

    assert order["order_id"] == "4821"
    assert order["customer_email"] == PRIYA
    assert order["charges_paise"] == [360_000, 360_000]
    assert order["charged_paise"] == 720_000
    assert order["refunded_paise"] == 360_000


def test_get_order_reports_an_unknown_order_as_data(db):
    run_id = insert_run(db)

    order = execute(db, run_id, 1, lookup("9999")).result

    assert "9999" in order["error"]


def test_escalation_is_recorded_and_always_succeeds(db):
    run_id = insert_run(db)
    action = ProposedAction(
        tool="escalate_to_human", args={"reason": "not sure"}, confidence="0.2", reasoning="unclear"
    )

    outcome = execute(db, run_id, 3, action)

    assert outcome.result == {"escalated": True, "reason": "not sure"}


def test_search_policy_is_not_something_the_executor_runs(db):
    """Retrieval belongs to the graph's retrieve step; the executor only touches the ledger."""
    run_id = insert_run(db)
    action = ProposedAction(
        tool="search_policy", args={"question": "charged twice"}, confidence="0.5", reasoning="look it up"
    )

    with pytest.raises(ValueError, match="search_policy"):
        execute(db, run_id, 2, action)


def test_a_tool_the_executor_does_not_run_records_nothing(db):
    """Refused before any key is claimed, so there is nothing to roll back."""
    run_id = insert_run(db)
    action = ProposedAction(
        tool="search_policy", args={"question": "charged twice"}, confidence="0.5", reasoning="look it up"
    )

    with pytest.raises(ValueError):
        execute(db, run_id, 2, action)

    assert db.execute("SELECT count(*) FROM tool_calls").fetchone()[0] == 0


def test_a_tool_that_crashes_after_its_key_is_claimed_rolls_the_key_back(fresh_database, monkeypatch):
    """
    The key is claimed, then the tool raises. Nothing may stay committed, or the
    retry would find a key with no result and the operation could never finish.
    """
    with psycopg.connect(fresh_database) as setup:
        run_id = insert_run(setup)
    action = ProposedAction(
        tool="escalate_to_human", args={"reason": "not sure"}, confidence="0.2", reasoning="unclear"
    )

    def crash(*_args: object) -> dict:
        raise RuntimeError("tool crashed")

    monkeypatch.setitem(executor_module.TOOLS, "escalate_to_human", crash)
    with psycopg.connect(fresh_database) as connection, pytest.raises(RuntimeError, match="tool crashed"):
        execute(connection, run_id, 3, action)

    with psycopg.connect(fresh_database) as check:
        assert check.execute("SELECT count(*) FROM tool_calls").fetchone()[0] == 0

    monkeypatch.undo()
    with psycopg.connect(fresh_database) as connection:
        retried = execute(connection, run_id, 3, action)
    assert retried.replayed is False
    assert retried.result == {"escalated": True, "reason": "not sure"}


def test_only_the_refund_cap_is_reported_as_a_refusal(db):
    """
    Any other constraint the refund breaks is a bug upstream, not an answer for
    the planner. Built with model_construct to get past the schema that normally
    stops a zero amount, as a loosened schema one day would.
    """
    run_id = insert_run(db)
    insert_order(db)
    unchecked = ProposedAction.model_construct(
        tool="issue_refund",
        args={"order_id": "4821", "amount_paise": 0, "reason": "charged twice"},
        confidence="0.9",
        reasoning="bypasses validation",
    )

    with pytest.raises(psycopg.errors.CheckViolation):
        execute(db, run_id, 5, unchecked)
