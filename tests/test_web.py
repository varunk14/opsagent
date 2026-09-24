"""
The approvals screen: a local page where a person approves or rejects.

Server-rendered, bound to 127.0.0.1, and deliberately thin: it lists what
app.approvals returns and calls approvals.decide. It never loads the executor --
an approved refund is paid by the worker, never by the web process -- and it
shows only what was put to the person: never the run's raw state, never the
worker that held it. Everything on it came from a customer's email or a model's
reading of one, so all of it is escaped, and every form that changes anything
carries a token that a page on another origin can neither read nor send.

These tests commit, so each one gets its own scratch database.
"""

import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from app.approvals import open_approval
from app.contracts import ProposedAction
from app.guardrails import set_limits
from app.web import HOST, create_app

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parent.parent
LOCAL = "http://127.0.0.1"
XSS = "<script>alert('pwned')</script>"
OVER_THE_LIMIT = "Rs 7,200 is not under the Rs 5,000 limit for automatic refunds"


def client_for(dsn: str, operator: str | None = "asha", history_path: Path | None = None) -> TestClient:
    return TestClient(create_app(dsn=dsn, operator=operator, history_path=history_path), base_url=LOCAL)


def insert_run(connection, *, status: str, node: str, state: dict, key: str, locked_by: str | None = None) -> str:
    run_id = uuid4()
    connection.execute(
        "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key, locked_by) "
        "VALUES (%s, 'email', %s, %s, %s, %s, %s)",
        (run_id, status, node, Jsonb(state), key, locked_by),
    )
    return str(run_id)


def waiting_refund(
    dsn: str,
    *,
    body: str = "Hi, I was charged twice for order #4821.",
    reason: str = OVER_THE_LIMIT,
    key: str = "email_msg_web",
    locked_by: str | None = None,
) -> tuple[str, int]:
    """A run waiting on a Rs 7,200 refund, shaped as the worker leaves it."""
    untrusted = {"sender": "priya@example.com", "subject": "Charged twice", "body": body}
    lookup = {
        "step": 1,
        "tool": "get_order",
        "args": {"order_id": "4821"},
        "result": {"order_id": "4821", "charges_paise": [360_000, 360_000], "charged_paise": 720_000, "refunded_paise": 0},
        "replayed": False,
    }
    with psycopg.connect(dsn) as connection:
        run_id = insert_run(
            connection,
            status="waiting_approval",
            node="approval",
            state={"untrusted": untrusted, "agent": {"policy": ["INTERNAL POLICY TEXT"], "steps": [lookup]}},
            key=key,
            locked_by=locked_by,
        )
        action = ProposedAction(
            tool="issue_refund",
            args={"order_id": "4821", "amount_paise": 720_000, "reason": "charged twice"},
            confidence=Decimal("0.9"),
            reasoning="the ledger shows a duplicate charge",
        )
        evidence = {
            **untrusted,
            "classification": {"intent": "duplicate_charge", "confidence": "0.9", "reasoning": "charged twice"},
            "extraction": None,
            "policy_sources": ["duplicate-payments#1"],
            "steps": [lookup],
        }
        approval_id = open_approval(connection, run_id, action, evidence, reason)
    return run_id, approval_id


def token_from(page: str) -> str:
    found = re.search(r'name="csrf" value="([^"]+)"', page)
    assert found, "the page carries no CSRF token"
    return found.group(1)


def run_status(dsn: str, run_id: str) -> tuple:
    with psycopg.connect(dsn) as connection:
        return connection.execute("SELECT status, failure_class FROM runs WHERE id = %s", (run_id,)).fetchone()


def decision(dsn: str, approval_id: int) -> tuple:
    with psycopg.connect(dsn) as connection:
        return connection.execute(
            "SELECT status, decided_by, decision_note FROM approvals WHERE id = %s", (approval_id,)
        ).fetchone()


# --- what the screen shows ----------------------------------------------------------------


def test_a_pending_approval_is_shown_with_what_the_person_needs(fresh_database):
    waiting_refund(fresh_database)

    page = client_for(fresh_database).get("/approvals")

    assert page.status_code == 200
    for expected in ["Rs 7,200", "4821", "Rs 3,600", "0.90", OVER_THE_LIMIT, "priya@example.com", "Charged twice"]:
        assert expected in page.text, f"{expected!r} is missing from the page"


def test_what_the_customer_wrote_is_escaped(fresh_database):
    waiting_refund(fresh_database, body=f"Refund me {XSS}")

    page = client_for(fresh_database).get("/approvals").text

    assert XSS not in page
    assert "&lt;script&gt;" in page


def test_what_the_model_wrote_is_escaped(fresh_database):
    waiting_refund(fresh_database, reason=f"because {XSS}")

    page = client_for(fresh_database).get("/approvals").text

    assert XSS not in page
    assert "&lt;script&gt;" in page


def test_the_worker_that_held_the_run_is_never_shown(fresh_database):
    """Carried from an earlier review: locked_by is a hostname and a process id."""
    waiting_refund(fresh_database, locked_by="build-host-4242")

    assert "build-host-4242" not in client_for(fresh_database).get("/approvals").text


def test_the_runs_raw_state_is_never_shown(fresh_database):
    waiting_refund(fresh_database)

    assert "INTERNAL POLICY TEXT" not in client_for(fresh_database).get("/approvals").text


def test_a_run_handed_to_a_person_without_an_approval_is_shown_too(fresh_database):
    """Security review of Unit B: a refused refund or an escalation must be visible somewhere."""
    with psycopg.connect(fresh_database) as connection:
        insert_run(
            connection,
            status="waiting_approval",
            node="act",
            state={
                "untrusted": {"sender": "dev@example.com", "subject": "Refund", "body": "please"},
                "agent": {"failure": f"act: the ledger refused the refund: {XSS}"},
            },
            key="email_msg_refused",
        )

    page = client_for(fresh_database).get("/approvals").text

    assert "the ledger refused the refund" in page
    assert "dev@example.com" in page
    assert XSS not in page


def test_the_guardrails_page_shows_the_limits_in_force(fresh_database):
    page = client_for(fresh_database).get("/guardrails")

    assert page.status_code == 200
    assert "Rs 5,000" in page.text
    assert "0.85" in page.text


# --- deciding -------------------------------------------------------------------------------


def test_approving_sends_the_run_back_to_the_queue(fresh_database):
    run_id, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    token = token_from(client.get("/approvals").text)

    response = client.post(
        f"/approvals/{approval_id}/approve", data={"csrf": token, "note": "both charges on the ledger"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/approvals"
    assert run_status(fresh_database, run_id) == ("queued", None)
    assert decision(fresh_database, approval_id) == ("approved", "asha", "both charges on the ledger")


def test_rejecting_finishes_the_run(fresh_database):
    run_id, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    token = token_from(client.get("/approvals").text)

    response = client.post(f"/approvals/{approval_id}/reject", data={"csrf": token}, follow_redirects=False)

    assert response.status_code == 303
    assert run_status(fresh_database, run_id) == ("done", "rejected")
    assert decision(fresh_database, approval_id) == ("rejected", "asha", None)


def test_a_second_decision_is_refused_with_a_message(fresh_database):
    run_id, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    token = token_from(client.get("/approvals").text)
    client.post(f"/approvals/{approval_id}/approve", data={"csrf": token}, follow_redirects=False)

    again = client.post(f"/approvals/{approval_id}/reject", data={"csrf": token}, follow_redirects=False)

    assert again.status_code == 409
    assert "already decided" in again.text
    assert run_status(fresh_database, run_id) == ("queued", None)


def test_an_unknown_decision_is_not_found(fresh_database):
    _, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    token = token_from(client.get("/approvals").text)

    response = client.post(f"/approvals/{approval_id}/maybe", data={"csrf": token}, follow_redirects=False)

    assert response.status_code == 404


def test_an_overlong_note_is_refused_and_nothing_is_decided(fresh_database):
    _, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    token = token_from(client.get("/approvals").text)

    response = client.post(
        f"/approvals/{approval_id}/approve", data={"csrf": token, "note": "x" * 1001}, follow_redirects=False
    )

    assert response.status_code == 400
    assert decision(fresh_database, approval_id)[0] == "pending"


def test_deciding_needs_to_know_who_is_deciding(fresh_database):
    _, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database, operator=None)
    token = token_from(client.get("/approvals").text)

    response = client.post(f"/approvals/{approval_id}/approve", data={"csrf": token}, follow_redirects=False)

    assert response.status_code == 403
    assert "OPSAGENT_OPERATOR" in response.text
    assert decision(fresh_database, approval_id)[0] == "pending"


# --- what another site cannot do --------------------------------------------------------------


@pytest.mark.parametrize("form", [{}, {"csrf": "guessed"}], ids=["no-token", "wrong-token"])
def test_a_decision_without_the_pages_token_is_refused(fresh_database, form):
    _, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    client.get("/approvals")

    response = client.post(f"/approvals/{approval_id}/approve", data=form, follow_redirects=False)

    assert response.status_code == 403
    assert decision(fresh_database, approval_id)[0] == "pending"


def test_a_decision_without_the_cookie_is_refused(fresh_database):
    """A form posted from another site carries no SameSite=Strict cookie, even if it somehow had the token."""
    _, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    token = token_from(client.get("/approvals").text)
    client.cookies.clear()

    response = client.post(f"/approvals/{approval_id}/approve", data={"csrf": token}, follow_redirects=False)

    assert response.status_code == 403
    assert decision(fresh_database, approval_id)[0] == "pending"


def test_the_token_cookie_is_strict_and_out_of_reach_of_scripts(fresh_database):
    cookie = client_for(fresh_database).get("/approvals").headers["set-cookie"].lower()

    assert "samesite=strict" in cookie
    assert "httponly" in cookie


def test_each_screen_has_its_own_token(fresh_database):
    """A token taken from one run of the screen is worthless against another."""
    waiting_refund(fresh_database)
    first = token_from(client_for(fresh_database).get("/approvals").text)
    second = token_from(client_for(fresh_database).get("/approvals").text)

    assert first != second


def test_a_request_for_another_host_is_refused(fresh_database):
    """DNS rebinding: a page on evil.example resolving to 127.0.0.1 still sends its own Host."""
    response = client_for(fresh_database).get("/approvals", headers={"host": "evil.example"})

    assert response.status_code == 400


def test_pages_allow_no_scripts_and_no_framing(fresh_database):
    headers = client_for(fresh_database).get("/approvals").headers

    policy = headers["content-security-policy"]
    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "form-action 'self'" in policy
    assert headers["x-content-type-options"] == "nosniff"
    # Found in the real gate run: under "no-referrer" a browser sends "Origin: null" on the screen's own
    # form posts, and the Origin check refused every decision. "same-origin" still sends no referrer
    # to other sites, and a cross-site post still arrives as "Origin: null".
    assert headers["referrer-policy"] == "same-origin"
    # Security review: customer emails must not linger in a browser cache; framing refused for old browsers too.
    assert headers["cache-control"] == "no-store"
    assert headers["x-frame-options"] == "DENY"


# --- what the web process is not allowed to be -----------------------------------------------


def test_the_web_process_never_loads_the_executor_or_the_worker():
    """Carried from Unit B's security review: approvals go through the worker, never through here."""
    loaded = subprocess.run(
        [sys.executable, "-c", "import sys, app.web; print(sorted({'app.executor', 'app.run_agent'} & set(sys.modules)))"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()

    assert loaded == "[]"


def test_no_template_turns_escaping_off():
    templates = sorted((ROOT / "app" / "templates").glob("*.html"))

    assert templates, "no templates found"
    for template in templates:
        text = template.read_text()
        for switch in ("|safe", "| safe", "autoescape false", "Markup("):
            assert switch not in text, f"{template.name} uses {switch}"


def test_the_screen_listens_only_on_this_machine():
    assert HOST == "127.0.0.1"


# --- found in review ------------------------------------------------------------------------


def test_a_decision_posted_from_another_origin_is_refused(fresh_database):
    """Security review: SameSite=Strict already stops this; checking Origin as well costs one comparison."""
    _, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    token = token_from(client.get("/approvals").text)

    response = client.post(
        f"/approvals/{approval_id}/approve",
        data={"csrf": token},
        headers={"origin": "http://evil.example"},
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert decision(fresh_database, approval_id)[0] == "pending"


def test_a_decision_whose_origin_the_browser_withheld_is_refused(fresh_database):
    """A cross-site form post under the screen's referrer policy arrives as "Origin: null"."""
    _, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    token = token_from(client.get("/approvals").text)

    response = client.post(
        f"/approvals/{approval_id}/approve", data={"csrf": token}, headers={"origin": "null"}, follow_redirects=False
    )

    assert response.status_code == 403
    assert decision(fresh_database, approval_id)[0] == "pending"


def test_a_decision_posted_with_this_screens_own_origin_is_accepted(fresh_database):
    """Browsers send Origin on every form POST; the check must not refuse the screen itself."""
    _, approval_id = waiting_refund(fresh_database)
    client = client_for(fresh_database)
    token = token_from(client.get("/approvals").text)

    response = client.post(
        f"/approvals/{approval_id}/approve", data={"csrf": token}, headers={"origin": LOCAL}, follow_redirects=False
    )

    assert response.status_code == 303
    assert decision(fresh_database, approval_id)[0] == "approved"


def test_a_page_that_fails_keeps_its_protections_and_gives_nothing_away():
    """FastAPI review: an unhandled error -- here an unreachable database -- used to drop every security header."""
    unreachable = "postgresql://opsagent:not-a-real-secret@127.0.0.1:1/opsagent?connect_timeout=1"
    client = TestClient(create_app(dsn=unreachable, operator="asha"), base_url=LOCAL, raise_server_exceptions=False)

    response = client.get("/approvals")

    assert response.status_code == 500
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"
    assert "not-a-real-secret" not in response.text
    assert "127.0.0.1:1" not in response.text


def test_a_missing_detail_reads_as_missing_not_as_none(fresh_database):
    """FastAPI review: Jinja prints Python's None as the word "None" on the decision screen."""
    with psycopg.connect(fresh_database) as connection:
        run_id = insert_run(connection, status="waiting_approval", node="approval", state={}, key="email_msg_sparse")
        refund = ProposedAction(
            tool="issue_refund",
            args={"order_id": "4821", "amount_paise": 720_000, "reason": "charged twice"},
            confidence=Decimal("0.9"),
            reasoning="duplicate",
        )
        open_approval(connection, run_id, refund, {}, OVER_THE_LIMIT)
        insert_run(connection, status="waiting_approval", node="act", state={}, key="email_msg_bare")

    page = client_for(fresh_database).get("/approvals").text

    assert "None" not in page
    assert "not recorded" in page


def test_the_guardrails_page_shows_what_one_run_may_spend(fresh_database):
    """An operator changing a ceiling has to be able to see the one in force first."""
    page = client_for(fresh_database).get("/guardrails").text

    assert "10,000" in page and "token" in page.lower()
    assert "0.002" in page
    assert "180" in page and "second" in page.lower()


def test_a_ceiling_switched_off_says_so_rather_than_showing_zero(fresh_database):
    """Zero means no ceiling here, and "0 tokens" reads like the harshest setting there is."""
    with psycopg.connect(fresh_database) as connection:
        set_limits(connection, max_tokens_per_run=0, by="an operator")
        connection.commit()

    page = client_for(fresh_database).get("/guardrails").text

    assert "no ceiling" in page.lower()


def test_the_front_door_lands_on_the_home_page(fresh_database):
    """
    Opening the screen at its root used to answer `{"detail":"Not Found"}`.

    Every page is on a named path, so a person who types the host and nothing else -- which is
    what a person does -- got a JSON 404 from a tool whose whole job is to be opened in a hurry.
    Now it lands on a home page that says what this is, before any guessing.
    """
    landing = client_for(fresh_database).get("/", follow_redirects=False)

    assert landing.status_code == 200
    assert "How a run flows" in landing.text
    assert "Approvals" in landing.text and "Runs" in landing.text
