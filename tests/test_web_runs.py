"""
The runs pages: pick a run, then see everything it did.

/runs lists the newest runs, each linking to its own page. /runs/{id} shows the run's
trace as rows, each at the depth it was recorded -- tick, step, model call -- with how
long it took, the model and prompt version, tokens in and out, its reference cost and
what it did; then the calls' total beside the cost recorded on the run; then the
approvals the run waited on.

Everything on these pages came from a customer's email or a model reading one, so all
of it is escaped. Neither page shows locked_by or the run's raw state, and both carry
the same headers as the approvals screen: no scripts, no framing, no caching. A link to
the same trace in Langfuse appears only when one is configured, and only on this machine.
"""

import re
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.graph.build import build_graph
from app.graph.prompts import PROMPT_VERSIONS
from app.llm import DEFAULT_MODEL, ModelUnavailable
from app.run_agent import work_next
from app.web import SECURITY_HEADERS, create_app, langfuse_link_base, span_view
from tests.fakes import CLASSIFIED_DUPLICATE, OUTAGE, FakeRetriever, ScriptedModel
from tests.test_approval_path import decide_on, refund_model, work
from tests.test_retrieval import loaded, retriever
from tests.test_run_agent import happy_model, ledger, queue
from tests.test_traces import insert_run, span
from tests.test_web import LOCAL, XSS, client_for

pytestmark = pytest.mark.db

LANGFUSE = "http://127.0.0.1:3000/project/opsagent-local"
ROW = re.compile(r'<tr class="span[^"]*" data-depth="(\d+)">\s*<td class="name">([^<]+)</td>')


def rows(html: str) -> list[tuple[int, str]]:
    """The trace table as (depth, span name), top to bottom."""
    return [(int(depth), name.strip()) for depth, name in ROW.findall(html)]


def worked_run(dsn: str) -> str:
    ledger(dsn)
    run_id = queue(dsn)
    work(dsn, happy_model())
    return run_id


# --- the list -----------------------------------------------------------------------------


def test_the_runs_page_links_to_each_run(fresh_database, exported):
    run_id = worked_run(fresh_database)

    page = client_for(fresh_database).get("/runs")

    assert page.status_code == 200
    assert f'href="/runs/{run_id}"' in page.text
    assert "Charged twice for order #4821" in page.text
    assert "priya@example.com" in page.text
    assert "done" in page.text


def test_every_screen_links_to_the_runs_page(fresh_database):
    page = client_for(fresh_database).get("/approvals")

    assert 'href="/runs"' in page.text


# --- one run ------------------------------------------------------------------------------


def test_a_run_page_shows_every_step_nested_under_its_tick(fresh_database, exported):
    run_id = worked_run(fresh_database)

    page = client_for(fresh_database).get(f"/runs/{run_id}")

    assert page.status_code == 200
    assert rows(page.text) == [
        (0, "tick"),
        (1, "classify"),
        (2, "classify.generate"),
        (1, "extract"),
        (2, "extract.generate"),
        (1, "retrieve"),
        (1, "plan"),
        (2, "plan.generate"),
        (1, "act"),
        (0, "tick"),
        (1, "plan"),
        (2, "plan.generate"),
        (1, "act"),
        (2, "guardrail"),
    ]


def test_each_model_call_shows_its_model_version_tokens_and_cost(fresh_database, exported):
    run_id = worked_run(fresh_database)

    html = client_for(fresh_database).get(f"/runs/{run_id}").text

    assert DEFAULT_MODEL in html, "the model that answered, which a stand-in stands in for"
    assert PROMPT_VERSIONS["classify"] in html
    assert PROMPT_VERSIONS["plan"] in html
    assert "$0.0000045" in html, "each call: 10 tokens in, 5 out, at the reference rate"
    assert re.search(r"<td class=\"tokens\">10 in, 5 out</td>", html)


def test_the_calls_total_is_shown_beside_the_cost_recorded_on_the_run(fresh_database, exported):
    run_id = worked_run(fresh_database)

    html = client_for(fresh_database).get(f"/runs/{run_id}").text

    assert re.search(r"Model calls, added up.*?\$0\.000018", html, re.DOTALL)
    assert re.search(r"Recorded on the run.*?\$0\.000018", html, re.DOTALL)
    assert "reference cost" in html.lower()


def test_what_each_step_did_is_shown(fresh_database, exported):
    run_id = worked_run(fresh_database)

    html = client_for(fresh_database).get(f"/runs/{run_id}").text

    for shown in ("looked up", "refunded", "duplicate_charge", "get_order", "issue_refund", "runs"):
        assert shown in html, shown


def test_an_embedding_says_its_cost_is_not_counted(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    graph = build_graph(happy_model(), retriever(fresh_database, loaded(fresh_database)))
    with psycopg.connect(fresh_database) as connection:
        work_next(connection, graph)

    html = client_for(fresh_database).get(f"/runs/{run_id}").text

    assert re.search(r'<td class="name">embed_query</td>.*?<td class="cost">not counted</td>', html, re.DOTALL)


def test_a_step_that_failed_is_marked_with_why(fresh_database, exported):
    run_id = queue(fresh_database)
    with psycopg.connect(fresh_database) as connection, pytest.raises(ModelUnavailable):
        work_next(connection, build_graph(ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=OUTAGE), FakeRetriever()))

    html = client_for(fresh_database).get(f"/runs/{run_id}").text

    assert re.search(r'<tr class="span error" data-depth="1">\s*<td class="name">extract</td>', html)
    assert "model_unavailable" in html


def test_the_approvals_a_run_waited_on_are_listed_with_who_decided(fresh_database, exported):
    ledger(fresh_database)
    run_id = queue(fresh_database)
    work(fresh_database, refund_model(720_000))
    decide_on(fresh_database, run_id, approved=True)

    html = client_for(fresh_database).get(f"/runs/{run_id}").text

    assert '<p class="reason">Rs 7,200 is not under the Rs 5,000 limit' in html
    assert "approved" in html
    assert "asha" in html


def test_only_a_model_call_shows_a_cost_and_only_whole_counts_show_tokens():
    step = span_view(span("classify", "tick", 1, 20), 1)
    half_counted = span("classify.generate", "classify", 2, 9, kind="generation", cost="0.0000045")
    half_counted.output_tokens = None

    assert (step["cost"], step["tokens"]) == ("", "")
    assert span_view(half_counted, 2)["tokens"] == ""


def test_a_run_not_worked_yet_says_so(fresh_database):
    run_id = queue(fresh_database)

    html = client_for(fresh_database).get(f"/runs/{run_id}").text

    assert "Nothing has been traced for this run yet." in html


# --- what the pages must never do -----------------------------------------------------------


def test_customer_text_is_escaped_on_both_pages(fresh_database):
    run_id = insert_run(fresh_database, minute=0, sender=f"{XSS}@example.com", subject=XSS, steps=1)
    client = client_for(fresh_database)

    for path in ("/runs", f"/runs/{run_id}"):
        html = client.get(path).text
        assert "<script>alert" not in html, path
        assert "&lt;script&gt;" in html, path


def test_neither_page_shows_the_worker(fresh_database):
    run_id = insert_run(fresh_database, minute=0, sender="priya@example.com", subject="Charged twice", steps=1)
    client = client_for(fresh_database)

    for path in ("/runs", f"/runs/{run_id}"):
        assert "worker-on-host" not in client.get(path).text, path


def test_an_unknown_run_is_not_found(fresh_database):
    page = client_for(fresh_database).get(f"/runs/{uuid4()}")

    assert page.status_code == 404
    assert "No such run" in page.text


def test_a_malformed_run_id_is_not_found_without_touching_the_database():
    unreachable = "postgresql://nobody:nothing@127.0.0.1:1/none"
    client = TestClient(create_app(dsn=unreachable, operator="asha"), base_url=LOCAL)

    for path in ("/runs/not-a-run", "/runs/1%27%20OR%201=1", f"/runs/{uuid4().hex}x"):
        page = client.get(path)
        assert page.status_code == 404, path
        assert "No such run" in page.text, path


def test_the_runs_pages_carry_the_screens_security_headers(fresh_database):
    run_id = queue(fresh_database)
    client = client_for(fresh_database)

    for path in ("/runs", f"/runs/{run_id}"):
        headers = client.get(path).headers
        for name, value in SECURITY_HEADERS.items():
            assert headers[name] == value, (path, name)


# --- the same trace in Langfuse ---------------------------------------------------------------


def test_a_link_to_the_trace_in_langfuse_appears_when_one_is_configured(fresh_database, exported):
    run_id = worked_run(fresh_database)
    client = TestClient(create_app(dsn=fresh_database, operator="asha", langfuse_project_url=LANGFUSE), base_url=LOCAL)

    html = client.get(f"/runs/{run_id}").text

    assert f'href="{LANGFUSE}/traces/{UUID(run_id).hex}"' in html


def test_no_langfuse_link_without_one_configured(fresh_database, exported):
    run_id = worked_run(fresh_database)

    html = client_for(fresh_database).get(f"/runs/{run_id}").text

    assert "/traces/" not in html
    assert "in Langfuse" not in html


@pytest.mark.parametrize("url", [None, "", "   "])
def test_a_blank_langfuse_setting_means_no_link(url):
    assert langfuse_link_base(url) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://cloud.langfuse.com/project/x",
        "http://10.0.0.5:3000/project/x",
        "javascript:alert(1)",
        "http://127.0.0.1.evil.example/project/x",
        "https://127.0.0.1:3000/project/x",
    ],
)
def test_a_langfuse_link_off_this_machine_is_refused(url):
    with pytest.raises(ValueError, match="this machine"):
        create_app(dsn="postgresql://unused", operator="asha", langfuse_project_url=url)
