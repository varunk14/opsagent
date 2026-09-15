"""
Tracing: one OpenTelemetry trace per run, kept in our own table and copied to Langfuse.

A run's trace id is the run's own UUID, so every tick, every worker, a resume after a
crash and an approval days later all land in one trace. Finished spans are buffered in
memory and written to the spans table inside the transaction that commits the step, so
a tick that never committed leaves no spans, exactly as it leaves no steps. The copy
to Langfuse goes over OTLP to a loopback address only, and a Langfuse that is down or
slow never raises into a tick or holds it up.

The resource sent with every span names the service and nothing about this machine:
the worker's host name and pid are what `locked_by` holds, and they stay out of spans
for the same reason they stay off the approvals screen.
"""

import time
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from psycopg.types.json import Jsonb

from app.tracing import (
    SERVICE_NAME,
    Attr,
    Tracing,
    discard,
    exporter_from_env,
    loopback_exporter,
    record_spans,
    rows_for,
    run_context,
    tracer,
)

RUN = UUID("a2a97b37-62fb-4a4d-ab07-60172aa05ef9")
OTHER_RUN = UUID("812d8a9e-cd5b-41b5-bf36-618f2155bfb7")
KEYS = {"public_key": "pk-lf-test", "secret_key": "sk-lf-test"}


def finished(exported: InMemorySpanExporter) -> dict[str, object]:
    return {span.name: span for span in exported.get_finished_spans()}


# --- one trace per run ------------------------------------------------------------


def test_a_span_inside_a_run_takes_the_run_id_as_its_trace_id(exported):
    with run_context(RUN), tracer().start_as_current_span("tick"), tracer().start_as_current_span("classify"):
        pass

    spans = finished(exported)
    assert spans["tick"].context.trace_id == RUN.int
    assert spans["classify"].context.trace_id == RUN.int
    assert spans["classify"].parent.span_id == spans["tick"].context.span_id


def test_two_ticks_of_one_run_in_separate_contexts_share_the_trace(exported):
    for _ in range(2):
        with run_context(RUN), tracer().start_as_current_span("tick"):
            pass

    first, second = exported.get_finished_spans()
    assert first.context.trace_id == second.context.trace_id == RUN.int
    assert first.context.span_id != second.context.span_id


def test_two_runs_in_one_process_never_share_a_trace(exported):
    with run_context(RUN), tracer().start_as_current_span("tick"):
        pass
    with run_context(OTHER_RUN), tracer().start_as_current_span("tick"):
        pass

    first, second = exported.get_finished_spans()
    assert (first.context.trace_id, second.context.trace_id) == (RUN.int, OTHER_RUN.int)


def test_a_span_outside_any_run_gets_a_trace_id_of_its_own(exported):
    with tracer().start_as_current_span("housekeeping"):
        pass
    with tracer().start_as_current_span("housekeeping"):
        pass

    first, second = exported.get_finished_spans()
    assert first.context.trace_id not in (0, second.context.trace_id)


# --- nothing about this machine leaves it ---------------------------------------


def test_the_resource_names_the_service_and_nothing_else(monkeypatch):
    """Even with the standard environment variable set, no host name or pid travels with a span."""
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "host.name=leaky-mac,process.pid=4242")
    exported = InMemorySpanExporter()
    own = Tracing(exported, immediate=True)

    with own.tracer.start_as_current_span("tick"):
        pass

    (span,) = exported.get_finished_spans()
    assert dict(span.resource.attributes) == {"service.name": SERVICE_NAME}
    own.shutdown()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/api/public/otel/v1/traces",
        "http://10.0.0.5:3000/api/public/otel/v1/traces",
        "http://langfuse:3000/api/public/otel/v1/traces",
        "https://cloud.langfuse.com/api/public/otel/v1/traces",
        "http://127.0.0.1.evil.example/api/public/otel/v1/traces",
        "ftp://127.0.0.1/api/public/otel/v1/traces",
    ],
)
def test_an_endpoint_off_this_machine_is_refused(endpoint):
    with pytest.raises(ValueError, match="this machine"):
        loopback_exporter(endpoint, **KEYS)


@pytest.mark.parametrize(
    "endpoint",
    ["http://127.0.0.1:3000/api/public/otel/v1/traces", "http://localhost:3000/api/public/otel/v1/traces"],
)
def test_a_loopback_endpoint_is_accepted(endpoint):
    assert loopback_exporter(endpoint, **KEYS) is not None


def test_nothing_is_exported_unless_the_endpoint_is_set():
    assert exporter_from_env({}) is None


def test_an_endpoint_without_the_langfuse_keys_is_refused():
    with pytest.raises(ValueError, match="OPSAGENT_LANGFUSE_SECRET_KEY"):
        exporter_from_env(
            {"OPSAGENT_OTLP_ENDPOINT": "http://127.0.0.1:3000/api/public/otel", "OPSAGENT_LANGFUSE_PUBLIC_KEY": "pk"}
        )


def test_the_environment_builds_a_loopback_exporter():
    environ = {
        "OPSAGENT_OTLP_ENDPOINT": "http://127.0.0.1:3000/api/public/otel",
        "OPSAGENT_LANGFUSE_PUBLIC_KEY": "pk-lf-x",
        "OPSAGENT_LANGFUSE_SECRET_KEY": "sk-lf-y",
    }

    assert exporter_from_env(environ) is not None


def test_a_langfuse_that_cannot_be_reached_neither_raises_nor_stalls():
    """Port 9 on loopback has nothing listening. Spans still finish at once, and shutdown is bounded."""
    own = Tracing(loopback_exporter("http://127.0.0.1:9/api/public/otel/v1/traces", **KEYS))

    started = time.perf_counter()
    for _ in range(50):
        with own.tracer.start_as_current_span("tick"):
            pass
    assert time.perf_counter() - started < 1.0

    started = time.perf_counter()
    own.shutdown()
    assert time.perf_counter() - started < 15.0


def test_installing_a_second_tracing_in_one_process_is_refused(tracing):
    """OpenTelemetry keeps the first global provider; a silent second one would trace nothing."""
    second = Tracing(InMemorySpanExporter(), immediate=True)

    with pytest.raises(RuntimeError, match="already installed"):
        second.install()
    second.shutdown()


def test_the_global_tracer_provider_is_the_installed_one(tracing):
    assert trace.get_tracer_provider() is tracing.provider


# --- what a span becomes as a row ----------------------------------------------


def test_rows_are_ordered_parents_first_by_start_time(exported):
    with run_context(RUN), tracer().start_as_current_span("tick"):
        with tracer().start_as_current_span("classify"), tracer().start_as_current_span("generate"):
            pass
        with tracer().start_as_current_span("act"):
            pass

    rows = rows_for(exported.get_finished_spans())

    assert [row.name for row in rows] == ["tick", "classify", "generate", "act"]
    by_name = {row.name: row for row in rows}
    assert by_name["classify"].parent_span_id == by_name["tick"].span_id
    assert by_name["generate"].parent_span_id == by_name["classify"].span_id
    assert by_name["tick"].parent_span_id is None
    assert all(row.trace_id == RUN for row in rows)
    assert all(len(row.span_id) == 16 for row in rows)
    assert by_name["generate"].started_at >= by_name["classify"].started_at
    assert by_name["generate"].ended_at <= by_name["classify"].ended_at


def test_a_generation_row_carries_model_tokens_cost_and_version(exported):
    with run_context(RUN), tracer().start_as_current_span("generate") as span:
        span.set_attribute(Attr.TYPE, "generation")
        span.set_attribute(Attr.MODEL, "llama3.1:8b")
        span.set_attribute(Attr.INPUT_TOKENS, 310)
        span.set_attribute(Attr.OUTPUT_TOKENS, 40)
        span.set_attribute(Attr.COST_USD, "0.0000705")
        span.set_attribute(Attr.PROMPT_VERSION, "4c0e5dd7b3a9")
        span.set_attribute(Attr.LATENCY_MS, 412)

    (row,) = rows_for(exported.get_finished_spans())

    assert (row.kind, row.model, row.input_tokens, row.output_tokens) == ("generation", "llama3.1:8b", 310, 40)
    assert row.cost_usd == Decimal("0.0000705")
    assert row.prompt_version == "4c0e5dd7b3a9"
    assert row.status == "ok"
    assert row.attributes[Attr.LATENCY_MS] == 412


def test_a_span_with_no_type_is_a_plain_span_with_nothing_counted(exported):
    with run_context(RUN), tracer().start_as_current_span("tick"):
        pass

    (row,) = rows_for(exported.get_finished_spans())

    assert (row.kind, row.model, row.input_tokens, row.output_tokens, row.cost_usd, row.prompt_version) == (
        "span",
        None,
        None,
        None,
        None,
        None,
    )


def test_an_error_span_keeps_its_status_and_message(exported):
    with run_context(RUN), tracer().start_as_current_span("classify") as span:
        span.set_status(StatusCode.ERROR, "model_unavailable")

    (row,) = rows_for(exported.get_finished_spans())

    assert (row.status, row.status_message) == ("error", "model_unavailable")


# --- written with the step, or not at all ----------------------------------------


def insert_run(connection, run_id: UUID) -> None:
    connection.execute(
        "INSERT INTO runs (id, channel, status, current_node, state, idempotency_key) "
        "VALUES (%s, 'email', 'running', 'classify', %s, %s)",
        (run_id, Jsonb({}), f"email_msg_{run_id.hex}"),
    )


def stored(connection, run_id: UUID) -> list[tuple]:
    # A parent and its first child can start in the same microsecond; the root is listed first.
    return connection.execute(
        "SELECT name, parent_span_id IS NULL FROM spans WHERE trace_id = %s "
        "ORDER BY started_at, parent_span_id IS NOT NULL, span_id",
        (run_id,),
    ).fetchall()


@pytest.mark.db
def test_record_spans_writes_the_buffer_in_the_callers_transaction(db, exported):
    run_id = uuid4()
    insert_run(db, run_id)
    with run_context(run_id), tracer().start_as_current_span("tick"), tracer().start_as_current_span("classify"):
        pass

    written = record_spans(db, run_id)

    assert written == 2
    assert stored(db, run_id) == [("tick", True), ("classify", False)]
    assert record_spans(db, run_id) == 0  # the buffer was taken, not copied


@pytest.mark.db
def test_another_runs_spans_are_not_written_with_this_one(db, exported):
    mine, theirs = uuid4(), uuid4()
    insert_run(db, mine)
    insert_run(db, theirs)
    with run_context(theirs), tracer().start_as_current_span("tick"):
        pass
    with run_context(mine), tracer().start_as_current_span("tick"):
        pass

    assert record_spans(db, mine) == 1
    assert stored(db, theirs) == []
    assert record_spans(db, theirs) == 1


@pytest.mark.db
def test_discard_drops_what_a_tick_that_never_committed_buffered(db, exported):
    """A worker that lost its claim, or raised, must not write those spans with its next tick."""
    run_id = uuid4()
    insert_run(db, run_id)
    with run_context(run_id), tracer().start_as_current_span("tick"):
        pass

    discard(run_id)
    with run_context(run_id), tracer().start_as_current_span("tick"):
        pass

    assert record_spans(db, run_id) == 1


@pytest.mark.db
def test_a_span_that_ends_after_the_write_waits_for_the_next_one(db, exported):
    """Only finished spans are written; one still open goes with the next write."""
    run_id = uuid4()
    insert_run(db, run_id)
    with run_context(run_id), tracer().start_as_current_span("tick"):
        with tracer().start_as_current_span("classify"):
            pass
        assert record_spans(db, run_id) == 1

    assert record_spans(db, run_id) == 1
    assert [name for name, _ in stored(db, run_id)] == ["tick", "classify"]


@pytest.mark.db
def test_the_exported_copy_and_the_recorded_rows_are_the_same_spans(db, exported):
    run_id = uuid4()
    insert_run(db, run_id)
    with run_context(run_id), tracer().start_as_current_span("tick"), tracer().start_as_current_span("plan"):
        pass

    record_spans(db, run_id)

    recorded = {span_id for (span_id,) in db.execute("SELECT span_id FROM spans WHERE trace_id = %s", (run_id,))}
    assert recorded == {format(span.context.span_id, "016x") for span in exported.get_finished_spans()}


# --- found in review: which run a span belongs to is settled when it starts -----------


def test_a_span_started_in_a_run_is_kept_for_it_even_when_it_ends_outside(tracing, exported):
    """Its trace id was fixed when it started; ending after the run's context closed must not lose it."""
    with run_context(RUN):
        late = tracer().start_span("plan.generate")
    late.end()

    assert [span.name for span in tracing.recorder.take(RUN)] == ["plan.generate"]


def test_a_span_that_ends_inside_another_runs_context_stays_with_its_own_run(tracing, exported):
    with run_context(RUN):
        span = tracer().start_span("tick")
    with run_context(OTHER_RUN):
        span.end()

    assert [held.name for held in tracing.recorder.take(RUN)] == ["tick"]
    assert tracing.recorder.take(OTHER_RUN) == []


# --- found in review: nothing about a span may abort the step it is written with -----

WELL_FORMED = {
    Attr.TYPE: "generation",
    Attr.MODEL: "llama3.1:8b",
    Attr.INPUT_TOKENS: 310,
    Attr.OUTPUT_TOKENS: 40,
    Attr.COST_USD: "0.0000705",
    Attr.PROMPT_VERSION: "4c0e5dd7b3a9",
}
COLUMN_OF = {
    Attr.INPUT_TOKENS: "input_tokens",
    Attr.OUTPUT_TOKENS: "output_tokens",
    Attr.COST_USD: "cost_usd",
    Attr.PROMPT_VERSION: "prompt_version",
}


def stored_row(connection, run_id: UUID) -> dict:
    cursor = connection.execute("SELECT * FROM spans WHERE trace_id = %s", (run_id,))
    names = [column.name for column in cursor.description]
    (values,) = cursor.fetchall()
    return dict(zip(names, values, strict=True))


@pytest.mark.db
def test_a_generation_missing_its_cost_is_written_as_a_plain_span_not_refused(db, exported):
    """
    The table refuses a generation with no cost, and record_spans runs in the transaction
    that may be paying a refund. So such a span is written as a plain span that says
    why, rather than taking the payment down with it.
    """
    run_id = uuid4()
    insert_run(db, run_id)
    with run_context(run_id), tracer().start_as_current_span("generate") as span:
        span.set_attribute(Attr.TYPE, "generation")
        span.set_attribute(Attr.MODEL, "llama3.1:8b")

    assert record_spans(db, run_id) == 1

    stored = stored_row(db, run_id)
    assert stored["kind"] == "span"
    assert stored["attributes"][Attr.RECORDED_AS_SPAN] == "a generation needs a model, token counts and a cost"


@pytest.mark.db
@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        (Attr.INPUT_TOKENS, "many"),
        (Attr.INPUT_TOKENS, -3),
        (Attr.OUTPUT_TOKENS, 12.7),
        (Attr.OUTPUT_TOKENS, True),
        (Attr.COST_USD, "free"),
        (Attr.COST_USD, "-0.01"),
        (Attr.COST_USD, "NaN"),
        (Attr.COST_USD, "Infinity"),
        (Attr.PROMPT_VERSION, "v3"),
    ],
)
def test_a_malformed_count_cost_or_version_is_left_out_never_refused(db, exported, attribute, value):
    run_id = uuid4()
    insert_run(db, run_id)
    with run_context(run_id), tracer().start_as_current_span("generate") as span:
        span.set_attributes({**WELL_FORMED, attribute: value})

    assert record_spans(db, run_id) == 1

    stored = stored_row(db, run_id)
    assert stored[COLUMN_OF[attribute]] is None
    assert stored["attributes"][attribute] == value  # kept as it was set, for whoever investigates
    # A version is optional; a generation without its counts or cost is no longer a generation.
    assert stored["kind"] == ("generation" if attribute == Attr.PROMPT_VERSION else "span")


@pytest.mark.db
def test_a_span_ended_before_it_started_is_written_with_no_duration(db, exported):
    run_id = uuid4()
    insert_run(db, run_id)
    instant = 1_757_937_600_000_000_000
    with run_context(run_id):
        span = tracer().start_span("tick", start_time=instant)
        span.end(end_time=instant - 5_000_000)

    assert record_spans(db, run_id) == 1

    stored = stored_row(db, run_id)
    assert stored["ended_at"] == stored["started_at"]


@pytest.mark.db
def test_a_span_with_a_blank_name_is_written_under_a_placeholder(db, exported):
    run_id = uuid4()
    insert_run(db, run_id)
    with run_context(run_id):
        tracer().start_span("   ").end()

    assert record_spans(db, run_id) == 1

    assert stored_row(db, run_id)["name"] == "unnamed"


@pytest.mark.db
def test_every_field_is_stored_in_the_column_of_the_same_name(db, exported):
    """Two literals list the columns -- the row and the INSERT; a swap between them must fail here."""
    from dataclasses import asdict

    run_id = uuid4()
    insert_run(db, run_id)
    with run_context(run_id), tracer().start_as_current_span("plan.generate") as span:
        span.set_attributes(WELL_FORMED)
        span.set_status(StatusCode.ERROR, "model_unavailable")
    (expected,) = rows_for(exported.get_finished_spans())

    record_spans(db, run_id)

    assert stored_row(db, run_id) == asdict(expected)


# --- found in security review: loopback means where the connection goes, not the URL ---


class Listener:
    """An HTTP server on a free loopback port that records every request reaching it, or answers with a redirect."""

    def __init__(self, redirect_to: str | None = None):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        hits: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                hits.append(self.path)
                self.send_response(200 if redirect_to is None else 307)
                if redirect_to is not None:
                    self.send_header("Location", redirect_to)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.hits = hits
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def export_one_span(endpoint: str) -> None:
    own = Tracing(loopback_exporter(endpoint, **KEYS), immediate=True)
    with own.tracer.start_as_current_span("tick"):
        pass
    own.shutdown()


def test_a_proxy_set_in_the_environment_is_never_used_for_traces(monkeypatch):
    """requests honours HTTP_PROXY by default, which would carry spans addressed to 127.0.0.1 to the proxy's host."""
    with Listener() as langfuse, Listener() as proxy:
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, proxy.url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)

        export_one_span(f"{langfuse.url}/api/public/otel/v1/traces")

    assert proxy.hits == []
    assert langfuse.hits == ["/api/public/otel/v1/traces"]


def test_a_redirect_from_the_endpoint_is_not_followed():
    """A 307 would have the whole batch of spans sent again, to wherever it points."""
    with Listener() as elsewhere, Listener(redirect_to=f"{elsewhere.url}/stolen") as endpoint:
        export_one_span(f"{endpoint.url}/api/public/otel/v1/traces")

    assert endpoint.hits == ["/api/public/otel/v1/traces"]
    assert elsewhere.hits == []


def test_one_run_holds_a_bounded_number_of_spans(tracing, exported, monkeypatch):
    """A retry storm inside one step must not grow the worker's memory without limit."""
    from app import tracing as module

    monkeypatch.setattr(module, "MAX_HELD_SPANS_PER_RUN", 5)
    with run_context(RUN):
        for number in range(8):
            tracer().start_span(f"call {number}").end()

    assert [span.name for span in tracing.recorder.take(RUN)] == [f"call {number}" for number in range(5)]


@pytest.mark.db
def test_spans_past_the_bound_are_reported_when_the_rest_are_written(db, exported, monkeypatch, capsys):
    from app import tracing as module

    monkeypatch.setattr(module, "MAX_HELD_SPANS_PER_RUN", 2)
    run_id = uuid4()
    insert_run(db, run_id)
    with run_context(run_id):
        for number in range(5):
            tracer().start_span(f"call {number}").end()

    assert record_spans(db, run_id) == 2
    assert f"3 spans of run {run_id} were not recorded" in capsys.readouterr().err
    with run_context(run_id):
        tracer().start_span("next step").end()
    assert record_spans(db, run_id) == 1
    assert "not recorded" not in capsys.readouterr().err  # the count was cleared with what it counted


# --- found by mutation checks ------------------------------------------------------


def test_a_span_from_another_trace_that_ends_inside_a_run_is_not_held_for_it(tracing, exported):
    """Only the run's own trace is held; a stray span ending in its context would be written under the wrong run."""
    stray = tracer().start_span("housekeeping")
    with run_context(RUN):
        stray.end()

    assert tracing.recorder.take(RUN) == []


def test_a_parent_comes_before_a_child_that_started_in_the_same_instant(exported):
    """The child ends first, so it is exported first; the rows must still read parent, then child."""
    instant = 1_757_937_600_000_000_000
    with run_context(RUN), tracer().start_as_current_span("tick", start_time=instant, end_on_exit=False) as parent:
        child = tracer().start_span("classify", start_time=instant)
        child.end(end_time=instant + 1_000)
        parent.end(end_time=instant + 2_000)

    assert [span.name for span in exported.get_finished_spans()] == ["classify", "tick"]
    assert [row.name for row in rows_for(exported.get_finished_spans())] == ["tick", "classify"]


def test_an_observation_type_the_table_does_not_know_is_recorded_as_a_plain_span(exported):
    """As Langfuse does: a mistyped type must not abort the transaction that pays a refund."""
    with run_context(RUN), tracer().start_as_current_span("generate") as span:
        span.set_attribute(Attr.TYPE, "llm")

    (row,) = rows_for(exported.get_finished_spans())

    assert row.kind == "span"
    assert row.attributes[Attr.TYPE] == "llm"


@pytest.mark.parametrize("base", ["http://127.0.0.1:3000/api/public/otel", "http://127.0.0.1:3000/api/public/otel/"])
def test_the_endpoint_setting_is_langfuses_otlp_base_and_the_traces_path_is_added(base):
    environ = {
        "OPSAGENT_OTLP_ENDPOINT": base,
        "OPSAGENT_LANGFUSE_PUBLIC_KEY": "pk-lf-x",
        "OPSAGENT_LANGFUSE_SECRET_KEY": "sk-lf-y",
    }

    exporter = exporter_from_env(environ)

    assert exporter._endpoint == "http://127.0.0.1:3000/api/public/otel/v1/traces"


def test_record_spans_writes_nothing_when_tracing_is_not_installed(monkeypatch):
    """A process that never set tracing up has nothing to write, and must not fail for it."""
    from app import tracing as module

    monkeypatch.setattr(module, "_installed", None)

    assert record_spans(object(), uuid4()) == 0
