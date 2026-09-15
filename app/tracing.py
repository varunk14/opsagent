"""
Tracing: every run is one OpenTelemetry trace, recorded with the steps it describes.

Four decisions shape this module.

The trace id is the run's id. A run is worked in ticks, by whichever worker claims
it, sometimes days apart around an approval. Taking each tick's trace id from the
run's UUID -- both are 128 bits -- puts all of it in one trace without passing
anything between processes. Which run a span belongs to is settled when it starts,
exactly as its trace id is, so a span that ends after its run's context has closed
is still the run's.

Spans are recorded where the steps are. A processor holds each finished span under
its run, and the driver writes them with record_spans in the transaction that commits
the step. A tick that never commits -- a lost claim, a crash -- leaves no spans, just
as it leaves no steps, and `discard` drops what it held. A run holds at most
MAX_HELD_SPANS_PER_RUN, so a retry storm cannot grow a worker's memory without limit.

Writing spans never takes a step down. That transaction may be paying a refund, so
every value the spans table would refuse is left out of its column (and kept, as it
was set, among the attributes): a count that is not a whole number, a cost that is
not a finite amount, a version that is not a version. A generation left without its
model, counts or cost is written as a plain span that says why.

A copy goes to Langfuse, and only on this machine. The exporter exists only when
OPSAGENT_OTLP_ENDPOINT is set, refuses any host but 127.0.0.1 or localhost, and its
connection honours no proxy from the environment and follows no redirect -- either
would carry spans addressed to this machine somewhere else. It sends from a
background batch with a short timeout, so a Langfuse that is down never raises into a
tick or holds it up. The resource names the service and nothing else: a worker's host
name and pid are what `locked_by` holds, and they stay in the database.
"""

import argparse
import base64
import os
import re
import secrets
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import psycopg
import requests
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator
from opentelemetry.trace import StatusCode
from psycopg.types.json import Jsonb

SERVICE_NAME = "opsagent"
TRACER_NAME = "opsagent"

ENDPOINT_VAR = "OPSAGENT_OTLP_ENDPOINT"
PUBLIC_KEY_VAR = "OPSAGENT_LANGFUSE_PUBLIC_KEY"
SECRET_KEY_VAR = "OPSAGENT_LANGFUSE_SECRET_KEY"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
# Long enough for a local Langfuse under load, short enough that shutting a worker down stays quick.
EXPORT_TIMEOUT_SECONDS = 5

# A tick makes a few dozen spans. Far past that, something is looping, and holding
# every span it makes would turn a runaway step into a runaway worker.
MAX_HELD_SPANS_PER_RUN = 1_000

# What the spans table accepts. Any other type is recorded as a plain span, as Langfuse
# itself does, so a mistyped attribute can never abort the transaction that pays a refund.
KINDS = frozenset({"span", "chain", "generation", "retriever", "embedding", "tool", "guardrail"})
PROMPT_VERSION_PATTERN = re.compile(r"[0-9a-f]{12}")
GENERATION_NEEDS = "a generation needs a model, token counts and a cost"
UNNAMED = "unnamed"


class Attr:
    """Span attribute names: Langfuse's own where Langfuse reads one (checked against 4.36.1), ours otherwise."""

    TYPE = "langfuse.observation.type"
    TRACE_NAME = "langfuse.trace.name"
    MODEL = "gen_ai.request.model"
    INPUT_TOKENS = "gen_ai.usage.input_tokens"
    OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    # Langfuse reads cost as JSON; the exact decimal string is ours, and is what the table stores.
    COST_DETAILS = "langfuse.observation.cost_details"
    COST_USD = "opsagent.cost_usd"
    # Filterable in Langfuse under metadata, and kept verbatim under our own name.
    PROMPT_VERSION = "opsagent.prompt_version"
    PROMPT_VERSION_METADATA = "langfuse.observation.metadata.prompt_version"
    LATENCY_MS = "opsagent.latency_ms"
    # Why a span that asked to be a generation was written as a plain span.
    RECORDED_AS_SPAN = "opsagent.recorded_as_span"
    # What a step decided: values chosen from our own lists, never text the customer wrote.
    INTENT = "opsagent.intent"
    CLASSIFICATION_CONFIDENCE = "opsagent.classification_confidence"
    ORDER_ID_FOUND = "opsagent.order_id_found"
    AMOUNT_FOUND = "opsagent.amount_found"
    PASSAGES = "opsagent.passages"
    SOURCES = "opsagent.sources"
    TOOL = "opsagent.tool"
    PROPOSAL_CONFIDENCE = "opsagent.proposal_confidence"
    FAILURE = "opsagent.failure"
    # Policy search.
    COST_COUNTED = "opsagent.cost_counted"
    K = "opsagent.k"
    MAX_DISTANCE = "opsagent.max_distance"
    TOP_DISTANCE = "opsagent.top_distance"
    # A tick, and what its act step did.
    ATTEMPT = "opsagent.attempt"
    OUTCOME = "opsagent.outcome"
    RESULT = "opsagent.result"
    APPROVAL_ID = "opsagent.approval_id"
    VERDICT = "opsagent.verdict"
    LIMIT_PAISE = "opsagent.limit_paise"
    MIN_CONFIDENCE = "opsagent.min_confidence"
    REASON = "opsagent.reason"


_current_run: ContextVar[UUID | None] = ContextVar("opsagent_current_run", default=None)


@contextmanager
def run_context(run_id: UUID) -> Iterator[None]:
    """Everything traced inside belongs to this run's trace."""
    token = _current_run.set(run_id)
    try:
        yield
    finally:
        _current_run.reset(token)


class RunTraceIds(IdGenerator):
    """A trace started inside run_context takes the run's id; any other trace gets a random one."""

    def __init__(self) -> None:
        self._random = RandomIdGenerator()

    def generate_span_id(self) -> int:
        return self._random.generate_span_id()

    def generate_trace_id(self) -> int:
        run_id = _current_run.get()
        return run_id.int if run_id is not None else self._random.generate_trace_id()


class RunSpanRecorder(SpanProcessor):
    """
    Holds each finished span of a run until the driver writes it or drops it.

    A span is the run's if it started inside that run's context, in that run's trace --
    decided at start, as its trace id was -- so where it happens to end does not
    matter, and spans traced outside any run are never held at all.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_in_a_run: set[tuple[int, int]] = set()
        self._held: dict[int, list[ReadableSpan]] = {}
        self._dropped: dict[int, int] = {}

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        run_id = _current_run.get()
        context = span.get_span_context()
        if run_id is not None and context.trace_id == run_id.int:
            with self._lock:
                self._started_in_a_run.add((context.trace_id, context.span_id))

    def on_end(self, span: ReadableSpan) -> None:
        context = span.context
        if context is None:  # pragma: no cover - the SDK gives every span a context
            return
        with self._lock:
            key = (context.trace_id, context.span_id)
            if key not in self._started_in_a_run:
                return
            self._started_in_a_run.discard(key)
            held = self._held.setdefault(context.trace_id, [])
            if len(held) >= MAX_HELD_SPANS_PER_RUN:
                self._dropped[context.trace_id] = self._dropped.get(context.trace_id, 0) + 1
                return
            held.append(span)

    def take(self, run_id: UUID) -> list[ReadableSpan]:
        return self.take_with_dropped(run_id)[0]

    def take_with_dropped(self, run_id: UUID) -> tuple[list[ReadableSpan], int]:
        """The run's finished spans, and how many past the bound were not kept. Both are cleared."""
        with self._lock:
            return self._held.pop(run_id.int, []), self._dropped.pop(run_id.int, 0)

    def clear(self) -> None:
        with self._lock:
            self._started_in_a_run.clear()
            self._held.clear()
            self._dropped.clear()

    def shutdown(self) -> None:
        self.clear()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


_installed: "Tracing | None" = None


class Tracing:
    """
    One process's tracing: a provider whose spans are held for the database and,
    when an exporter is given, copied to it.

    `immediate` exports each span as it ends, for tests; a worker batches.
    """

    def __init__(self, exporter: SpanExporter | None = None, *, immediate: bool = False) -> None:
        self.exporter = exporter
        self.recorder = RunSpanRecorder()
        # Built from this one attribute rather than Resource.create(), which would also
        # read OTEL_RESOURCE_ATTRIBUTES -- a way for a host name or pid to leave the machine.
        self.provider = TracerProvider(resource=Resource({"service.name": SERVICE_NAME}), id_generator=RunTraceIds())
        self.provider.add_span_processor(self.recorder)
        if exporter is not None:
            processor: SpanProcessor
            if immediate:
                processor = SimpleSpanProcessor(exporter)
            else:
                processor = BatchSpanProcessor(exporter, export_timeout_millis=EXPORT_TIMEOUT_SECONDS * 1000)
            self.provider.add_span_processor(processor)
        self.tracer = self.provider.get_tracer(TRACER_NAME)

    def install(self) -> "Tracing":
        """Make this the process's tracing. OpenTelemetry keeps the first; a second would silently trace nothing."""
        global _installed
        if _installed is not None:
            raise RuntimeError("tracing is already installed in this process")
        trace.set_tracer_provider(self.provider)
        _installed = self
        return self

    def shutdown(self) -> None:
        """Send what is still batched, within the export timeout, and stop."""
        self.provider.shutdown()


def tracer() -> trace.Tracer:
    """The tracer the agent's code uses: the installed one, or OpenTelemetry's no-op one when none is."""
    return trace.get_tracer(TRACER_NAME)


class LoopbackSession(requests.Session):
    """
    The connection traces leave on, with two of requests' defaults turned off.

    requests takes a proxy from HTTP_PROXY or ALL_PROXY, which would send spans
    addressed to 127.0.0.1 to the proxy's host; and it follows redirects, which would
    send a batch again to wherever a redirect points. A Langfuse on this machine needs
    neither, so neither happens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.trust_env = False
        self.max_redirects = 0  # a second guard: were a redirect ever seen, it would be refused

    def get_redirect_target(self, resp: requests.Response) -> str | None:
        """No response names somewhere else to go: a redirect comes back as the failure it is."""
        return None


def loopback_exporter(endpoint: str, public_key: str, secret_key: str) -> OTLPSpanExporter:
    """An OTLP/HTTP exporter to a Langfuse on this machine. Any other host is refused."""
    parts = urlsplit(endpoint)
    if parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS:
        # Only scheme and host are echoed: an endpoint can carry credentials in it.
        raise ValueError(
            f"traces are only sent to this machine (http://127.0.0.1 or http://localhost), "
            f"not {parts.scheme}://{parts.hostname}"
        )
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return OTLPSpanExporter(
        endpoint=endpoint,
        headers={"Authorization": f"Basic {credentials}", "x-langfuse-ingestion-version": "4"},
        timeout=EXPORT_TIMEOUT_SECONDS,
        session=LoopbackSession(),
    )


def exporter_from_env(environ: Mapping[str, str]) -> OTLPSpanExporter | None:
    """
    The Langfuse exporter the environment asks for, or None when it asks for none.

    OPSAGENT_OTLP_ENDPOINT is Langfuse's OTLP base, e.g. http://127.0.0.1:3000/api/public/otel.
    """
    endpoint = environ.get(ENDPOINT_VAR, "").strip()
    if not endpoint:
        return None
    missing = [name for name in (PUBLIC_KEY_VAR, SECRET_KEY_VAR) if not environ.get(name, "").strip()]
    if missing:
        raise ValueError(f"{ENDPOINT_VAR} is set, so {' and '.join(missing)} must be set too")
    return loopback_exporter(
        endpoint.rstrip("/") + "/v1/traces",
        public_key=environ[PUBLIC_KEY_VAR].strip(),
        secret_key=environ[SECRET_KEY_VAR].strip(),
    )


@dataclass(frozen=True)
class SpanRow:
    """One span as the spans table stores it: each field is the column of the same name."""

    trace_id: UUID
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    started_at: datetime
    ended_at: datetime
    status: str
    status_message: str | None
    model: str | None
    prompt_version: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: Decimal | None
    attributes: dict[str, Any]


INSERT_SPAN = """
    INSERT INTO spans (trace_id, span_id, parent_span_id, name, kind, started_at, ended_at, status,
                       status_message, model, prompt_version, input_tokens, output_tokens, cost_usd, attributes)
    VALUES (%(trace_id)s, %(span_id)s, %(parent_span_id)s, %(name)s, %(kind)s, %(started_at)s, %(ended_at)s,
            %(status)s, %(status_message)s, %(model)s, %(prompt_version)s, %(input_tokens)s, %(output_tokens)s,
            %(cost_usd)s, %(attributes)s)
"""


def _when(nanoseconds: int) -> datetime:
    """Nanoseconds since the epoch, to the microsecond Postgres keeps, without float rounding."""
    seconds, remainder = divmod(nanoseconds, 1_000_000_000)
    return datetime.fromtimestamp(seconds, UTC) + timedelta(microseconds=remainder // 1_000)


def _hex(span_id: int) -> str:
    return format(span_id, "016x")


def _count(value: object) -> int | None:
    """A token count as the table takes it -- a whole number, not negative -- or nothing."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _cost(value: object) -> Decimal | None:
    """A cost as the table takes it -- a finite amount, not negative -- or nothing."""
    if value is None or isinstance(value, bool):
        return None
    try:
        cost = Decimal(str(value))
    except InvalidOperation:
        return None
    return cost if cost.is_finite() and cost >= 0 else None


def _version(value: object) -> str | None:
    return value if isinstance(value, str) and PROMPT_VERSION_PATTERN.fullmatch(value) else None


def _row(span: ReadableSpan) -> SpanRow:
    """One span as a row the spans table will accept, whatever was set on it."""
    context = span.context
    if context is None or span.start_time is None:  # pragma: no cover - the SDK sets both on every started span
        raise ValueError(f"span {span.name!r} was never started")
    attributes = {
        key: list(value) if isinstance(value, tuple) else value for key, value in (span.attributes or {}).items()
    }
    model = attributes.get(Attr.MODEL)
    model = None if model is None else str(model)
    input_tokens = _count(attributes.get(Attr.INPUT_TOKENS))
    output_tokens = _count(attributes.get(Attr.OUTPUT_TOKENS))
    cost = _cost(attributes.get(Attr.COST_USD))
    kind = str(attributes.get(Attr.TYPE, "span"))
    if kind not in KINDS:
        kind = "span"
    if kind == "generation" and None in (model, input_tokens, output_tokens, cost):
        kind = "span"
        attributes[Attr.RECORDED_AS_SPAN] = GENERATION_NEEDS
    started_at = _when(span.start_time)
    ended_at = started_at if span.end_time is None else max(started_at, _when(span.end_time))
    return SpanRow(
        trace_id=UUID(int=context.trace_id),
        span_id=_hex(context.span_id),
        parent_span_id=_hex(span.parent.span_id) if span.parent is not None else None,
        name=span.name if span.name.strip() else UNNAMED,
        kind=kind,
        started_at=started_at,
        ended_at=ended_at,
        status="error" if span.status.status_code is StatusCode.ERROR else "ok",
        status_message=span.status.description,
        model=model,
        prompt_version=_version(attributes.get(Attr.PROMPT_VERSION)),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        attributes=attributes,
    )


def rows_for(spans: Sequence[ReadableSpan]) -> list[SpanRow]:
    """Spans as rows, by start time, a parent before a child that started in the same instant."""
    rows = [_row(span) for span in spans]
    parents = {row.span_id: row.parent_span_id for row in rows}

    def depth(row: SpanRow) -> int:
        levels, parent = 0, row.parent_span_id
        while parent is not None and parent in parents and levels < len(parents):
            levels, parent = levels + 1, parents[parent]
        return levels

    return sorted(rows, key=lambda row: (row.started_at, depth(row)))


def _parameters(row: SpanRow) -> dict[str, Any]:
    values = {field.name: getattr(row, field.name) for field in fields(row)}
    values["attributes"] = Jsonb(row.attributes)
    return values


def record_spans(connection: psycopg.Connection, run_id: UUID) -> int:
    """
    Write the run's finished spans in the caller's transaction, and say how many.

    Commits nothing: the caller's transaction is the step's, so the spans commit with
    it or not at all. A process that never installed tracing writes nothing.
    """
    if _installed is None:
        return 0
    spans, dropped = _installed.recorder.take_with_dropped(run_id)
    if dropped:
        print(
            f"  warning: {dropped} spans of run {run_id} were not recorded: "
            f"one step made more than {MAX_HELD_SPANS_PER_RUN}",
            file=sys.stderr,
        )
    rows = rows_for(spans)
    if rows:
        with connection.cursor() as cursor:
            cursor.executemany(INSERT_SPAN, [_parameters(row) for row in rows])
    return len(rows)


def discard(run_id: UUID) -> None:
    """Drop what a tick that will never commit was holding, so its next tick does not write it."""
    if _installed is not None:
        _installed.recorder.take_with_dropped(run_id)


# --- Langfuse on this machine ------------------------------------------------------------

PROJECT_URL_VAR = "OPSAGENT_LANGFUSE_PROJECT_URL"
LANGFUSE_URL = "http://127.0.0.1:3000"  # where compose.tracing.yml publishes the UI
LANGFUSE_PROJECT_ID = "opsagent-local"  # the project compose.tracing.yml creates
# Every secret compose.tracing.yml requires. Each is generated when the env file lacks it.
LANGFUSE_SECRETS = (
    "LANGFUSE_POSTGRES_PASSWORD",
    "LANGFUSE_CLICKHOUSE_PASSWORD",
    "LANGFUSE_MINIO_PASSWORD",
    "LANGFUSE_REDIS_PASSWORD",
    "LANGFUSE_NEXTAUTH_SECRET",
    "LANGFUSE_SALT",
    "LANGFUSE_ENCRYPTION_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


def generated_secrets() -> dict[str, str]:
    """Fresh values for every Langfuse secret: hex and UUIDs only, so the file is safe to source from a shell."""
    return {
        "LANGFUSE_POSTGRES_PASSWORD": secrets.token_hex(24),
        "LANGFUSE_CLICKHOUSE_PASSWORD": secrets.token_hex(24),
        "LANGFUSE_MINIO_PASSWORD": secrets.token_hex(24),
        "LANGFUSE_REDIS_PASSWORD": secrets.token_hex(24),
        "LANGFUSE_NEXTAUTH_SECRET": secrets.token_hex(32),
        "LANGFUSE_SALT": secrets.token_hex(32),
        "LANGFUSE_ENCRYPTION_KEY": secrets.token_hex(32),  # Langfuse requires exactly 256 bits, as hex
        "LANGFUSE_PUBLIC_KEY": f"pk-lf-{uuid4()}",
        "LANGFUSE_SECRET_KEY": f"sk-lf-{uuid4()}",
    }


def env_settings(text: str) -> dict[str, str]:
    """The NAME=value lines of an env file. Comments and anything else are skipped."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() and not name.lstrip().startswith("#"):
            found[name.strip()] = value.strip()
    return found


def write_env(path: Path) -> list[str]:
    """
    Add what Langfuse on this machine needs to the env file at `path`; return the names added.

    That is every secret compose.tracing.yml requires, generated fresh, and the settings a
    worker and the screen use to reach it. A name the file already has is left exactly as
    it is, and the worker's keys are taken from the Langfuse keys it already has: nothing
    is rewritten, only appended. A file this creates is readable by its owner only.
    """
    text = path.read_text() if path.exists() else ""
    present = env_settings(text)
    values = {**generated_secrets(), **present}
    wanted = {
        **{name: values[name] for name in LANGFUSE_SECRETS},
        ENDPOINT_VAR: f"{LANGFUSE_URL}/api/public/otel",
        PUBLIC_KEY_VAR: values["LANGFUSE_PUBLIC_KEY"],
        SECRET_KEY_VAR: values["LANGFUSE_SECRET_KEY"],
        PROJECT_URL_VAR: f"{LANGFUSE_URL}/project/{LANGFUSE_PROJECT_ID}",
    }
    added = [name for name in wanted if name not in present]
    if not added:
        return []
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a") as file:
        file.write(("\n" if text and not text.endswith("\n") else "") + "".join(f"{name}={wanted[name]}\n" for name in added))
    return added


def main(argv: Sequence[str]) -> int:
    """`python -m app.tracing env [--path .env]`: the settings for Langfuse on this machine."""
    parser = argparse.ArgumentParser(prog="python -m app.tracing", description="Tracing settings for this machine.")
    commands = parser.add_subparsers(dest="command", required=True)
    env = commands.add_parser("env", help="add fresh Langfuse secrets and the worker's settings, changing nothing already there")
    env.add_argument("--path", type=Path, default=Path(".env"), help="the env file (default: .env)")
    arguments = parser.parse_args(argv)

    added = write_env(arguments.path)
    # Names, never values: every value here but the public key is a secret.
    if added:
        print(f"  added to {arguments.path}: {', '.join(added)}")
    else:
        print(f"  {arguments.path} already has every setting; nothing changed")
    print(f"  Langfuse public key: {env_settings(arguments.path.read_text())[PUBLIC_KEY_VAR]}")
    print(f"  start Langfuse with `docker compose -f compose.tracing.yml up -d`, then open {LANGFUSE_URL}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
