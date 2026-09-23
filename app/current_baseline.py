"""
What one run costs on today's stack, measured the same way `app/baseline.py` measured the naive
one.

The naive baseline in `BASELINE.md` was one small model, one call per message, no routing, no
schema and no retrieval. Today's stack is the whole agent -- classify, extract, retrieve, plan,
tools, guardrail, reply -- so a run is many model calls and its cost is the sum. The comparison
that matters is against the same messages, the same passes, and the same reference rate. So this
script drives the same `fixtures/inbox.jsonl` three times through the real agent, records what
each run cost end-to-end, and writes the paired summary next to the naive one.

The reference rate cancels between the two sides. What survives the comparison is the ratio.

Run:  .venv/bin/python -m app.current_baseline
      (needs Ollama and a throwaway Postgres reachable via OPSAGENT_EVAL_ADMIN_URL)
"""

import json
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from app.adapters.fixture import read_messages
from app.baseline import REFERENCE_RATE, Baseline, StepMeasurement, summarise
from app.contracts import IncomingMessage
from app.db import apply_migrations, connect, database_url
from app.embeddings import Embedder, OllamaEmbedder
from app.graph.build import build_graph
from app.intake import accept
from app.llm import DEFAULT_MODEL, Model, Ollama, Reply
from app.policies import POLICY_DIR, ingest, load_policies
from app.retrieval import PolicyRetriever
from app.run_agent import work_next
from app.seed import load_ledger

PASSES = 3
FIXTURE_LEDGER = Path("fixtures/ledger.json")
FIXTURE_INBOX = Path("fixtures/inbox.jsonl")
CURRENT_MEASUREMENTS = Path("current-measurements.json")


class TrackingModel:
    """A live Ollama model that also records every call it answered."""

    def __init__(self, inner: Model) -> None:
        self.inner = inner
        self.replies: list[Reply] = []

    def generate(self, prompt: str) -> Reply:
        reply = self.inner.generate(prompt)
        self.replies.append(reply)
        return reply

    def take(self) -> list[Reply]:
        """The replies since the last take. Cleared, so the next run starts empty."""
        done = self.replies
        self.replies = []
        return done


def _url_with_dbname(admin: str, dbname: str) -> str:
    """Rewrite the connection string's database name; leave everything else."""
    parts = {k: str(v) for k, v in conninfo_to_dict(admin).items() if v is not None}
    parts["dbname"] = dbname
    return make_conninfo(**parts)


def scratch_database_url() -> str:
    """A fresh database, made by hand and dropped again at the end of the run."""
    admin = database_url()
    name = f"opsagent_current_baseline_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(_url_with_dbname(admin, "postgres"), autocommit=True) as maintenance:
        maintenance.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return _url_with_dbname(admin, name)


def drop_database(dsn: str) -> None:
    name = str(conninfo_to_dict(dsn).get("dbname", ""))
    if not name:
        return
    with psycopg.connect(_url_with_dbname(dsn, "postgres"), autocommit=True) as maintenance:
        maintenance.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )


def as_incoming(fixture_message: IncomingMessage, pass_number: int) -> IncomingMessage:
    """Rewrite the message's ids per pass so the intake sees it as new every time."""
    payload = fixture_message.model_dump()
    payload["external_id"] = f"{fixture_message.external_id}#pass{pass_number}"
    return IncomingMessage.model_validate(payload)


def measure_one_run(
    dsn: str, graph, message: IncomingMessage, tracker: TrackingModel
) -> StepMeasurement:
    """Accept a fresh copy of the message, work its run to rest, and add up what it cost."""
    tracker.take()
    with psycopg.connect(dsn) as connection:
        accept(connection, message)
    started = time.perf_counter()
    with psycopg.connect(dsn) as connection:
        while work_next(connection, graph) is not None:
            pass
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    replies = tracker.take()
    if not replies:
        raise ValueError(
            f"the run for {message.external_id} made no model calls; something is not wired"
        )
    return StepMeasurement(
        model=replies[0].model,
        prompt_tokens=sum(r.prompt_tokens for r in replies),
        completion_tokens=sum(r.completion_tokens for r in replies),
        latency_ms=elapsed_ms,
    )


def report(baseline: Baseline) -> str:
    """A short summary next to the numbers the naive baseline reported."""
    return (
        "# Current stack baseline\n\n"
        f"**Recorded:** {datetime.now(UTC).date()}\n"
        f"**Model:** `{baseline.model}` for planning; `nomic-embed-text` for policy retrieval,\n"
        "run locally through Ollama. The whole agent: classify, extract, retrieve, plan, tools,\n"
        "guardrail, reply -- as it runs in production.\n\n"
        f"**Sample:** {baseline.runs} runs -- {PASSES} passes over the {baseline.runs // PASSES}\n"
        "messages in `fixtures/inbox.jsonl`, the same inbox the naive baseline used.\n\n"
        "## The numbers, next to the naive baseline\n\n"
        "| | Naive (BASELINE.md) | Current stack |\n"
        "|---|---|---|\n"
        f"| Tokens per run | 338.5 | {baseline.tokens_per_run:,.2f} |\n"
        f"| Cost per 100 runs | $0.0164 | ${baseline.cost_per_100_runs:.4f} |\n"
        f"| Latency p50 | 5,936 ms | {baseline.p50_ms:,} ms |\n"
        f"| Latency p95 | 11,642 ms | {baseline.p95_ms:,} ms |\n\n"
        "Percentiles are nearest-rank over 12 runs, exactly the way the naive baseline is\n"
        "computed. That is not enough to characterise a tail; it is enough that both sides\n"
        "compare like for like.\n\n"
        "The reference rate is the same on both sides (see `BASELINE.md`), so it cancels. What\n"
        "survives the comparison is the ratio. The raw per-run token counts are in\n"
        "`current-measurements.json` alongside the naive `baseline-measurements.json`, so a\n"
        "different rate can rederive both without measuring again.\n"
    )


def main(argv: list[str]) -> int:  # pragma: no cover - interactive driver
    _ = argv
    print("Setting up a scratch database (migrated, policies ingested, ledger loaded).")
    dsn = scratch_database_url()
    try:
        with psycopg.connect(dsn) as connection:
            apply_migrations(connection)
        embedder: Embedder = OllamaEmbedder()
        with psycopg.connect(dsn) as connection:
            ingest(connection, embedder, load_policies(POLICY_DIR))
        with psycopg.connect(dsn) as connection:
            load_ledger(connection, FIXTURE_LEDGER)

        messages = list(read_messages(FIXTURE_INBOX))
        tracker = TrackingModel(Ollama())
        graph = build_graph(tracker, PolicyRetriever(lambda: connect(dsn), embedder))

        print(f"Measuring {len(messages) * PASSES} live runs on {DEFAULT_MODEL}. This is slow.")
        print("  warming up (discarded, so a cold load stays out of the sample)")
        try:
            measure_one_run(dsn, graph, as_incoming(messages[0], 0), tracker)
        except (OSError, TimeoutError, ValueError) as exc:
            print(f"  cannot warm up: {exc}")
            return 1

        measurements: list[StepMeasurement] = []
        total = len(messages) * PASSES
        for number, (pass_no, message) in enumerate(
            [(p, m) for p in range(1, PASSES + 1) for m in messages], start=1
        ):
            try:
                step = measure_one_run(dsn, graph, as_incoming(message, pass_no), tracker)
            except (OSError, TimeoutError, ValueError) as exc:
                print(f"  run {number}/{total} failed: {exc}")
                return 1
            measurements.append(step)
            print(
                f"  {number}/{total}  {step.prompt_tokens:>6} in  "
                f"{step.completion_tokens:>5} out  {step.latency_ms:>7} ms"
            )

        current = summarise(measurements, REFERENCE_RATE)
        CURRENT_MEASUREMENTS.write_text(
            json.dumps([asdict(m) for m in measurements], indent=2) + "\n",
            encoding="utf-8",
        )
        Path("CURRENT.md").write_text(report(current), encoding="utf-8")

        print(f"\n  tokens/run       {current.tokens_per_run:,.2f}")
        print(f"  cost/100 runs    ${current.cost_per_100_runs:.4f}")
        print(f"  latency p50/p95  {current.p50_ms:,} / {current.p95_ms:,} ms")
        print("\n  wrote CURRENT.md and current-measurements.json")
        return 0
    finally:
        drop_database(dsn)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
