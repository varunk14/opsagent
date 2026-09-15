"""
The evaluation commands.

    python -m evals record    on a machine with Ollama: run the golden cases live, keep every reply and vector
    python -m evals gate      anywhere, with no model: replay, score, compare with the committed baseline
    python -m evals accept    after a deliberate change: write the baseline and scoreboard the recordings score
    python -m evals verify    on a machine with Ollama: record again and report anything that differs
    python -m evals full      on a machine with Ollama, before a release: judge every case, write evals/full.md

`record` reuses whatever is already recorded, so a recording that stopped partway carries on
where it was; `--fresh` starts again from nothing, which is what a changed prompt or model needs.
Whatever was recorded is saved even when the run fails. Once the runs have rested, the same
model judges each smoke case, and its verdicts are recorded with everything else.

`gate` fails -- exit status 1, every reason printed -- when a measure is worse than the
committed baseline, when a case that was safe became unsafe, when a prompt or case changed
since recording, or when the committed scoreboard no longer says what the recordings score.
The judge's verdicts are on that scoreboard, so a changed verdict shows, but the judge's score
fails nothing. There is no baseline until `accept` writes one, and changing it is a reviewed
change to a committed file.

`verify` is the check a replay cannot make. A recording is keyed by its prompt, not by what the
model said, so a hand-edited reply or verdict would replay as real; recording again live, where
replies are exact, finds it. It reads the committed recordings and never writes them.

`full` is layer 3: everything `record` does, then the judge on every case rather than the smoke
cases, with the whole board written for a reviewed commit. It gates nothing; it is read.

Each command runs in a database of its own, created from an admin connection
(OPSAGENT_EVAL_ADMIN_URL, by default the local development Postgres) and dropped afterwards,
so no command ever touches the application's database.
"""

import argparse
import os
import sys
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.embeddings import EMBEDDING_MODEL, Embedder, OllamaEmbedder
from app.llm import DEFAULT_MODEL, Model, Ollama
from evals.golden import EVALS_DIR, GoldenCase, load_cases
from evals.judge import Verdict, judge_board_of, judge_cases, render_judge
from evals.recording import (
    RecordedEmbedder,
    RecordedModel,
    RecordingEmbedder,
    RecordingMissing,
    RecordingModel,
    Recordings,
)
from evals.runner import CaseResult, run_cases
from evals.scoring import Scoreboard, compare, render_markdown, scoreboard_of

RECORDINGS = EVALS_DIR / "recordings.jsonl"
BASELINE = EVALS_DIR / "baseline.json"
SCOREBOARD = EVALS_DIR / "scoreboard.md"
FULL = EVALS_DIR / "full.md"

ADMIN_URL_VAR = "OPSAGENT_EVAL_ADMIN_URL"
# The maintenance database of the local stack in docker-compose.yml; only ever used to create and drop scratch ones.
DEFAULT_ADMIN_URL = "postgresql://opsagent:dev@localhost:5432/postgres"


@contextmanager
def scratch_database(admin_url: str) -> Iterator[str]:
    """A database made for one command and dropped after it, whatever happened in between."""
    name = f"opsagent_eval_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        # The admin connection's host, user and password, with only the database changed.
        yield make_conninfo(admin_url, dbname=name)
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def record(
    admin_url: str,
    cases: Sequence[GoldenCase],
    model: Model,
    embedder: Embedder,
    recordings_path: Path = RECORDINGS,
    model_name: str = DEFAULT_MODEL,
    fresh: bool = False,
) -> list[CaseResult]:
    """
    Run `cases` with a live model, judge the smoke cases among them, and keep what it said at `recordings_path`.

    What is already recorded is reused rather than asked again, unless `fresh`. What was
    recorded is saved even if the run stops partway, so a long recording can be run again.
    """
    recordings = Recordings() if fresh else Recordings.load(recordings_path)
    recording = RecordingModel(model, recordings, model=model_name, reuse=not fresh)
    try:
        with scratch_database(admin_url) as dsn:
            results = run_cases(dsn, cases, recording, RecordingEmbedder(embedder, recordings, reuse=not fresh))
        judge_cases(recording, cases, results)
        return results
    finally:
        recordings.save(recordings_path)


def full(
    admin_url: str,
    cases: Sequence[GoldenCase],
    model: Model,
    embedder: Embedder,
    recordings_path: Path = RECORDINGS,
    full_path: Path = FULL,
    model_name: str = DEFAULT_MODEL,
) -> Scoreboard:
    """
    Layer 3: record `cases` as `record` does, have the model judge every one, and write the whole board to `full_path`.

    The verdicts are kept in the same recordings, so whatever is already recorded is never asked again.
    """
    results = record(admin_url, cases, model, embedder, recordings_path, model_name)
    recordings = Recordings.load(recordings_path)
    try:
        verdicts = judge_cases(
            RecordingModel(model, recordings, model=model_name, reuse=True), cases, results, every_case=True
        )
    finally:
        recordings.save(recordings_path)
    board = scoreboard_of(cases, results)
    judged = judge_board_of(cases, results, verdicts, every_case=True)
    full_path.write_text(render_markdown(board) + render_judge(judged, every_case=True))
    return board


def verify(
    admin_url: str,
    cases: Sequence[GoldenCase],
    model: Model,
    embedder: Embedder,
    recordings_path: Path = RECORDINGS,
    model_name: str = DEFAULT_MODEL,
) -> list[str]:
    """Every reply or vector a fresh live recording of `cases` gives that the committed recordings do not."""
    committed = Recordings.load(recordings_path)
    with tempfile.TemporaryDirectory() as scratch:
        live_path = Path(scratch) / "live.jsonl"
        record(admin_url, cases, model, embedder, live_path, model_name, fresh=True)
        live = Recordings.load(live_path)

    problems = []
    for key, reply in sorted(live.replies.items()):
        was = committed.replies.get(key)
        if was is None:
            problems.append(f"a {reply.task} prompt ({key[:12]}) is not in the committed recordings")
        elif was != reply:
            problems.append(f"the {reply.task} reply ({key[:12]}) differs from the committed recording")
    for key, (_, packed) in sorted(live.vectors.items()):
        stored = committed.vectors.get(key)
        if stored is None:
            problems.append(f"an embedding ({key[:12]}) is not in the committed recordings")
        elif stored[1] != packed:
            problems.append(f"an embedding ({key[:12]}) differs from the committed recording")
    return problems


def replay(
    admin_url: str,
    cases: Sequence[GoldenCase],
    recordings_path: Path = RECORDINGS,
    model_name: str = DEFAULT_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
) -> list[CaseResult]:
    """Run `cases` from recordings alone. Anything never recorded raises RecordingMissing."""
    recordings = Recordings.load(recordings_path)
    with scratch_database(admin_url) as dsn:
        return run_cases(
            dsn, cases, RecordedModel(recordings, model=model_name), RecordedEmbedder(recordings, model=embedding_model)
        )


def recorded_verdicts(
    cases: Sequence[GoldenCase],
    results: Sequence[CaseResult],
    recordings_path: Path = RECORDINGS,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Verdict | None]:
    """The judge's recorded verdicts on the smoke cases. A verdict never recorded raises RecordingMissing."""
    return judge_cases(RecordedModel(Recordings.load(recordings_path), model=model_name), cases, results)


def scoreboard_text(cases: Sequence[GoldenCase], results: Sequence[CaseResult], verdicts: dict[str, Verdict | None]) -> str:
    """The committed scoreboard: layer 1, then the judge's section."""
    return render_markdown(scoreboard_of(cases, results)) + render_judge(judge_board_of(cases, results, verdicts))


def gate(
    admin_url: str,
    cases: Sequence[GoldenCase],
    recordings_path: Path = RECORDINGS,
    baseline_path: Path = BASELINE,
    scoreboard_path: Path = SCOREBOARD,
    model_name: str = DEFAULT_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
) -> list[str]:
    """Every reason the recordings do not pass. Empty means nothing is worse and no safe case became unsafe."""
    if not baseline_path.exists():
        return [f"there is no baseline at {baseline_path}: run `python -m evals accept` once and commit what it writes"]
    try:
        results = replay(admin_url, cases, recordings_path, model_name, embedding_model)
        verdicts = recorded_verdicts(cases, results, recordings_path, model_name)
    except RecordingMissing as missing:
        return [str(missing)]

    problems = compare(scoreboard_of(cases, results), Scoreboard.from_json(baseline_path.read_text()))
    if not scoreboard_path.exists() or scoreboard_path.read_text() != scoreboard_text(cases, results, verdicts):
        problems.append(
            f"the committed scoreboard {scoreboard_path.name} does not say what the recordings score: "
            "run `python -m evals accept` if the change is deliberate"
        )
    return problems


def accept(
    admin_url: str,
    cases: Sequence[GoldenCase],
    recordings_path: Path = RECORDINGS,
    baseline_path: Path = BASELINE,
    scoreboard_path: Path = SCOREBOARD,
    model_name: str = DEFAULT_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
) -> Scoreboard:
    """Score the recordings and write that as the baseline and scoreboard, for a reviewed commit."""
    results = replay(admin_url, cases, recordings_path, model_name, embedding_model)
    board = scoreboard_of(cases, results)
    baseline_path.write_text(board.to_json())
    scoreboard_path.write_text(scoreboard_text(cases, results, recorded_verdicts(cases, results, recordings_path, model_name)))
    return board


def report(problems: Sequence[str], passed: str) -> int:
    """Print every problem and exit 1, or say what passed and exit 0."""
    for problem in problems:
        print(f"  FAIL {problem}")
    if problems:
        return 1
    print(f"  {passed}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals", description="Record, gate, accept and verify the golden set.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, purpose in (
        ("record", "run the golden cases with the live model and keep what it said"),
        ("gate", "replay the recordings and fail on anything worse than the baseline or newly unsafe"),
        ("accept", "write the baseline and scoreboard the recordings score"),
        ("verify", "record every case again, live, and report anything that differs from the committed recordings"),
        ("full", "record and judge every case with the live model, and write the whole board"),
    ):
        command = commands.add_parser(name, help=purpose)
        command.add_argument(
            "--admin-url",
            default=os.environ.get(ADMIN_URL_VAR, DEFAULT_ADMIN_URL),
            help=f"where scratch databases are created (default: ${ADMIN_URL_VAR} or the local stack)",
        )
        command.add_argument("--cases", help="comma-separated case ids (default: every case)")
        command.add_argument("--recordings", type=Path, default=RECORDINGS)
        command.add_argument("--baseline", type=Path, default=BASELINE)
        command.add_argument("--scoreboard", type=Path, default=SCOREBOARD)
        command.add_argument("--full", type=Path, default=FULL, help="where `full` writes the whole board")
        if name == "record":
            command.add_argument("--fresh", action="store_true", help="discard what is recorded and ask the model again")
    arguments = parser.parse_args(argv[1:])

    everything = {case.id: case for case in load_cases()}
    cases = list(everything.values())
    if arguments.cases:
        wanted = [case_id.strip() for case_id in arguments.cases.split(",")]
        unknown = [case_id for case_id in wanted if case_id not in everything]
        if unknown:
            parser.error(f"no such case: {', '.join(unknown)}")
        cases = [everything[case_id] for case_id in wanted]

    if arguments.command == "record":  # pragma: no cover - needs Ollama and its models
        results = record(
            arguments.admin_url, cases, Ollama(), OllamaEmbedder(), arguments.recordings, fresh=arguments.fresh
        )
        print(f"  recorded {len(results)} case(s) into {arguments.recordings}")
        return 0
    if arguments.command == "verify":  # pragma: no cover - needs Ollama and its models
        problems = verify(arguments.admin_url, cases, Ollama(), OllamaEmbedder(), arguments.recordings)
        return report(problems, f"{len(cases)} case(s): the live recording matches the committed one")
    if arguments.command == "full":  # pragma: no cover - needs Ollama and its models
        full(arguments.admin_url, cases, Ollama(), OllamaEmbedder(), arguments.recordings, arguments.full)
        print(arguments.full.read_text())
        return 0
    if arguments.command == "accept":
        accept(
            arguments.admin_url, cases, arguments.recordings, arguments.baseline, arguments.scoreboard,
            embedding_model=EMBEDDING_MODEL,
        )
        print(arguments.scoreboard.read_text())
        return 0

    problems = gate(
        arguments.admin_url, cases, arguments.recordings, arguments.baseline, arguments.scoreboard,
        embedding_model=EMBEDDING_MODEL,
    )
    return report(problems, f"{len(cases)} case(s): no worse than the baseline, no safe case became unsafe")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
