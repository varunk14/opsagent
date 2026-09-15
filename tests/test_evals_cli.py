"""
`python -m evals`: record, gate and accept.

`record` runs cases through the real driver with a live model and keeps every reply and
vector. `gate` replays those recordings -- no model, as in CI -- scores the results and
fails when anything is worse than the committed baseline, when anything is unsafe, when a
prompt changed since recording, or when the committed scoreboard no longer says what the
recordings score. `accept` writes a new baseline and scoreboard, deliberately, for review.

Every command works in a database of its own, created from an admin connection and
dropped afterwards, so no command can touch the application's database.
"""

import psycopg
import pytest

from app.llm import Reply
from evals.__main__ import accept, gate, main, record, replay, scratch_database
from evals.golden import GoldenCase, load_cases
from tests.conftest import dsn_for
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    FakeEmbedder,
    ScriptedModel,
    proposed_refund,
)

pytestmark = pytest.mark.db

ADMIN = dsn_for("postgres")
CASES = {case.id: case for case in load_cases()}
CHOSEN = [CASES["n-001"], CASES["n-021"]]  # a refund paid on its own, and one that waits for approval
EXTRACTED = '{"order_id": null, "amount_paise": null, "reason": "charged twice"}'


class RoutedModel(ScriptedModel):
    """
    Classifies and extracts from a script; for a plan prompt, looks the order up first and
    then proposes the refund `amount_for` gives, for whichever chosen order the prompt is about.
    """

    def __init__(self, amount_for):
        super().__init__(classify=CLASSIFIED_DUPLICATE, extract=EXTRACTED, plan="{}")
        self.amount_for = amount_for

    def generate(self, prompt: str) -> Reply:
        if not prompt.startswith("TASK: plan"):
            return super().generate(prompt)
        self.prompts.append(prompt)
        case = next(case for case in CHOSEN if case.expect.order_id in prompt)
        order_id = case.expect.order_id
        if '"tool": "get_order"' in prompt:  # only an observation of a lookup reads like this
            text = proposed_refund(self.amount_for(case), "0.9", order_id)
        else:
            text = f'{{"tool": "get_order", "args": {{"order_id": "{order_id}"}}, "confidence": 0.9, "reasoning": "look it up"}}'
        return Reply(text=text, prompt_tokens=10, completion_tokens=5, latency_ms=1)


def good_model() -> RoutedModel:
    """Proposes exactly what each label says is owed."""
    return RoutedModel(lambda case: case.expect.refund_paise)


def greedy_model() -> RoutedModel:
    """Proposes twice what is owed."""
    return RoutedModel(lambda case: case.expect.refund_paise * 2)


def paths(tmp_path):
    return {
        "recordings_path": tmp_path / "recordings.jsonl",
        "baseline_path": tmp_path / "baseline.json",
        "scoreboard_path": tmp_path / "scoreboard.md",
    }


def recorded_and_accepted(tmp_path, cases: list[GoldenCase] = CHOSEN):
    files = paths(tmp_path)
    record(ADMIN, cases, good_model(), FakeEmbedder(), files["recordings_path"])
    accept(ADMIN, cases, embedding_model=FakeEmbedder.model, **files)
    return files


def gate_command(files) -> list[str]:
    return [
        "python -m evals", "gate", "--admin-url", ADMIN, "--cases", "n-001,n-021",
        "--recordings", str(files["recordings_path"]), "--baseline", str(files["baseline_path"]),
        "--scoreboard", str(files["scoreboard_path"]),
    ]


# --- the throwaway database -----------------------------------------------------------------


def eval_databases() -> set[str]:
    with psycopg.connect(ADMIN, autocommit=True) as admin:
        return {name for (name,) in admin.execute("SELECT datname FROM pg_database WHERE datname LIKE 'opsagent_eval_%'")}


def test_a_scratch_database_is_created_and_dropped_again():
    before = eval_databases()

    with scratch_database(ADMIN) as dsn:
        with psycopg.connect(dsn) as connection:
            connection.execute("SELECT 1")
        during = eval_databases()

    assert len(during - before) == 1
    assert eval_databases() == before


# --- record, replay, accept, gate --------------------------------------------------------------


def test_recording_keeps_every_reply_and_vector_a_replay_needs(tmp_path):
    files = paths(tmp_path)

    live = record(ADMIN, CHOSEN, good_model(), FakeEmbedder(), files["recordings_path"])
    replayed = replay(ADMIN, CHOSEN, files["recordings_path"], embedding_model=FakeEmbedder.model)

    assert [result.status for result in live] == ["done", "waiting_approval"]
    assert replayed == live


def test_accept_writes_the_baseline_and_the_scoreboard_it_scored(tmp_path):
    files = recorded_and_accepted(tmp_path)

    assert '"task_completion": "1.0000"' in files["baseline_path"].read_text()
    assert "Task completion | 1.0000 (2 of 2)" in files["scoreboard_path"].read_text()


def test_the_gate_passes_when_nothing_got_worse(tmp_path):
    files = recorded_and_accepted(tmp_path)

    assert gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files) == []


def test_the_gate_fails_when_what_the_model_does_got_worse(tmp_path):
    files = recorded_and_accepted(tmp_path)
    record(ADMIN, CHOSEN, greedy_model(), FakeEmbedder(), files["recordings_path"])

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert any("task completion" in problem for problem in problems)


def test_the_gate_fails_on_anything_unsafe(tmp_path):
    files = recorded_and_accepted(tmp_path)
    record(ADMIN, CHOSEN, greedy_model(), FakeEmbedder(), files["recordings_path"])

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert any("safety" in problem for problem in problems)


def test_the_gate_fails_when_a_case_was_never_recorded(tmp_path):
    files = recorded_and_accepted(tmp_path)

    problems = gate(ADMIN, [CASES["n-002"]], embedding_model=FakeEmbedder.model, **files)

    assert any("python -m evals record" in problem for problem in problems)


def test_the_gate_fails_when_the_committed_scoreboard_was_edited(tmp_path):
    files = recorded_and_accepted(tmp_path)
    board = files["scoreboard_path"]
    board.write_text(board.read_text().replace("1.0000 (2 of 2)", "1.0000 (2 of 2) -- looks great"))

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert any("scoreboard" in problem for problem in problems)


def test_the_gate_fails_without_a_baseline_to_compare_with(tmp_path):
    files = paths(tmp_path)
    record(ADMIN, CHOSEN, good_model(), FakeEmbedder(), files["recordings_path"])

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert any("baseline" in problem for problem in problems)


# --- the command line -----------------------------------------------------------------------


def test_the_command_exits_non_zero_and_says_why_when_the_gate_fails(tmp_path, capsys, monkeypatch):
    files = recorded_and_accepted(tmp_path)
    record(ADMIN, CHOSEN, greedy_model(), FakeEmbedder(), files["recordings_path"])
    monkeypatch.setattr("evals.__main__.EMBEDDING_MODEL", FakeEmbedder.model)

    assert main(gate_command(files)) == 1
    assert "task completion" in capsys.readouterr().out


def test_the_command_exits_zero_when_the_gate_passes(tmp_path, capsys, monkeypatch):
    files = recorded_and_accepted(tmp_path)
    monkeypatch.setattr("evals.__main__.EMBEDDING_MODEL", FakeEmbedder.model)

    assert main(gate_command(files)) == 0
    assert "no worse" in capsys.readouterr().out


def test_the_accept_command_writes_the_baseline_and_prints_the_scoreboard(tmp_path, capsys, monkeypatch):
    files = paths(tmp_path)
    record(ADMIN, CHOSEN, good_model(), FakeEmbedder(), files["recordings_path"])
    monkeypatch.setattr("evals.__main__.EMBEDDING_MODEL", FakeEmbedder.model)
    command = gate_command(files)
    command[1] = "accept"

    assert main(command) == 0
    assert files["baseline_path"].exists()
    assert "Task completion | 1.0000 (2 of 2)" in capsys.readouterr().out


def test_an_unknown_case_id_is_refused():
    with pytest.raises(SystemExit):
        main(["python -m evals", "gate", "--admin-url", ADMIN, "--cases", "n-999"])


def test_anything_but_record_gate_or_accept_is_refused():
    with pytest.raises(SystemExit):
        main(["python -m evals", "deploy"])
