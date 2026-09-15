"""
The judge inside the commands.

`record` asks the same model to judge each smoke case once its run has rested, so the
verdicts are recorded beside the replies they judge and replayed in CI like everything
else. `accept` writes the judge's board to a baseline of its own and a section of the
scoreboard. `gate` fails when the judge thinks less of the decisions than its baseline did,
or when there is no judge baseline to compare with. `verify` finds a hand-edited verdict
the same way it finds a hand-edited reply.
"""

from dataclasses import replace
from decimal import Decimal

import pytest

from evals.__main__ import JUDGE_BASELINE, accept, gate, record, verify
from evals.judge import JudgeBoard, judge_cases, render_judge
from evals.recording import Recordings
from tests.fakes import FakeEmbedder, ScriptedModel
from tests.test_evals_cli import ADMIN, CASES, CHOSEN, JUDGED_FAIR, good_model, paths, recorded_and_accepted
from tests.test_evals_scoring import perfect

UNFAIR = '{"grounded": true, "appropriate": false, "reason": "it should have asked a person"}'


def judged_unfairly():
    model = good_model()
    model.replies["judge"] = [UNFAIR]
    return model


def board(**changes) -> JudgeBoard:
    fields = {
        "judged": 25,
        "unjudged": 0,
        "grounded": Decimal("0.8000"),
        "appropriate": Decimal("0.6000"),
        "agreement": Decimal("0.4000"),
    }
    return JudgeBoard(**{**fields, **changes})


# --- which cases, and the board as a file ------------------------------------------------------


def test_only_the_smoke_cases_are_judged():
    smoke, other = CASES["n-001"], CASES["n-003"]
    assert (smoke.smoke, other.smoke) == (True, False)
    model = ScriptedModel(judge=JUDGED_FAIR)

    verdicts = judge_cases(model, [smoke, other], [perfect(smoke), perfect(other)])

    assert set(verdicts) == {"n-001"}
    assert model.tasks() == ["judge"]


def test_a_judge_board_round_trips_through_its_baseline():
    empty = board(judged=0, unjudged=25, grounded=None, appropriate=None, agreement=None)

    assert JudgeBoard.from_json(board().to_json()) == board()
    assert JudgeBoard.from_json(empty.to_json()) == empty


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '{"judged": 25, "unjudged": 0, "grounded": "0.8", "appropriate": "0.6"}',
        '{"judged": -1, "unjudged": 0, "grounded": "0.8", "appropriate": "0.6", "agreement": "0.4"}',
        '{"judged": 25, "unjudged": 0, "grounded": "1.5", "appropriate": "0.6", "agreement": "0.4"}',
        '{"judged": 25, "unjudged": 0, "grounded": "NaN", "appropriate": "0.6", "agreement": "0.4"}',
    ],
)
def test_a_judge_baseline_that_is_not_a_board_is_refused(text):
    with pytest.raises(ValueError):
        JudgeBoard.from_json(text)


def test_the_judge_has_a_section_of_the_scoreboard_with_its_agreement_beside_its_score():
    text = render_judge(board())

    assert text.startswith("\n## Judge (smoke cases)\n")
    assert "| Judged appropriate | 0.6000 |" in text
    assert "| Judged grounded | 0.8000 |" in text
    assert "| Agreement with layer 1 | 0.4000 |" in text
    assert "| Cases judged | 25 (0 unjudged) |" in text
    assert render_judge(board()) == text


# --- through the commands ---------------------------------------------------------------------


@pytest.mark.db
def test_accept_writes_the_judge_baseline_and_its_section_of_the_scoreboard(tmp_path):
    files = recorded_and_accepted(tmp_path)

    judged = JudgeBoard.from_json(files["judge_baseline_path"].read_text())
    assert (judged.judged, judged.unjudged, judged.appropriate) == (2, 0, Decimal("1.0000"))
    assert "## Judge (smoke cases)" in files["scoreboard_path"].read_text()


@pytest.mark.db
def test_the_gate_fails_when_the_judge_thinks_less_of_the_decisions(tmp_path):
    files = recorded_and_accepted(tmp_path)
    record(ADMIN, CHOSEN, judged_unfairly(), FakeEmbedder(), files["recordings_path"], fresh=True)

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert any("judged appropriate fell from 1.0000 to 0.0000" in problem for problem in problems)
    assert not any("task completion" in problem for problem in problems)


@pytest.mark.db
def test_the_gate_fails_without_a_judge_baseline_to_compare_with(tmp_path):
    files = recorded_and_accepted(tmp_path)
    files["judge_baseline_path"] = tmp_path / "no-judge-baseline.json"

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert any("judge baseline" in problem for problem in problems)


@pytest.mark.db
def test_a_hand_edited_verdict_is_found_by_recording_again(tmp_path):
    path = paths(tmp_path)["recordings_path"]
    record(ADMIN, CHOSEN, good_model(), FakeEmbedder(), path)
    recordings = Recordings.load(path)
    key = next(key for key, reply in recordings.replies.items() if reply.task == "judge")
    recordings.replies[key] = replace(recordings.replies[key], text=UNFAIR)
    recordings.save(path)

    problems = verify(ADMIN, CHOSEN, good_model(), FakeEmbedder(), path)

    assert any("judge" in problem and "differs" in problem for problem in problems)


@pytest.mark.db
def test_accept_leaves_the_repositorys_own_judge_baseline_alone(tmp_path):
    before = JUDGE_BASELINE.read_bytes() if JUDGE_BASELINE.exists() else None
    files = paths(tmp_path)
    record(ADMIN, CHOSEN, good_model(), FakeEmbedder(), files["recordings_path"])
    accept(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert (JUDGE_BASELINE.read_bytes() if JUDGE_BASELINE.exists() else None) == before
