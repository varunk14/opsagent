"""
The judge inside the commands: published, not gated.

`record` asks the same model to judge each smoke case once its run has rested, so the
verdicts are recorded beside the replies they judge and replayed in CI like everything
else. `accept` writes the judge's board as a section of the scoreboard. A small local judge
is not trustworthy yet -- on the first real recording it called no decision appropriate --
so the gate never fails on its score. A changed verdict still shows: the committed
scoreboard must say what the recordings score. `verify` finds a hand-edited verdict the
same way it finds a hand-edited reply.
"""

from dataclasses import replace
from decimal import Decimal

import pytest

from evals.__main__ import accept, gate, record, verify
from evals.judge import JudgeBoard, judge_cases, render_judge
from evals.recording import Recordings
from tests.fakes import FakeEmbedder, ScriptedModel
from tests.test_evals_cli import (
    ADMIN,
    CASES,
    CHOSEN,
    JUDGED_FAIR,
    good_model,
    paths,
    recorded_and_accepted,
)
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


# --- which cases, and the judge's section of the scoreboard ------------------------------------


def test_only_the_smoke_cases_are_judged():
    smoke, other = CASES["n-001"], CASES["n-003"]
    assert (smoke.smoke, other.smoke) == (True, False)
    model = ScriptedModel(judge=JUDGED_FAIR)

    verdicts = judge_cases(model, [smoke, other], [perfect(smoke), perfect(other)])

    assert set(verdicts) == {"n-001"}
    assert model.tasks() == ["judge"]


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
def test_accept_writes_the_judges_section_of_the_scoreboard(tmp_path):
    files = recorded_and_accepted(tmp_path)

    text = files["scoreboard_path"].read_text()
    assert "## Judge (smoke cases)" in text
    assert "| Cases judged | 2 (0 unjudged) |" in text
    assert "| Judged appropriate | 1.0000 |" in text


@pytest.mark.db
def test_the_gate_never_fails_on_the_judges_score(tmp_path):
    files = recorded_and_accepted(tmp_path)
    record(ADMIN, CHOSEN, judged_unfairly(), FakeEmbedder(), files["recordings_path"], fresh=True)
    # The scoreboard is brought up to date; the layer 1 baseline is left as it was accepted.
    accept(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **{**files, "baseline_path": tmp_path / "ignored.json"})

    assert "| Judged appropriate | 0.0000 |" in files["scoreboard_path"].read_text()
    assert gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files) == []


@pytest.mark.db
def test_a_changed_verdict_still_shows_because_the_scoreboard_must_match(tmp_path):
    files = recorded_and_accepted(tmp_path)
    record(ADMIN, CHOSEN, judged_unfairly(), FakeEmbedder(), files["recordings_path"], fresh=True)

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert len(problems) == 1
    assert "scoreboard" in problems[0]


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
