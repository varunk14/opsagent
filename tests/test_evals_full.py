"""
Layer 3, run on a machine with the model before a release: every case, judged.

`full` records every case the way `record` does -- reusing whatever is already recorded --
then has the same model judge all of them, not only the smoke cases, and writes the whole
board to evals/full.md for a reviewed commit. Its verdicts go into the same recordings, so
the smoke cases CI replays are untouched. It needs the live model; CI never runs it.
"""

import pytest

from evals.__main__ import full, record
from evals.judge import judge_board_of, judge_cases, render_judge
from tests.fakes import FakeEmbedder, ScriptedModel
from tests.test_evals_cli import ADMIN, CASES, CHOSEN, JUDGED_FAIR, good_model, paths
from tests.test_evals_judge_gate import board
from tests.test_evals_scoring import perfect


def test_every_case_can_be_judged_not_only_the_smoke_cases():
    smoke, other = CASES["n-001"], CASES["n-003"]
    results = [perfect(smoke), perfect(other)]
    model = ScriptedModel(judge=JUDGED_FAIR)

    verdicts = judge_cases(model, [smoke, other], results, every_case=True)

    assert set(verdicts) == {"n-001", "n-003"}
    assert judge_board_of([smoke, other], results, verdicts, every_case=True).judged == 2


def test_a_non_smoke_case_with_no_result_is_refused_when_every_case_is_judged():
    with pytest.raises(ValueError, match="n-003"):
        judge_cases(ScriptedModel(judge=JUDGED_FAIR), [CASES["n-003"]], [], every_case=True)


def test_the_full_board_says_every_case_was_judged():
    assert render_judge(board(), every_case=True).startswith("\n## Judge (all cases)\n")
    assert render_judge(board()).startswith("\n## Judge (smoke cases)\n")


@pytest.mark.db
def test_full_records_judges_every_case_and_writes_the_whole_board(tmp_path):
    recordings_path = paths(tmp_path)["recordings_path"]
    full_path = tmp_path / "full.md"
    model = good_model()

    full(ADMIN, CHOSEN, model, FakeEmbedder(), recordings_path, full_path)

    text = full_path.read_text()
    assert "Task completion | 1.0000 (2 of 2)" in text
    assert "## Judge (all cases)" in text
    assert "| Cases judged | 2 (0 unjudged) |" in text
    assert model.tasks().count("judge") == 2
    assert '"task": "judge"' in recordings_path.read_text()


@pytest.mark.db
def test_full_asks_the_model_nothing_that_is_already_recorded(tmp_path):
    recordings_path = paths(tmp_path)["recordings_path"]
    record(ADMIN, CHOSEN, good_model(), FakeEmbedder(), recordings_path)
    model = good_model()

    full(ADMIN, CHOSEN, model, FakeEmbedder(), recordings_path, tmp_path / "full.md")

    assert model.prompts == []
