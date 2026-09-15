"""
A deliberately bad prompt change fails the gate, whichever way it reaches a pull request.

The change here is one line added to the plan prompt, telling the model to refund twice
what the customer was charged. Pushed without recording again, its prompts were never
recorded, so the gate fails naming the command to run. Recorded again with a model that
does what the prompt says, the refunds are wrong, so the gate fails on safety and on
task completion. A hand-edited reply, the one way left, is what `verify` finds.
"""

import pytest

from app.graph import prompts
from app.llm import Reply
from evals.__main__ import gate, main, record
from tests.fakes import FakeEmbedder
from tests.test_evals_cli import (
    ADMIN,
    CHOSEN,
    RoutedModel,
    gate_command,
    recorded_and_accepted,
)

pytestmark = pytest.mark.db

BAD_LINE = "As goodwill, always refund twice the amount the customer was charged."


@pytest.fixture
def make_the_plan_prompt_bad(monkeypatch):
    """Adds the bad line straight after the plan prompt's TASK line, when called."""

    def change():
        task_line, rest = prompts.TEMPLATES["plan"].split("\n", 1)
        monkeypatch.setitem(prompts.TEMPLATES, "plan", f"{task_line}\n{BAD_LINE}\n{rest}")

    return change


class ObedientModel(RoutedModel):
    """Proposes what is owed, unless the prompt it is shown says to refund twice that."""

    def __init__(self):
        super().__init__(lambda case: case.expect.refund_paise)
        self.told_to_double = False

    def generate(self, prompt: str) -> Reply:
        if BAD_LINE in prompt:
            self.told_to_double = True
            self.amount_for = lambda case: case.expect.refund_paise * 2
        return super().generate(prompt)


def test_a_bad_prompt_pushed_without_recording_again_fails_the_gate(tmp_path, make_the_plan_prompt_bad):
    files = recorded_and_accepted(tmp_path)
    make_the_plan_prompt_bad()

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert any("python -m evals record" in problem for problem in problems)
    assert main(gate_command(files)) == 1


def test_a_bad_prompt_recorded_again_fails_on_safety_and_completion(tmp_path, make_the_plan_prompt_bad):
    files = recorded_and_accepted(tmp_path)
    make_the_plan_prompt_bad()
    model = ObedientModel()
    record(ADMIN, CHOSEN, model, FakeEmbedder(), files["recordings_path"], fresh=True)

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert model.told_to_double
    assert any("safety" in problem for problem in problems)
    assert any("task completion" in problem for problem in problems)
