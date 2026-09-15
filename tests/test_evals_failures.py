"""
The failure mix on the golden set, and its history.

The worker's classifier names a failure from what the run left behind. On the golden set the
label says what should have happened, so two more kinds can be named: a refund paid when a
person should have decided, or a case handed over that should have completed, is a wrong
escalation; the right outcome with the wrong money is tool misuse. A case that did what its
label says has no category, whatever the classifier's rules would say about it.

The mix -- how many of the 150 failed each way -- is on the scoreboard and in the baseline,
and every `accept` appends one line to evals/history.jsonl: the date, the code, the prompt
versions, the completion, the violations and the mix. That file is the trend: each accepted
baseline is a point, and the gate checks it ends with the baseline it is comparing against.
"""

import json
from dataclasses import replace

import pytest
from evals.failures import failure_mix, failure_of

from evals.__main__ import accept, gate
from evals.golden import load_cases
from evals.scoring import Scoreboard, render_markdown
from tests.fakes import FakeEmbedder
from tests.test_evals_cli import (
    ADMIN,
    CHOSEN,
    good_model,
    paths,
    record,
    recorded_and_accepted,
)
from tests.test_evals_scoring import baseline_with, full_board, perfect

CASES = {case.id: case for case in load_cases()}
REFUNDED = CASES["n-001"]
WAITING = CASES["n-021"]
HANDED_OVER = CASES["n-041"]
SIX = ["hallucinated_field", "tool_misuse", "loop", "context_overflow", "wrong_escalation", "drift"]


def looked_up(case, *charges: int) -> str:
    """Evidence of one lookup of the case's order, returning these charges."""
    order = case.expect.order_id
    return json.dumps(
        {
            "policy": [],
            "steps": [
                {"tool": "get_order", "args": {"order_id": order}, "result": {"order_id": order, "charges_paise": list(charges)}}
            ],
            "proposal": {"tool": "escalate_to_human", "args": {"reason": "a person should look"}},
            "failure": None,
        },
        sort_keys=True,
    )


# --- one case ---------------------------------------------------------------------------------


def test_a_case_that_did_what_its_label_says_has_no_category():
    assert failure_of(REFUNDED, perfect(REFUNDED)) is None
    assert failure_of(HANDED_OVER, perfect(HANDED_OVER)) is None


def test_a_refund_paid_when_a_person_should_have_decided_is_a_wrong_escalation():
    paid = replace(perfect(HANDED_OVER), status="done", refunds_paise=(24990,), evidence=looked_up(HANDED_OVER, 24990))

    assert failure_of(HANDED_OVER, paid) == "wrong_escalation"


def test_a_case_handed_over_that_should_have_completed_is_a_wrong_escalation():
    handed_over = replace(
        perfect(REFUNDED),
        status="waiting_approval",
        refunds_paise=(),
        approval_paise=None,
        evidence=looked_up(REFUNDED, 49900, 49900),
    )

    assert failure_of(REFUNDED, handed_over) == "wrong_escalation"


def test_a_refund_queued_for_approval_that_should_have_been_handed_over_is_a_wrong_escalation():
    """Found in Unit 0: 21 change-of-mind and cancellation refunds sat in the approval queue instead of with a person."""
    queued = replace(perfect(HANDED_OVER), approval_paise=24990, evidence=looked_up(HANDED_OVER, 24990))

    assert failure_of(HANDED_OVER, queued) == "wrong_escalation"


def test_the_right_outcome_with_the_wrong_money_is_tool_misuse():
    twice = replace(perfect(REFUNDED), refunds_paise=(49900, 49900), evidence=looked_up(REFUNDED, 49900, 49900))

    assert failure_of(REFUNDED, twice) == "tool_misuse"


def test_what_the_classifier_names_from_the_run_itself_comes_first():
    looped = replace(
        perfect(REFUNDED),
        status="waiting_approval",
        refunds_paise=(),
        failure="plan: repeated an earlier step",
        evidence=looked_up(REFUNDED, 49900, 49900),
    )

    assert failure_of(REFUNDED, looped) == "loop"


# --- the mix ----------------------------------------------------------------------------------


def test_the_mix_counts_every_category_in_a_fixed_order_with_zeros():
    cases = [REFUNDED, WAITING, HANDED_OVER]
    results = [
        replace(perfect(REFUNDED), status="waiting_approval", refunds_paise=(), failure="plan: repeated an earlier step"),
        perfect(WAITING),
        replace(perfect(HANDED_OVER), status="done", refunds_paise=(24990,), evidence=looked_up(HANDED_OVER, 24990)),
    ]

    mix = failure_mix(cases, results)

    assert list(mix) == SIX
    assert mix == {"hallucinated_field": 0, "tool_misuse": 0, "loop": 1, "context_overflow": 0, "wrong_escalation": 1, "drift": 0}


def test_the_scoreboard_carries_the_mix_and_shows_it():
    board = full_board()

    assert board.failure_mix == dict.fromkeys(SIX, 0)
    text = render_markdown(board)
    assert "## Failure mix" in text
    assert "| loop | 0 |" in text
    assert Scoreboard.from_json(board.to_json()) == board


@pytest.mark.parametrize(
    "mix",
    [
        {"loop": 1},
        {**dict.fromkeys(SIX, 0), "loop": -1},
        {**dict.fromkeys(SIX, 0), "loop": "1"},
        {**dict.fromkeys(SIX, 0), "sloth": 1},
        [],
    ],
)
def test_a_baseline_whose_mix_is_not_the_six_counts_is_refused(mix):
    with pytest.raises(ValueError, match="failure_mix"):
        Scoreboard.from_json(baseline_with(failure_mix=mix))


def test_a_baseline_whose_mix_exceeds_its_failures_is_refused():
    """150 cases, 150 complete, yet a failure counted: two fields a pull request can edit apart."""
    with pytest.raises(ValueError, match="failure_mix"):
        Scoreboard.from_json(baseline_with(failure_mix={**dict.fromkeys(SIX, 0), "loop": 1}))


# --- the history --------------------------------------------------------------------------------


@pytest.mark.db
def test_accept_appends_one_line_of_history_per_accepted_baseline(tmp_path):
    files = recorded_and_accepted(tmp_path)
    accept(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    lines = [json.loads(line) for line in files["history_path"].read_text().splitlines()]

    assert len(lines) == 2
    assert (lines[-1]["cases"], lines[-1]["completed"], lines[-1]["safety_violations"]) == (2, 2, 0)
    assert lines[-1]["failure_mix"] == dict.fromkeys(SIX, 0)
    assert set(lines[-1]["prompt_versions"]) == {"classify", "extract", "plan"}
    assert lines[-1]["golden_sha256"] == json.loads(files["baseline_path"].read_text())["golden_sha256"]
    assert lines[-1]["accepted_on"] >= lines[0]["accepted_on"]


@pytest.mark.db
def test_the_gate_passes_when_the_history_ends_with_the_baseline(tmp_path):
    files = recorded_and_accepted(tmp_path)

    assert gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files) == []


@pytest.mark.db
def test_the_gate_fails_when_the_history_does_not_end_with_the_baseline(tmp_path):
    files = recorded_and_accepted(tmp_path)
    history = files["history_path"]
    last = json.loads(history.read_text().splitlines()[-1])
    last["completed"] = 1
    history.write_text(json.dumps(last) + "\n")

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert any("history" in problem for problem in problems)


@pytest.mark.db
def test_the_gate_fails_without_a_history_to_show_the_trend(tmp_path):
    files = paths(tmp_path)
    record(ADMIN, CHOSEN, good_model(), FakeEmbedder(), files["recordings_path"])
    accept(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)
    files["history_path"].unlink()

    problems = gate(ADMIN, CHOSEN, embedding_model=FakeEmbedder.model, **files)

    assert any("history" in problem and "python -m evals accept" in problem for problem in problems)
