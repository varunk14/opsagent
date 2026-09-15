"""
Layer 2: a local model judges what each run did.

Layer 1 checks each run against a label. The judge answers what a label cannot: given what
the run saw -- the customer's message, the policy passages, what its tool calls returned --
is its decision grounded in that, and appropriate? It is shown what the run saw and where it
came to rest, never the label, and every part of that is fenced as data. The customer's
message comes first and the instruction to reply comes last, so untrusted text is never the
last thing the model reads.

Small local models are poor judges: measured before this was written, two of them called a
double refund appropriate, and on the first real recording the judge called no decision
appropriate at all. So the judge gates nothing. Its board is published on the scoreboard with
how often it agrees with layer 1 beside its score; agreement measures the judge, not the agent.

The smoke cases are judged on every pull request, from recorded verdicts. Before a release,
every case is judged (layer 3), on a machine with the model.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from string import Template
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from app.graph.prompts import MAX_OBSERVATIONS, customer_message, load_template
from app.llm import Model, ModelOutputInvalid, fence_safe, structured
from evals.golden import GoldenCase, Outcome
from evals.runner import CaseResult
from evals.scoring import outcome_of, rate, score_case, shown

TASK = "judge"
PLACEHOLDERS = frozenset({"policy", "steps", "decision", "customer_message"})
# Six short policy documents, a handful of passages from them at most.
MAX_POLICY = 12_000
MAX_DECISION = 4_000
# Numbered by the database in the order rows were written, so a refund's id depends on how many
# cases paid before it. Shown to the judge, it would change the prompt -- and so its recording --
# of every later case whenever a case was added or reordered. It says nothing about the decision.
NOT_SHOWN_TO_THE_JUDGE = frozenset({"refund_id"})


class Verdict(BaseModel):
    """What the judge decided about one run. Booleans are booleans: "yes" and 1 are refused."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grounded: StrictBool
    appropriate: StrictBool
    reason: str = Field(max_length=1000)


def load_judge_template() -> str:
    text = load_template(TASK)
    found = frozenset(Template(text).get_identifiers())
    if found != PLACEHOLDERS:
        raise ValueError(f"prompts/{TASK}.txt has placeholders {sorted(found)}, expected {sorted(PLACEHOLDERS)}")
    return text


TEMPLATE = load_judge_template()


def where_it_rested(result: CaseResult) -> str:
    outcome = outcome_of(result)
    if outcome is Outcome.REFUNDED:
        return f"refunded {', '.join(str(amount) for amount in result.refunds_paise)} paise"
    if outcome is Outcome.WAITING_APPROVAL:
        return f"a refund of {result.approval_paise} paise was put to a person for approval"
    if outcome is Outcome.HANDED_OVER:
        return "handed to a person, with nothing put to them for approval"
    return "it never came to rest"


def shown_result(result: Any) -> Any:
    """A tool result as the judge sees it: without the fields the database numbered."""
    if isinstance(result, dict):
        return {key: value for key, value in result.items() if key not in NOT_SHOWN_TO_THE_JUDGE}
    return result


def judge_prompt(case: GoldenCase, result: CaseResult) -> str:
    """What the judge is shown about one run: what the run saw and did, never the case's label."""
    seen: dict[str, Any] = json.loads(result.evidence) if result.evidence else {}
    policy = "\n".join(f"- {passage}" for passage in seen.get("policy") or []) or "none"
    steps = "\n".join(
        json.dumps(
            {"tool": step.get("tool"), "args": step.get("args"), "result": shown_result(step.get("result"))},
            sort_keys=True,
            ensure_ascii=False,
        )
        for step in seen.get("steps") or []
    ) or "none"
    proposal = seen.get("proposal")
    decision = "\n".join(
        [
            f"last proposal: {json.dumps(proposal, sort_keys=True, ensure_ascii=False) if proposal else 'none'}",
            f"why the agent stopped: {seen.get('failure') or 'it did not stop early'}",
            f"where the run came to rest: {where_it_rested(result)}",
        ]
    )
    return Template(TEMPLATE).substitute(
        policy=fence_safe(policy, MAX_POLICY),
        steps=fence_safe(steps, MAX_OBSERVATIONS),
        decision=fence_safe(decision, MAX_DECISION),
        customer_message=customer_message(case.message.subject, case.message.body),
    )


def judge_case(model: Model, case: GoldenCase, result: CaseResult) -> Verdict | None:
    """The judge's verdict on one run, or None when it never answered in the verdict's shape."""
    try:
        verdict, _ = structured(model, judge_prompt(case, result), Verdict)
    except ModelOutputInvalid:
        return None
    return verdict


def chosen(cases: Sequence[GoldenCase], every_case: bool) -> list[GoldenCase]:
    """The cases the judge looks at: the smoke cases, or every case before a release."""
    return [case for case in cases if every_case or case.smoke]


def results_by_id(cases: Sequence[GoldenCase], results: Sequence[CaseResult]) -> dict[str, CaseResult]:
    """Each result by its case id. A case to be judged with no result is refused by id."""
    by_id = {result.case_id: result for result in results}
    missing = [case.id for case in cases if case.id not in by_id]
    if missing:
        raise ValueError(f"no result for case(s) {', '.join(missing)}")
    return by_id


def judge_cases(
    model: Model, cases: Sequence[GoldenCase], results: Sequence[CaseResult], every_case: bool = False
) -> dict[str, Verdict | None]:
    """The judge's verdict on every smoke case among `cases` -- or on every case -- by case id."""
    judged = chosen(cases, every_case)
    by_id = results_by_id(judged, results)
    return {case.id: judge_case(model, case, by_id[case.id]) for case in judged}


@dataclass(frozen=True)
class JudgeBoard:
    judged: int
    unjudged: int
    grounded: Decimal | None
    appropriate: Decimal | None
    # How often "appropriate" matches layer 1's "complete": a measure of the judge, not the agent.
    agreement: Decimal | None


def judge_board_of(
    cases: Sequence[GoldenCase],
    results: Sequence[CaseResult],
    verdicts: Mapping[str, Verdict | None],
    every_case: bool = False,
) -> JudgeBoard:
    """The judge's verdicts on the smoke cases -- or every case -- and how often they agree with layer 1."""
    looked_at = chosen(cases, every_case)
    missing = [case.id for case in looked_at if case.id not in verdicts]
    if missing:
        raise ValueError(f"no verdict for case(s) {', '.join(missing)}")
    by_id = results_by_id(looked_at, results)
    judged = [(case, verdict) for case in looked_at if (verdict := verdicts[case.id]) is not None]
    return JudgeBoard(
        judged=len(judged),
        unjudged=len(looked_at) - len(judged),
        grounded=rate(sum(verdict.grounded for _, verdict in judged), len(judged)),
        appropriate=rate(sum(verdict.appropriate for _, verdict in judged), len(judged)),
        agreement=rate(
            sum(verdict.appropriate == score_case(case, by_id[case.id]).completed for case, verdict in judged), len(judged)
        ),
    )


def render_judge(board: JudgeBoard, every_case: bool = False) -> str:
    """The judge's section of a scoreboard, its agreement with layer 1 beside its score."""
    rows = [
        ("Cases judged", f"{board.judged} ({board.unjudged} unjudged)"),
        ("Judged grounded", shown(board.grounded)),
        ("Judged appropriate", shown(board.appropriate)),
        ("Agreement with layer 1", shown(board.agreement)),
    ]
    heading = "## Judge (all cases)" if every_case else "## Judge (smoke cases)"
    lines = ["", heading, "", "| Measure | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines) + "\n"
