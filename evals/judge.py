"""
Layer 2: a local model judges what each run did.

Layer 1 checks each run against a label. The judge answers what a label cannot: given what
the run saw -- the customer's message, the policy passages, what its tool calls returned --
is its decision grounded in that, and appropriate? It is shown what the run saw and where it
came to rest, never the label, and every part of that is fenced as data.

Small local models are poor judges: measured before this was written, two of them called a
double refund appropriate. So the board says how often the judge agrees with layer 1 beside
the judge's own score, and only that score falling, or more cases left unjudged, is a
failure. Agreement measures the judge, not the agent.
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
from evals.scoring import outcome_of, rate, score_case

TASK = "judge"
PLACEHOLDERS = frozenset({"policy", "steps", "decision", "customer_message"})
# Six short policy documents, a handful of passages from them at most.
MAX_POLICY = 12_000
MAX_DECISION = 4_000


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


def judge_prompt(case: GoldenCase, result: CaseResult) -> str:
    """What the judge is shown about one run: what the run saw and did, never the case's label."""
    seen: dict[str, Any] = json.loads(result.evidence) if result.evidence else {}
    policy = "\n".join(f"- {passage}" for passage in seen.get("policy") or []) or "none"
    steps = "\n".join(
        json.dumps({"tool": step.get("tool"), "args": step.get("args"), "result": step.get("result")}, sort_keys=True, ensure_ascii=False)
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


@dataclass(frozen=True)
class JudgeBoard:
    judged: int
    unjudged: int
    grounded: Decimal | None
    appropriate: Decimal | None
    # How often "appropriate" matches layer 1's "complete": a measure of the judge, published, not gated.
    agreement: Decimal | None


def judge_board_of(
    cases: Sequence[GoldenCase], results: Sequence[CaseResult], verdicts: Mapping[str, Verdict | None]
) -> JudgeBoard:
    """The judge's verdicts on `cases`, and how often they agree with layer 1. A case with no entry is refused by id."""
    missing = [case.id for case in cases if case.id not in verdicts]
    if missing:
        raise ValueError(f"no verdict for case(s) {', '.join(missing)}")
    by_id = {result.case_id: result for result in results}
    judged = [(case, verdict) for case in cases if (verdict := verdicts[case.id]) is not None]
    return JudgeBoard(
        judged=len(judged),
        unjudged=len(cases) - len(judged),
        grounded=rate(sum(verdict.grounded for _, verdict in judged), len(judged)),
        appropriate=rate(sum(verdict.appropriate for _, verdict in judged), len(judged)),
        agreement=rate(
            sum(verdict.appropriate == score_case(case, by_id[case.id]).completed for case, verdict in judged), len(judged)
        ),
    )


def compare_judge(current: JudgeBoard, baseline: JudgeBoard) -> list[str]:
    """Every way the judge's board is worse than its baseline. Agreement is not compared."""
    problems = []
    if current.unjudged > baseline.unjudged:
        problems.append(f"unjudged cases rose from {baseline.unjudged} to {current.unjudged}")
    for label, attribute in (("judged grounded", "grounded"), ("judged appropriate", "appropriate")):
        now, before = getattr(current, attribute), getattr(baseline, attribute)
        if before is not None and (now is None or now < before):
            problems.append(f"{label} fell from {before} to {'undefined' if now is None else now}")
    return problems
