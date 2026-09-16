"""
The failure taxonomy on the golden set: the worker's rules, plus what only a label can say.

`app.failures.classify` names a failure from what the run left behind, with no label. Here
the label says what should have happened, so two more kinds can be named once the run's own
evidence has been tried: a refund paid when a person should have decided, or a case handed
over (or queued for approval) that should have ended elsewhere, is a wrong escalation; the
right outcome with the wrong money is tool misuse -- right intent, wrong arguments. A case
that did what its label says has no category, whatever the rules would say about it.

Drift -- the same case, the same prompts and model, a different outcome -- cannot show in a
replay, which is deterministic by construction. It shows when the model is asked again, live:
`python -m evals verify` reports every recorded reply that differs. So nothing here assigns
drift; the category exists for that report.
"""

import json
from collections.abc import Sequence

from app.failures import FailureCategory, RunRest, classify
from evals.golden import GoldenCase
from evals.runner import CaseResult
from evals.scoring import outcome_of, score_case


def rest_for(case: GoldenCase, result: CaseResult) -> RunRest:
    """What a golden run left behind, in the shape the worker's classifier reads."""
    seen = json.loads(result.evidence) if result.evidence else {}
    waiting = result.status == "waiting_approval" and result.approval_paise is not None
    return RunRest(
        status=result.status,
        failure_class=result.failure_class,
        failure=result.failure,
        steps=list(seen.get("steps") or []),
        proposal=seen.get("proposal"),
        extraction={"order_id": result.order_id, "amount_paise": result.stated_amount_paise},
        refunds_paise=result.refunds_paise,
        approval_paise=result.approval_paise,
        # Nobody decides during an evaluation: an approval is pending or there is none.
        approval_status="pending" if waiting else None,
        message_text=f"{case.message.subject or ''}\n\n{case.message.body}",
        policy=tuple(seen.get("policy") or []),
    )


def failure_of(case: GoldenCase, result: CaseResult) -> FailureCategory | None:
    """One category for a golden case that did not do what its label says; None for one that did."""
    if score_case(case, result).completed:
        return None
    named = classify(rest_for(case, result))
    if named is not None:
        return named
    if outcome_of(result) is case.expect.outcome:
        return FailureCategory.TOOL_MISUSE
    return FailureCategory.WRONG_ESCALATION


def failure_mix(cases: Sequence[GoldenCase], results: Sequence[CaseResult]) -> dict[str, int]:
    """How many cases failed each way, every category present, in the taxonomy's order."""
    by_id = {result.case_id: result for result in results}
    mix = {category.value: 0 for category in FailureCategory}
    for case in cases:
        category = failure_of(case, by_id[case.id])
        if category is not None:
            mix[category.value] += 1
    return mix
