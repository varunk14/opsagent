"""
Asking a model for something a program can use.

Week 0 settled what works: constrain the output and check it against a schema at
the boundary. Raw prose scored 0 of 5; constrained output plus validation scored
5 of 5. This module is that result made reusable, so no node has to remember it.

Nothing here touches Ollama. The model is injected, so the tests are about the
contract -- what happens when the reply is wrong, and what it costs to find out.
"""


import pytest
from pydantic import BaseModel, Field

from app.llm import ModelOutputInvalid, Reply, structured


class Answer(BaseModel):
    order_id: str
    amount_paise: int = Field(ge=0)


class FakeModel:
    """Returns the replies it was given, in order, and remembers the prompts."""

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> Reply:
        self.prompts.append(prompt)
        return Reply(
            text=self.texts.pop(0),
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100,
        )


# --- the ordinary case ------------------------------------------------------


def test_a_reply_that_fits_the_schema_is_returned_typed():
    model = FakeModel('{"order_id": "4821", "amount_paise": 360000}')

    answer, _ = structured(model, "extract it", Answer)

    assert answer == Answer(order_id="4821", amount_paise=360_000)


def test_one_good_reply_costs_one_call():
    model = FakeModel('{"order_id": "4821", "amount_paise": 360000}')

    _, replies = structured(model, "extract it", Answer)

    assert len(replies) == 1


# --- when the model gets it wrong -------------------------------------------


def test_prose_instead_of_json_is_retried():
    """
    The exact week 0 failure. The model explains itself instead of answering,
    and the explanation is worthless to a program.
    """
    model = FakeModel(
        "Sure! The order you're asking about is 4821.",
        '{"order_id": "4821", "amount_paise": 360000}',
    )

    answer, replies = structured(model, "extract it", Answer)

    assert answer.order_id == "4821"
    assert len(replies) == 2


def test_json_that_breaks_the_schema_is_retried():
    """Well-formed JSON saying the wrong thing is the harder case to notice."""
    model = FakeModel(
        '{"order_id": "4821", "amount_paise": -5}',
        '{"order_id": "4821", "amount_paise": 360000}',
    )

    answer, _ = structured(model, "extract it", Answer)

    assert answer.amount_paise == 360_000


def test_the_retry_tells_the_model_what_was_wrong():
    """
    Asking again with the identical prompt mostly gets the identical answer.
    The complaint is what makes the second attempt different from the first.
    """
    model = FakeModel("not json at all", '{"order_id": "4821", "amount_paise": 1}')

    structured(model, "extract it", Answer)

    assert "not json at all" in model.prompts[1]
    assert len(model.prompts[1]) > len(model.prompts[0])


def test_giving_up_says_which_shape_it_wanted_and_what_it_got():
    model = FakeModel("nonsense", "still nonsense", "nonsense again")

    with pytest.raises(ModelOutputInvalid) as raised:
        structured(model, "extract it", Answer, attempts=3)

    assert "Answer" in str(raised.value)
    assert "nonsense again" in str(raised.value)


def test_it_does_not_try_forever():
    model = FakeModel(*["nope"] * 10)

    with pytest.raises(ModelOutputInvalid):
        structured(model, "extract it", Answer, attempts=2)

    assert len(model.prompts) == 2


# --- what it cost to get there ----------------------------------------------


def test_every_attempt_is_counted_not_only_the_one_that_worked():
    """
    The attempts that failed were still paid for. Counting only the successful
    one understates the cost of a prompt that needs two goes, which is exactly
    the cost week 9 is trying to bring down.
    """
    model = FakeModel("rubbish", "more rubbish", '{"order_id": "1", "amount_paise": 0}')

    _, replies = structured(model, "extract it", Answer)

    assert len(replies) == 3
    assert sum(r.completion_tokens for r in replies) == 15


def test_a_failed_call_still_reports_what_it_spent():
    """Giving up is not free, and the run should be charged for it."""
    model = FakeModel("no", "no", "no")

    with pytest.raises(ModelOutputInvalid) as raised:
        structured(model, "extract it", Answer, attempts=3)

    assert sum(r.prompt_tokens for r in raised.value.replies) == 30


def test_asking_for_no_attempts_is_refused():
    with pytest.raises(ValueError, match="at least one attempt"):
        structured(FakeModel(), "extract it", Answer, attempts=0)
