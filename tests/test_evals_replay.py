"""
Record on this machine, replay anywhere.

A real model runs here, where there is a GPU; CI has none. So every model reply and
every embedding is recorded once, keyed by a hash of the model and the exact text it
was given, and replayed later through the real driver, executor, guardrail and
Postgres. Replay never falls back to a live call:

- a prompt that was never recorded -- because a prompt file changed, or a case did --
  stops the run with RecordingMissing, naming the task and how to record it again;
- a reply recorded for another model is not used;
- vectors come back exactly, bit for bit, so retrieval ranks passages the same way;
- the recordings file holds hashes of prompts, never the prompts, and saving it twice
  gives the same bytes, so a diff shows only what really changed;
- replay opens no network connection.

A scripted golden case is run through the real driver while being recorded, read back
as a CaseResult, and then replayed from the recording on a second database: the two
results are identical.
"""

import socket
from decimal import Decimal
from uuid import uuid4

import psycopg
import pytest

from app.llm import Reply
from evals.golden import load_cases
from evals.recording import (
    RecordedEmbedder,
    RecordedModel,
    RecordingEmbedder,
    RecordingMissing,
    RecordingModel,
    Recordings,
)
from evals.runner import CaseResult, read_back, run_cases
from tests.fakes import (
    CLASSIFIED_DUPLICATE,
    FakeEmbedder,
    ScriptedModel,
    proposed_refund,
)

CASES = {case.id: case for case in load_cases()}
CLASSIFY_PROMPT = "TASK: classify\nWhat does this customer want?"


def recorded(text: str = '{"ok": true}', model: str = "llama3.1:8b") -> Recordings:
    recordings = Recordings()
    RecordingModel(ScriptedModel(classify=text), recordings, model=model).generate(CLASSIFY_PROMPT)
    return recordings


# --- replies --------------------------------------------------------------------------------


def test_a_recorded_reply_is_given_back_with_its_token_counts():
    reply = RecordedModel(recorded('{"intent": "other"}'), model="llama3.1:8b").generate(CLASSIFY_PROMPT)

    assert reply == Reply(text='{"intent": "other"}', prompt_tokens=10, completion_tokens=5, latency_ms=1)


def test_a_prompt_never_recorded_is_refused_naming_its_task_and_how_to_record_it():
    replay = RecordedModel(recorded(), model="llama3.1:8b")

    with pytest.raises(RecordingMissing, match=r"plan.*python -m evals record"):
        replay.generate("TASK: plan\nPropose one tool call.")


def test_a_reply_recorded_for_another_model_is_not_used():
    replay = RecordedModel(recorded(model="llama3.2"), model="llama3.1:8b")

    with pytest.raises(RecordingMissing):
        replay.generate(CLASSIFY_PROMPT)


def test_recording_passes_the_real_reply_through():
    recordings = Recordings()
    inner = ScriptedModel(classify=CLASSIFIED_DUPLICATE)

    reply = RecordingModel(inner, recordings, model="llama3.1:8b").generate(CLASSIFY_PROMPT)

    assert reply.text == CLASSIFIED_DUPLICATE
    assert inner.prompts == [CLASSIFY_PROMPT]


def test_the_model_name_is_the_one_traces_and_costs_report():
    assert RecordedModel(Recordings(), model="llama3.1:8b").model == "llama3.1:8b"
    assert RecordingModel(ScriptedModel(), Recordings(), model="llama3.1:8b").model == "llama3.1:8b"


# --- embeddings -----------------------------------------------------------------------------


def test_vectors_come_back_exactly_as_they_were_recorded(tmp_path):
    vector = [0.1, 1 / 3, -2.5e-300, 123456.789, 0.0]
    recordings = Recordings()
    inner = FakeEmbedder()
    inner.embed = lambda texts: [vector for _ in texts]  # type: ignore[method-assign]
    RecordingEmbedder(inner, recordings).embed(["search_query: charged twice"])
    path = tmp_path / "recordings.jsonl"
    recordings.save(path)

    replayed = RecordedEmbedder(Recordings.load(path), model=inner.model).embed(["search_query: charged twice"])

    assert replayed == [vector]


def test_an_embedding_never_recorded_is_refused():
    with pytest.raises(RecordingMissing, match="embedding"):
        RecordedEmbedder(Recordings(), model="nomic-embed-text").embed(["search_query: anything"])


def test_the_embedder_reports_the_model_ingest_compares_against():
    assert RecordedEmbedder(Recordings(), model="nomic-embed-text").model == "nomic-embed-text"
    assert RecordingEmbedder(FakeEmbedder(), Recordings()).model == FakeEmbedder.model


# --- the recordings file --------------------------------------------------------------------


def test_saving_gives_the_same_bytes_whatever_order_things_were_recorded_in(tmp_path):
    first, second = Recordings(), Recordings()
    prompts = ["TASK: classify\none", "TASK: extract\ntwo", "TASK: plan\nthree"]
    for prompt in prompts:
        RecordingModel(ScriptedModel(classify="{}", extract="{}", plan="{}"), first, model="m").generate(prompt)
    for prompt in reversed(prompts):
        RecordingModel(ScriptedModel(classify="{}", extract="{}", plan="{}"), second, model="m").generate(prompt)

    first.save(tmp_path / "a.jsonl")
    second.save(tmp_path / "b.jsonl")
    Recordings.load(tmp_path / "a.jsonl").save(tmp_path / "c.jsonl")

    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes() == (tmp_path / "c.jsonl").read_bytes()


def test_the_file_holds_hashes_of_prompts_never_the_prompts(tmp_path):
    secretish = "TASK: classify\nPriya's message about order 4821 and her card"
    recordings = Recordings()
    RecordingModel(ScriptedModel(classify="{}"), recordings, model="m").generate(secretish)
    RecordingEmbedder(FakeEmbedder(), recordings).embed(["search_query: Priya's card"])
    path = tmp_path / "recordings.jsonl"

    recordings.save(path)

    text = path.read_text()
    assert "Priya" not in text
    assert "4821" not in text


def test_a_missing_recordings_file_is_an_empty_recording(tmp_path):
    with pytest.raises(RecordingMissing):
        RecordedModel(Recordings.load(tmp_path / "absent.jsonl"), model="m").generate(CLASSIFY_PROMPT)


def test_replay_opens_no_network_connection(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("replay tried to open a connection")

    recordings = recorded()
    RecordingEmbedder(FakeEmbedder(), recordings).embed(["search_query: x"])
    monkeypatch.setattr(socket.socket, "connect", refuse)

    RecordedModel(recordings, model="llama3.1:8b").generate(CLASSIFY_PROMPT)
    RecordedEmbedder(recordings, model=FakeEmbedder.model).embed(["search_query: x"])


# --- a golden case through the real driver ---------------------------------------------------


def script_for(order_id: str, refund_paise: int) -> ScriptedModel:
    extracted = f'{{"order_id": "{order_id}", "amount_paise": null, "reason": "charged twice"}}'
    lookup = f'{{"tool": "get_order", "args": {{"order_id": "{order_id}"}}, "confidence": 0.9, "reasoning": "look it up"}}'
    return ScriptedModel(
        classify=CLASSIFIED_DUPLICATE, extract=extracted, plan=[lookup, proposed_refund(refund_paise, "0.9", order_id)]
    )


@pytest.mark.db
def test_a_case_runs_through_the_real_driver_and_is_read_back(fresh_database):
    case = CASES["n-001"]
    recordings = Recordings()
    model = RecordingModel(script_for(case.expect.order_id, case.expect.refund_paise), recordings, model="llama3.1:8b")

    (result,) = run_cases(fresh_database, [case], model, RecordingEmbedder(FakeEmbedder(), recordings))

    assert isinstance(result, CaseResult)
    assert (result.case_id, result.status, result.failure) == ("n-001", "done", None)
    assert result.tools == ("get_order", "issue_refund")
    assert result.refunds_paise == (case.expect.refund_paise,)
    assert result.approval_paise is None
    assert (result.intent, result.order_id) == ("duplicate_charge", case.expect.order_id)
    assert result.model_calls == 4
    assert result.cost_usd > Decimal(0)


@pytest.mark.db
def test_an_over_limit_case_is_read_back_waiting_with_the_amount_put_to_approval(fresh_database):
    case = CASES["n-021"]
    assert case.category == "duplicate_over_limit"
    model = script_for(case.expect.order_id, case.expect.refund_paise)

    (result,) = run_cases(fresh_database, [case], model, FakeEmbedder())

    assert result.status == "waiting_approval"
    assert result.refunds_paise == ()
    assert result.approval_paise == case.expect.refund_paise


@pytest.mark.db
def test_replaying_a_recording_reproduces_the_same_results(fresh_database, empty_database):
    cases = [CASES["n-001"], CASES["n-021"]]
    recordings = Recordings()
    live = [
        run_cases(
            fresh_database,
            [case],
            RecordingModel(script_for(case.expect.order_id, case.expect.refund_paise), recordings, model="llama3.1:8b"),
            RecordingEmbedder(FakeEmbedder(), recordings),
        )[0]
        for case in cases
    ]

    replayed = run_cases(
        empty_database, cases, RecordedModel(recordings, model="llama3.1:8b"), RecordedEmbedder(recordings, model=FakeEmbedder.model)
    )

    assert replayed == live


@pytest.mark.db
def test_a_case_whose_prompts_were_never_recorded_stops_with_recording_missing(fresh_database):
    recordings = Recordings()
    first = CASES["n-001"]
    run_cases(
        fresh_database,
        [first],
        RecordingModel(script_for(first.expect.order_id, first.expect.refund_paise), recordings, model="llama3.1:8b"),
        RecordingEmbedder(FakeEmbedder(), recordings),
    )

    with pytest.raises(RecordingMissing):
        run_cases(
            fresh_database,
            [CASES["n-002"]],
            RecordedModel(recordings, model="llama3.1:8b"),
            RecordedEmbedder(recordings, model=FakeEmbedder.model),
        )


@pytest.mark.db
def test_a_handed_over_case_reads_back_why_and_the_amount_the_customer_stated(fresh_database):
    case = CASES["n-002"]
    assert case.expect.stated_amount_paise is not None
    order_id = case.expect.order_id
    extracted = (
        f'{{"order_id": "{order_id}", "amount_paise": {case.expect.stated_amount_paise}, "reason": "charged twice"}}'
    )
    lookup = f'{{"tool": "get_order", "args": {{"order_id": "{order_id}"}}, "confidence": 0.9, "reasoning": "look it up"}}'
    model = ScriptedModel(classify=CLASSIFIED_DUPLICATE, extract=extracted, plan=[lookup, lookup])

    (result,) = run_cases(fresh_database, [case], model, FakeEmbedder())

    assert result.status == "waiting_approval"
    assert result.failure is not None and "repeated an earlier step" in result.failure
    assert result.stated_amount_paise == case.expect.stated_amount_paise
    assert (result.refunds_paise, result.approval_paise) == ((), None)


def test_blank_lines_in_a_recordings_file_are_skipped(tmp_path):
    path = tmp_path / "recordings.jsonl"
    recorded('{"intent": "other"}').save(path)
    path.write_text("\n  \n" + path.read_text() + "\n\n")

    reply = RecordedModel(Recordings.load(path), model="llama3.1:8b").generate(CLASSIFY_PROMPT)

    assert reply.text == '{"intent": "other"}'


@pytest.mark.db
def test_a_run_that_is_not_in_the_database_is_refused_naming_its_case(fresh_database):
    with psycopg.connect(fresh_database) as connection, pytest.raises(LookupError, match="n-001"):
        read_back(connection, "n-001", uuid4())
