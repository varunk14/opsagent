"""
Layer 3's integrity check: record again, live, and compare with what is committed.

A recording is keyed by its prompt, not by what the model said, so a hand-edited reply
would replay as if the model had said it. On a machine with the model, `verify` records
every case again into a scratch file -- at temperature 0 with a fixed seed, replies are
exact -- and reports every reply or vector that differs from the committed one, and every
prompt the committed recordings do not hold. It never changes the committed file.
"""

from dataclasses import replace

import pytest

from evals.__main__ import record, verify
from evals.recording import Recordings
from tests.fakes import FakeEmbedder
from tests.test_evals_cli import ADMIN, CHOSEN, good_model, greedy_model, paths

pytestmark = pytest.mark.db


def recorded(tmp_path):
    path = paths(tmp_path)["recordings_path"]
    record(ADMIN, CHOSEN, good_model(), FakeEmbedder(), path)
    return path


def test_a_live_recording_that_matches_the_committed_one_finds_nothing(tmp_path):
    path = recorded(tmp_path)

    assert verify(ADMIN, CHOSEN, good_model(), FakeEmbedder(), path) == []


def test_a_hand_edited_reply_is_found_by_recording_again(tmp_path):
    path = recorded(tmp_path)
    recordings = Recordings.load(path)
    key = next(key for key, reply in recordings.replies.items() if reply.task == "plan")
    recordings.replies[key] = replace(recordings.replies[key], text='{"tool": "escalate_to_human", "args": {}}')
    recordings.save(path)

    problems = verify(ADMIN, CHOSEN, good_model(), FakeEmbedder(), path)

    assert any("plan" in problem and "differs" in problem for problem in problems)


def test_a_model_that_now_answers_differently_is_found(tmp_path):
    path = recorded(tmp_path)

    problems = verify(ADMIN, CHOSEN, greedy_model(), FakeEmbedder(), path)

    assert any("differs" in problem for problem in problems)


def test_an_embedding_that_changed_is_found(tmp_path):
    path = recorded(tmp_path)

    class ShiftedEmbedder(FakeEmbedder):
        def embed(self, texts):
            return [[value + 0.5 for value in vector] for vector in super().embed(texts)]

    problems = verify(ADMIN, CHOSEN, good_model(), ShiftedEmbedder(), path)

    assert any("embedding" in problem for problem in problems)


def test_prompts_and_texts_the_committed_recordings_never_held_are_found(tmp_path):
    path = paths(tmp_path)["recordings_path"]
    record(ADMIN, CHOSEN[:1], good_model(), FakeEmbedder(), path)

    problems = verify(ADMIN, CHOSEN, good_model(), FakeEmbedder(), path)

    assert any("prompt" in problem and "not in the committed recordings" in problem for problem in problems)
    assert any("embedding" in problem and "not in the committed recordings" in problem for problem in problems)


def test_verifying_never_changes_the_committed_recordings(tmp_path):
    path = recorded(tmp_path)
    before = path.read_bytes()

    verify(ADMIN, CHOSEN, greedy_model(), FakeEmbedder(), path)

    assert path.read_bytes() == before
