"""
Retry policy itself, independent of refunds.

A retry loop that never gives up is its own outage. These pin down the boundary.
"""

import pytest

from experiments import prevent_duplicate_refunds as refunds


def test_gives_up_after_max_attempts_and_reraises():
    attempts = []

    def always_times_out(*_args, **_kwargs):
        attempts.append(1)
        raise refunds.NetworkTimeout("gateway down")

    with pytest.raises(refunds.NetworkTimeout):
        refunds.call_with_retry(always_times_out, max_attempts=3)

    assert len(attempts) == 3, "must stop at max_attempts, not retry forever"


def test_succeeds_without_retrying_when_the_call_works():
    attempts = []

    def works_first_time(*_args, **_kwargs):
        attempts.append(1)
        return {"ok": True}

    result = refunds.call_with_retry(works_first_time)

    assert result == {"ok": True}
    assert len(attempts) == 1, "must not retry a call that succeeded"
