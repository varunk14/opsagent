"""
The measurement later cost work will be compared against.

The eventual claim is meant to be "cost per 100 runs fell from A to B". That claim is
only worth anything if A was recorded before anything was optimised, by the same
arithmetic that will later produce B. It cannot be reconstructed afterwards, so
it is taken now.

The honest part: inference here is local, so the cash cost is nil. What is
actually measured is tokens and wall-clock. Those are turned into a dollar figure
using a rate written down in the code, and the rate is not a quote from anybody.
It does not need to be. Both sides of the comparison use the same one, so it
cancels, and what survives is the ratio -- which is the number that was being
claimed all along.

These tests cover the arithmetic. The model call itself is a network call and is
exercised by running the thing, not by the suite.
"""

from decimal import Decimal

import pytest

from app.baseline import (
    REFERENCE_RATE,
    Baseline,
    ReferenceRate,
    StepMeasurement,
    percentile,
    read_token_counts,
    summarise,
)


def measurement(prompt: int = 1000, completion: int = 100, ms: int = 500) -> StepMeasurement:
    return StepMeasurement(
        model="llama3.2", prompt_tokens=prompt, completion_tokens=completion, latency_ms=ms
    )


# --- the money ---------------------------------------------------------------


def test_cost_is_exact_decimal_arithmetic():
    """
    Every number here is summed across runs and then divided. Done in floats the
    result drifts, and the drift is indistinguishable from the improvement being
    claimed.
    """
    rate = ReferenceRate(name="test", input_per_million=Decimal(1), output_per_million=Decimal(2))

    summary = summarise([measurement(prompt=1_000_000, completion=1_000_000)], rate)

    assert summary.cost_usd == Decimal(3)
    assert isinstance(summary.cost_usd, Decimal)


def test_a_run_that_used_no_tokens_costs_nothing():
    summary = summarise([measurement(prompt=0, completion=0)], REFERENCE_RATE)

    assert summary.cost_usd == Decimal(0)


def test_cost_per_hundred_runs_scales_from_what_was_measured():
    summary = summarise([measurement(), measurement(), measurement(), measurement()], REFERENCE_RATE)

    assert summary.cost_per_100_runs == summary.cost_usd / 4 * 100


def test_changing_the_rate_moves_both_sides_by_the_same_factor():
    """
    Why an invented rate is acceptable. Double it and every cost doubles, so the
    before-and-after ratio is untouched -- and the ratio is the claim.
    """
    single = ReferenceRate(
        name="single", input_per_million=Decimal(1), output_per_million=Decimal(1)
    )
    double = ReferenceRate(
        name="double", input_per_million=Decimal(2), output_per_million=Decimal(2)
    )

    cheap = summarise([measurement()], single)
    dear = summarise([measurement()], double)

    assert dear.cost_usd == cheap.cost_usd * 2


# --- latency -----------------------------------------------------------------


def test_percentiles_use_nearest_rank():
    values = [10, 20, 30, 40, 50]

    assert percentile(values, 50) == 30
    assert percentile(values, 95) == 50


def test_percentiles_do_not_care_what_order_they_arrive_in():
    assert percentile([50, 10, 40, 20, 30], 50) == 30


def test_a_single_measurement_is_its_own_percentile():
    assert percentile([42], 95) == 42


def test_the_summary_reports_both_percentiles():
    summary = summarise([measurement(ms=ms) for ms in (100, 200, 300, 400, 900)], REFERENCE_RATE)

    assert summary.p50_ms == 300
    assert summary.p95_ms == 900


# --- refusing to report a number that means nothing --------------------------


def test_summarising_nothing_is_refused():
    """
    An empty baseline would report a cost of zero and a latency of zero, and
    a later comparison would accept it happily. Refuse instead.
    """
    with pytest.raises(ValueError, match="no measurements"):
        summarise([], REFERENCE_RATE)


def test_the_summary_counts_what_went_into_it():
    summary = summarise([measurement(), measurement()], REFERENCE_RATE)

    assert summary.runs == 2
    assert summary.tokens_per_run == Decimal(1100)


def test_the_rate_used_is_carried_with_the_result(tmp_path):
    """A cost figure without the rate that produced it cannot be compared."""
    summary = summarise([measurement()], REFERENCE_RATE)

    assert isinstance(summary, Baseline)
    assert summary.rate is REFERENCE_RATE


# --- reading what the model reported -----------------------------------------


def test_token_counts_are_read_from_the_response():
    counts = read_token_counts({"prompt_eval_count": 98, "eval_count": 225})

    assert counts == (98, 225)


@pytest.mark.parametrize("missing", ["prompt_eval_count", "eval_count"])
def test_a_response_without_a_token_count_is_refused(missing):
    """
    The finding this function exists for.

    Reaching for the field with a default of zero turns any response that does
    not carry it -- an error body returned as 200, a renamed field in a later
    Ollama, a proxy that strips it -- into a run that apparently used no tokens
    and cost nothing. Those zeros fold into the totals, quietly deflating the
    figure later work will be measured against, and nothing anywhere says so.
    """
    payload = {"prompt_eval_count": 98, "eval_count": 225}
    del payload[missing]

    with pytest.raises(ValueError, match=missing):
        read_token_counts(payload)


def test_a_run_that_genuinely_produced_nothing_is_still_readable():
    """Zero is a legitimate answer when the model actually says zero."""
    assert read_token_counts({"prompt_eval_count": 0, "eval_count": 0}) == (0, 0)


# --- refusing to report a number that means nothing, part two -----------------


def test_a_percentile_of_nothing_is_refused():
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)


def test_a_baseline_of_no_runs_cannot_be_built_at_all():
    """
    summarise refuses an empty list, but Baseline is public and could be built
    directly. The invariant belongs with the type that depends on it.
    """
    with pytest.raises(ValueError, match="no runs"):
        Baseline(
            runs=0,
            model="llama3.2",
            rate=REFERENCE_RATE,
            prompt_tokens=0,
            completion_tokens=0,
            latencies_ms=(),
        )


def test_tokens_per_run_is_the_exact_mean():
    """
    1371 tokens across 4 runs is 342.75, not 342. Floor division sits next to
    exact decimal costs in the report and biases downward every time, which does
    not cancel between a before and an after taken over different run counts.
    """
    summary = summarise(
        [measurement(prompt=300, completion=43), measurement(prompt=300, completion=44)],
        REFERENCE_RATE,
    )

    assert summary.tokens_per_run == Decimal("343.5")
