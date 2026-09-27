"""Checks of nested, source-stratified cohort sampling."""

import pandas as pd
import pytest

from ecg_experiment.clean_cohorts import nested_quotas, nested_samples


def _candidates():
    counts = {"mimic": 700, "georgia": 200, "chapman_shaoxing": 100}
    rows = [(f"{source}:{i}", source) for source, count in counts.items() for i in range(count)]
    return pd.DataFrame(rows, columns=["record_id", "source"])


def test_smaller_cohorts_are_subsets_with_the_same_mix():
    samples, quotas = nested_samples(_candidates(), [100, 200, 400], seed=3, mimic_fraction=0.7)
    ids = {size: set(frame["record_id"]) for size, frame in samples.items()}
    assert ids[100] < ids[200] < ids[400]
    assert {size: len(frame) for size, frame in samples.items()} == {100: 100, 200: 200, 400: 400}
    assert quotas[100] == {"mimic": 70, "georgia": 20, "chapman_shaoxing": 10}
    assert quotas[400]["mimic"] == 280


def test_sampling_is_deterministic_and_seed_dependent():
    first, _ = nested_samples(_candidates(), [100], seed=3, mimic_fraction=0.7)
    again, _ = nested_samples(_candidates(), [100], seed=3, mimic_fraction=0.7)
    other, _ = nested_samples(_candidates(), [100], seed=4, mimic_fraction=0.7)
    assert first[100]["record_id"].tolist() == again[100]["record_id"].tolist()
    assert set(first[100]["record_id"]) != set(other[100]["record_id"])


def test_sampling_does_not_depend_on_input_row_order():
    candidates = _candidates()
    shuffled = candidates.sample(frac=1, random_state=0)
    first, _ = nested_samples(candidates, [100], seed=3, mimic_fraction=0.7)
    second, _ = nested_samples(shuffled, [100], seed=3, mimic_fraction=0.7)
    assert first[100]["record_id"].tolist() == second[100]["record_id"].tolist()


def test_quotas_that_cannot_nest_raise():
    with pytest.raises(ValueError, match="Insufficient"):
        nested_quotas({"mimic": 10, "georgia": 5}, [5, 20], mimic_fraction=0.7)
