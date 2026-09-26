"""Check reproducible CPC subset quotas and exposure cycling."""

import numpy as np

from ecg_experiment.cpc_subset25 import exposure_order, select_indices


def test_source_quota_and_repeatable_selection() -> None:
    """A subset preserves source proportions and includes each row once."""
    sources = ["a"] * 70 + ["b"] * 20 + ["c"] * 10
    first = select_indices(sources, 25, 18046)
    second = select_indices(sources, 25, 18046)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 25
    selected = [sources[index] for index in first]
    assert {name: selected.count(name) for name in ("a", "b", "c")} == {
        "a": 18, "b": 5, "c": 2,
    }


def test_exposure_order_visits_each_member_before_repeating() -> None:
    """Every full cycle contains the subset exactly once."""
    selected = np.asarray([7, 11, 13])
    order = exposure_order(selected, 8, 18047)
    assert len(order) == 8
    assert set(order[:3]) == set(selected)
    assert set(order[3:6]) == set(selected)
    assert np.array_equal(order, exposure_order(selected, 8, 18047))
