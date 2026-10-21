import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_ludb_coherence_audit as audit


def synthetic_signal():
    return np.tile(np.sin(np.arange(5000) * .018).astype(np.float32), (8, 1))


def synthetic_rows(cohort, count, offset):
    return [{"cohort": cohort, "index": index,
             "coherence": {"native_jaccard": .5 + offset, "shifted_jaccard": .2,
                           "difference": .3 + offset},
             "activity_occupancy": [.1] * 8, "empty_activity_masks": 0}
            for index in range(count)]


def test_robust_transform_preserves_power_of_two_scale():
    raw = synthetic_signal()
    first, metadata = audit.standardize(raw)
    second, _ = audit.standardize(raw * 8)
    np.testing.assert_array_equal(first, second)
    assert len(metadata["median"]) == len(metadata["q95_minus_q05"]) == 8


@pytest.mark.parametrize("raw", [np.zeros((8, 5000), dtype=np.float32),
                                np.ones((8, 5000), dtype=np.float64),
                                np.zeros((12, 5000), dtype=np.float32)])
def test_invalid_or_flat_representation_fails_entire_case(raw):
    with pytest.raises(ValueError, match="geometry|floor"):
        audit.standardize(raw)


def test_pair_order_and_empty_union_convention():
    masks = np.zeros((8, 5000), dtype=bool)
    assert audit.pair_jaccards(masks) == [0.] * 28
    masks[:2, :10] = True
    assert audit.pair_jaccards(masks) == [1., *[0.] * 27]


def test_matching_leads_and_fixed_cohort_seed():
    row, masks = audit.prepare(synthetic_signal(), 2, "ludb")
    assert row["control_seed"] == 2026106302
    assert row["record_id"] == 3
    assert row["coherence"]["native_jaccard"] == 1.
    assert masks.shape == (8, 5000)
    assert masks.dtype == bool
    assert all(0 < value <= .1 for value in row["activity_occupancy"])
    np.testing.assert_array_equal(row["coherence"]["shifts"],
                                  np.random.default_rng(2026106302).integers(1, 5000, 8))


def test_real_first_independent_bootstrap_order_and_no_pass_flag():
    generated = synthetic_rows("generated", 128, .1)
    real = synthetic_rows("ludb", 200, 0.)
    for row in real:
        row["coherence"]["difference"] = row["index"] / 200
    for row in generated:
        row["coherence"]["difference"] = row["index"] / 128
    summary, draws = audit.summarize(generated, real)
    rng = np.random.default_rng(2026106401)
    first = np.asarray([row["coherence"]["difference"] for row in real])[rng.integers(0, 200, 200)].mean()
    second = np.asarray([row["coherence"]["difference"] for row in generated])[
        rng.integers(0, 128, 128)].mean()
    np.testing.assert_array_equal(draws[0], [first, second, first - second])
    assert summary["primary_descriptive"]["seed"] == 2026106401
    assert "scientific_passed" not in summary
    assert summary["clinical_quality_claim"] is False


def test_missing_seed_rejected_without_dropping_case():
    with pytest.raises(ValueError, match="complete"):
        audit.summarize(synthetic_rows("generated", 127, .1), synthetic_rows("ludb", 200, 0.))
