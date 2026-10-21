"""Independent numerical-transfer integrity tests, including controls that cannot claim change ranking."""

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_background_editor_transfer_audit as audit
from ecg_experiment.ssl_synthetic_quality_saved_audit import canonical


def test_bipolar_kernel_retains_nominal_l2_and_balanced_shape_but_not_peak() -> None:
    for width in (61, 81, 101):
        original, heldout = audit.kernel(width, "hann"), audit.kernel(width, "biphasic")
        np.testing.assert_allclose(heldout, -heldout[::-1], atol=2e-15, rtol=0)
        assert heldout.sum() == pytest.approx(0, abs=1e-14)
        assert np.sum(heldout**2) == pytest.approx(np.sum(original**2), abs=2e-14)
        assert heldout[0] == heldout[-1] == 0
        assert heldout.min() < 0 < heldout.max()
        assert np.max(np.abs(heldout)) != pytest.approx(np.max(original))


def test_truth_is_final_float32_difference_after_canonical_derivation() -> None:
    native = np.random.default_rng(89).normal(size=(8, 5000)).astype(np.float32)
    untouched = native.copy()
    observed, strict, support, geometry = audit.make_case(native, 2000)
    np.testing.assert_array_equal(native, untouched)
    np.testing.assert_array_equal(observed[0], canonical(native))
    difference = observed[2:].astype(np.float64) - observed[1].astype(np.float64)
    np.testing.assert_array_equal(strict, difference != 0)
    np.testing.assert_array_equal(support, np.abs(difference) > 0.01)
    assert all(row["outside_hull_cells"] == 0 for row in geometry["checks"])
    assert all(row["support_cells"] > 0 for row in geometry["checks"])
    wrong = support.copy()
    wrong[0, 11, 0] = ~wrong[0, 11, 0]
    with pytest.raises(ValueError, match="final-input support"):
        audit.score_record(np.full_like(observed, 0.2), observed, wrong, "real", 0, 1, "B", support)


def test_static_maps_and_uniform_probability_shift_cannot_pass_same_mask_response() -> None:
    native = np.zeros((8, 5000), dtype=np.float32)
    observed, _, support, _ = audit.make_case(native, 2000)
    _, _, donor, _ = audit.make_case(native, 2001)
    pattern = np.arange(60000, dtype=np.float32).reshape(12, 5000) / 60000
    static = np.stack([pattern] * 6)
    row = audit.score_record(static, observed, support, "real", 0, 1, "B", donor)
    for value in row["positives"].values():
        assert value["response_ap"] == 0
        assert value["delta_inside"] == value["delta_outside"] == value["selectivity"] == 0
    shifted = np.full_like(observed, 0.5)
    shifted[1] = 0.125
    row = audit.score_record(shifted, observed, support, "real", 0, 1, "B", donor)
    for value in row["positives"].values():
        assert value["edited_ap"] == pytest.approx(value["prevalence"], abs=1e-15)
        assert value["donor_site_ap"] == pytest.approx(value["donor_prevalence"], abs=1e-15)
        assert value["response_ap"] == 0
        assert value["delta_inside"] == value["delta_outside"] == 0.375
        assert value["selectivity"] == 0


def test_monotone_absolute_input_predictor_has_zero_amplitude_advantage() -> None:
    native = np.zeros((8, 5000), dtype=np.float32)
    observed, _, support, _ = audit.make_case(native, 2000)
    _, _, donor, _ = audit.make_case(native, 2001)
    probabilities = np.abs(observed) / np.float32(2)
    row = audit.score_record(probabilities, observed, support, "generated", 0, 2, "P", donor)
    assert all(value["amplitude_gain"] == 0 for value in row["positives"].values())


def test_nonself_donor_permutation_preserves_support_distribution_without_filtering() -> None:
    counts = []
    for slot in range(64):
        _, _, support, _ = audit.make_case(np.zeros((8, 5000), dtype=np.float32), 2000 + slot)
        counts.append(support.sum(axis=(1, 2)))
    counts = np.asarray(counts)
    donors = counts[(np.arange(64) + 1) % 64]
    for positive in range(4):
        np.testing.assert_array_equal(np.sort(counts[:, positive]), np.sort(donors[:, positive]))
    np.testing.assert_allclose((counts - donors).mean(axis=0), 0, atol=0, rtol=0)


def fixture_rows() -> list[dict]:
    rows = []
    for source_index, source in enumerate(audit.SOURCES):
        for slot in range(64):
            for fit in audit.FITS:
                for arm_index, arm in enumerate(audit.ARMS):
                    positives = {}
                    for family_index, family in enumerate(audit.FAMILIES):
                        for mode_index, mode in enumerate(audit.MODES):
                            baseline = (
                                0.10 + source_index * 0.01 + arm_index * 0.02 + fit * 0.005 + slot * 0.001
                            )
                            baseline += family_index * 0.02 + mode_index * 0.003
                            positives[f"{family}_{mode}"] = {
                                metric: baseline + position * 0.004
                                for position, metric in enumerate(audit.METRICS)
                            }
                    control = dict.fromkeys(
                        ("mean", "q95", "maximum", "top600_mean", "maximum_gt_0p5"), slot * 0.001
                    )
                    rows.append(
                        {
                            "source": source,
                            "slot": slot,
                            "fit_id": fit,
                            "arm": arm,
                            "positives": positives,
                            "controls": {"sham": control, "nuisance": control},
                        }
                    )
    return rows


def test_shared_slot_bootstrap_holds_models_fixed_and_all_contrasts_share_draws() -> None:
    from ecg_experiment.intervals import patient_groups, patient_resample

    rows = fixture_rows()
    config = {"bootstrap": {"seed": 2026106601, "draws": 2000}}
    summary, draws = audit.summarize(rows, config)
    assert summary["primary"]["response_ap"]["mean"] == pytest.approx(0.181, abs=1e-15)
    names = summary["endpoint_order"]
    assert draws.shape == (2000, len(names))
    for metric in audit.METRICS:
        index = names.index(f"interaction__B__{metric}")
        np.testing.assert_allclose(draws[:, index], 0, atol=2e-16, rtol=0)
        index = names.index(f"real__PK_minus_B__biphasic__{metric}")
        np.testing.assert_allclose(draws[:, index], 0.06, atol=2e-16, rtol=0)
    rng = np.random.default_rng(2026106601)
    selected = patient_resample(patient_groups(np.arange(64)), rng)
    first = (0.1495 + 0.001 * np.arange(64))[selected].mean()
    assert draws[0, names.index("real__B__biphasic__response_ap")] == pytest.approx(first, abs=1e-15)
    assert len(summary["scientific_gates"]) == 6
    with pytest.raises(ValueError, match="1536"):
        audit.summarize([*rows[:-1], rows[0]], config)
