"""Scientific invariants for paired generator and observable-support evaluation."""

import inspect

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.incart_localization056 import score_signal
from ecg_experiment.lead_wave_maps import UnitMap
from ecg_experiment.synthetic_generator058 import KINDS, VECTORS, generate_subject, intervene, limb_projection
from ecg_experiment.synthetic_localization058 import grouped_summary, localization_metrics


def test_virtual_electrodes_obey_limb_constraints():
    electrode = np.random.default_rng(1).normal(size=(9, 100))
    lead = limb_projection(electrode)
    np.testing.assert_allclose(lead[2], lead[1] - lead[0], atol=1e-14)
    np.testing.assert_allclose(lead[3], -(lead[0] + lead[1]) / 2, atol=1e-14)
    np.testing.assert_allclose(lead[4], lead[0] - lead[1] / 2, atol=1e-14)
    np.testing.assert_allclose(lead[5], lead[1] - lead[0] / 2, atol=1e-14)


@pytest.mark.parametrize("family", ["ecgsyn", "compact"])
def test_generator_determinism_exact_sham_and_changed_scale(family):
    first = generate_subject(250, "iid", family)
    second = generate_subject(250, "iid", family)
    np.testing.assert_array_equal(first["signal"], second["signal"])
    np.testing.assert_array_equal(intervene(first, "sham")["signal"], first["signal"])
    np.testing.assert_array_equal(intervene(first, "sham")["difference"], 0)
    assert first["signal"].shape == (12, 6000)
    assert np.isclose(np.ptp(first["clean"][1]), 1.2)
    assert np.max(np.abs(intervene(first, "st")["difference"])) == pytest.approx(0.15)


@pytest.mark.parametrize("family", ["ecgsyn", "compact"])
def test_truth_comes_from_actual_difference_and_nuisance_stays_untargeted(family):
    subject = generate_subject(450, "shifted", family)
    for kind in KINDS:
        changed = intervene(subject, kind)
        difference = changed["noiseless"] - subject["clean"]
        expected = np.abs(difference) >= 0.01
        expected &= (expected.sum(axis=1) >= 10)[:, None]
        np.testing.assert_array_equal(changed["mask"], expected)
        np.testing.assert_array_equal(changed["energy"], difference**2)
        if kind in ("sham", "drift", "gain", "noise"):
            assert not changed["mask"].any()
    persistent = intervene(subject, "persistent_st")
    for anchor in subject["anchors"]:
        if 125 <= anchor < 5775:
            assert persistent["mask"][:, anchor + 50 : anchor + 120].any()
    inverted = intervene(subject, "persistent_t")
    expected_t = (
        subject["projection"] @ (-2 * VECTORS[4, :, None] * subject["components"][4, None])
    ) * subject["scale"]
    np.testing.assert_allclose(inverted["difference"], expected_t, atol=1e-14)


def test_tied_fixed_area_metrics_and_unscorable_case():
    mask = np.zeros((12, 500), bool)
    mask[2, 100:180] = True
    energy = mask.astype(float)
    unit = UnitMap(np.array([1.0, 1.0]), np.array([2, 2]), np.array([0.2, 0.6]), np.array([0.34, 0.74]))
    result = localization_metrics(unit, mask, energy)
    assert result["hit"] == 0.5
    assert result["ties"] == 2
    assert result["energy_capture"] == pytest.approx(70 / 80 / 2)
    assert result["iou"] == pytest.approx(70 / 80 / 2)
    assert localization_metrics(None, mask, energy)["hit"] == 0


def test_inference_api_cannot_receive_counterfactual_or_mask():
    assert set(inspect.signature(score_signal).parameters) == {"signal", "references", "fixed_peaks"}


def test_repeated_edits_are_averaged_within_subject():
    rows = []
    for subject in range(3):
        for kind in KINDS:
            for method in ("aligned_residual", "U_B_fixed"):
                hit = float(method == "aligned_residual" and subject > 0)
                rows.append(
                    {
                        "subject": subject,
                        "family": "ecgsyn",
                        "cohort": "iid",
                        "kind": kind,
                        "map": method,
                        "hit": hit,
                        "chance": 0.0,
                        "target_occupancy": 0.0,
                        "iou": 0.0,
                        "energy_capture": 0.0,
                        "flag": 0,
                        "informative": kind not in ("sham", "drift", "gain", "noise"),
                        "inference_failure": False,
                    }
                )
    result = grouped_summary(pd.DataFrame(rows))["ecgsyn/iid"]["transient_paired_gain"]
    assert result["value"] == pytest.approx(2 / 3)
    assert result["subjects"] == 3
