"""Tests of the MLC-03 Part A signal-quality measures on synthetic ECGs with planted defects."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_experiment.eda.signals import LEADS
from ecg_experiment.mlc03_signal_quality import (
    FS,
    INDEPENDENT,
    SAMPLES,
    duplicate_leads,
    edge_runs,
    extreme_runs,
    fill_missing,
    hf_noise,
    identity_residuals,
    isolated_spikes,
    longest_true_run,
    mains_amplitude,
    match_f1,
    measure_full_record,
    measure_window,
    quantisation_step,
    spike_size,
    step_size,
)

INDEX = {lead: position for position, lead in enumerate(LEADS)}
HEART_RATE = 72.0


def gaussian(time: np.ndarray, centre: float, width: float) -> np.ndarray:
    """
    Gaussian bump.

    Parameters
    ----------
    time : np.ndarray
        Times in seconds.
    centre, width : float
        Centre and standard deviation in seconds.

    Returns
    -------
    np.ndarray
        Values in [0, 1].
    """
    return np.exp(-0.5 * ((time - centre) / width) ** 2)


def synthetic_ecg(heart_rate: float = HEART_RATE, seed: int = 0, noise_mv: float = 0.004) -> np.ndarray:
    """
    Twelve-lead ECG from a dipole whose P, QRS and T loops point in fixed directions.

    Limb leads follow the identities exactly (I and II projected, the others derived), precordial leads
    progress from a negative QRS in V1 to a positive one in V6, and the result is rounded to microvolts.

    Parameters
    ----------
    heart_rate : float
        Beats per minute.
    seed : int
        Seed of the noise.
    noise_mv : float
        Standard deviation of white noise added to I, II and V1-V6.

    Returns
    -------
    np.ndarray
        Float64 ``(12, 5000)`` in mV.
    """
    time = np.arange(SAMPLES) / FS
    period = 60.0 / heart_rate
    beats = np.arange(0.4, 10.0 - 0.2, period)
    dipole = np.zeros((3, SAMPLES))
    waves = (((-0.16, 0.025), (0.15, 0.6, 0.05)), ((0.0, 0.012), (1.2, 0.9, 0.6)),
             ((0.012, 0.010), (-0.5, 0.3, 0.2)), ((0.28, 0.05), (0.3, 0.3, 0.25)))
    for beat in beats:
        for (offset, width), direction in waves:
            dipole += np.outer(direction, gaussian(time, beat + offset, width))
    rng = np.random.default_rng(seed)
    lead_i = dipole[0] + rng.normal(0, noise_mv, SAMPLES)
    lead_ii = 0.5 * dipole[0] + 0.866 * dipole[1] + rng.normal(0, noise_mv, SAMPLES)
    angles = np.deg2rad([115, 95, 75, 55, 30, 0])
    precordial = [np.cos(a) * dipole[0] + 0.3 * dipole[1] - np.sin(a) * dipole[2] * 1.4 for a in angles]
    signal = np.vstack([lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2, lead_i - lead_ii / 2,
                        lead_ii - lead_i / 2, *[v + rng.normal(0, noise_mv, SAMPLES) for v in precordial]])
    return np.round(signal * 1000.0) / 1000.0


@pytest.fixture(scope="module")
def clean() -> np.ndarray:
    """
    Clean synthetic ECG.

    Returns
    -------
    np.ndarray
        Output of ``synthetic_ecg``.
    """
    return synthetic_ecg()


@pytest.fixture(scope="module")
def clean_measures(clean: np.ndarray) -> dict:
    """
    Measures of the clean synthetic ECG.

    Returns
    -------
    dict
        Output of ``measure_window``.
    """
    return measure_window(clean)


def test_longest_true_run() -> None:
    assert longest_true_run(np.array([False, True, True, False, True])) == 2
    assert longest_true_run(np.zeros(4, dtype=bool)) == 0


def test_fill_missing_interpolates() -> None:
    window = np.array([[0.0, np.nan, 2.0], [np.nan, np.nan, np.nan]])
    assert np.array_equal(fill_missing(window), np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]))


def test_clean_ecg_has_no_defect(clean_measures: dict) -> None:
    measures = clean_measures
    assert measures["missing_samples"] == 0
    assert measures["n_duplicate_pairs"] == 0
    assert all(measures[f"flat__{lead}"] == 0 for lead in LEADS)
    assert max(measures[f"res_{lead}_max"] for lead in ("III", "aVR", "aVL", "aVF")) <= 0.002
    assert max(measures[f"mains50__{lead}"] for lead in LEADS) < 0.01
    assert max(measures[f"step__{lead}"] for lead in LEADS) < 0.2
    assert max(measures[f"hf_noise__{lead}"] for lead in LEADS) < 0.01
    assert max(measures[f"extreme_run__{lead}"] for lead in LEADS) <= 3
    assert max(measures[f"quiet_run__{lead}"] for lead in LEADS) < 100


def test_heart_rate_and_agreement(clean_measures: dict) -> None:
    for name in ("hr_multi", "hr_II", "hr_V5"):
        assert clean_measures[name] == pytest.approx(HEART_RATE, rel=0.02)
    assert clean_measures["agree_detectors"] == pytest.approx(1.0)
    assert clean_measures["agree_leads"] == pytest.approx(1.0)
    assert clean_measures["agree_multi_V5"] == pytest.approx(1.0)


def test_morphology_polarity(clean_measures: dict) -> None:
    assert clean_measures["qrs_area__I"] > 0
    assert clean_measures["qrs_area__aVR"] < 0
    assert clean_measures["p_amp__II"] > 0
    assert clean_measures["qrs_area__V1"] < 0 < clean_measures["qrs_area__V6"]


def test_missing_samples_are_counted(clean: np.ndarray) -> None:
    window = clean.copy()
    window[3, 100:110] = np.nan
    measures = measure_window(window)
    assert measures["missing_samples"] == 10
    assert measures["missing_leads"] == 1


def test_edge_runs_find_held_tail(clean: np.ndarray) -> None:
    window = clean.copy()
    window[:, -100:] = window[:, [-101]]
    head, tail = edge_runs(window)
    assert (tail >= 100).all()
    assert (head < 10).all()


def test_extreme_runs_find_clipping(clean: np.ndarray) -> None:
    window = clean.copy()
    window[1] = np.minimum(window[1], 0.5)
    runs, hits = extreme_runs(window)
    assert runs[1] >= 10
    assert hits[1] >= 10
    clean_runs, clean_hits = extreme_runs(clean)
    assert clean_runs[1] <= 3
    assert clean_hits[1] <= 1


def test_quantisation_step() -> None:
    rng = np.random.default_rng(1)
    fine = np.round(rng.normal(0, 0.2, (1, 5000)) * 1000) / 1000
    coarse = np.round(np.round(rng.normal(0, 0.2, (1, 5000)) / 0.00488) * 4.88) / 1000
    assert quantisation_step(fine)[0] == pytest.approx(1.0)
    assert quantisation_step(coarse)[0] == pytest.approx(4.88, abs=0.05)


def test_duplicate_leads(clean: np.ndarray) -> None:
    window = clean.copy()
    window[INDEX["V3"]] = window[INDEX["V2"]]
    window[INDEX["V5"]] = -window[INDEX["V4"]]
    assert duplicate_leads(window) == ["V2=V3", "V4=-V5"]
    assert duplicate_leads(clean) == []


def test_identity_residuals_find_swapped_leads(clean: np.ndarray) -> None:
    window = clean.copy()
    window[[INDEX["aVL"], INDEX["aVF"]]] = window[[INDEX["aVF"], INDEX["aVL"]]]
    residuals = identity_residuals(window)
    assert residuals["res_aVL_rms"] > 0.05
    assert residuals["res_aVF_rms"] > 0.05
    assert residuals["res_III_max"] <= 0.002


def test_mains_amplitude(clean: np.ndarray) -> None:
    time = np.arange(SAMPLES) / FS
    window = clean + 0.1 * np.sin(2 * np.pi * 50.0 * time)
    assert mains_amplitude(window, 50.0) == pytest.approx(np.full(12, 0.1), rel=0.1)
    assert (mains_amplitude(window, 60.0) < 0.01).all()


def test_hf_noise(clean: np.ndarray) -> None:
    noisy = clean + np.random.default_rng(2).normal(0, 0.05, clean.shape)
    assert (hf_noise(noisy) > 0.02).all()
    assert (hf_noise(clean) < 0.01).all()


def test_step_size(clean: np.ndarray) -> None:
    window = clean.copy()
    window[INDEX["V2"], 2500:] += 1.0
    interior, edge = step_size(window)
    assert interior[INDEX["V2"]] == pytest.approx(1.0, abs=0.15)
    assert np.delete(interior, INDEX["V2"]).max() < 0.2


def test_spike_size(clean: np.ndarray) -> None:
    window = clean.copy()
    window[INDEX["V3"], 2000] += 3.0
    assert spike_size(window)[INDEX["V3"]] >= 5.0
    assert spike_size(clean).max() < 1.0


def test_isolated_spike_ignores_shared_spikes(clean: np.ndarray) -> None:
    single = clean.copy()
    single[INDEX["V3"], 2000] += 3.0
    shared = clean.copy()
    shared[:, 2000] += 3.0
    assert isolated_spikes(single)[INDEPENDENT.index("V3")] >= 5.0
    assert isolated_spikes(shared).max() < 1.0
    assert isolated_spikes(clean).max() < 1.0


def test_qrs_dropout_finds_dead_lead(clean: np.ndarray, clean_measures: dict) -> None:
    window = clean.copy()
    window[INDEX["V4"], 2500:] = window[INDEX["V4"], 2500] + np.linspace(0, 0.02, 2500)
    measures = measure_window(window)
    assert measures["dropout_run__V4"] >= 5
    assert measures["dropout_share__V4"] > 0.4
    assert max(clean_measures[f"dropout_run__{lead}"] for lead in INDEPENDENT) == 0


def test_quiet_run_finds_contact_loss(clean: np.ndarray) -> None:
    window = clean.copy()
    window[INDEX["V3"], 2000:2600] = window[INDEX["V3"], 2000]
    measures = measure_window(window)
    assert measures["quiet_run__V3"] >= 600
    assert measures["const_run_interior__V3"] >= 600


def test_flat_lead(clean: np.ndarray) -> None:
    window = clean.copy()
    window[INDEX["V4"]] = 0.0
    measures = measure_window(window)
    assert measures["flat__V4"] == 1


def test_unmeasurable_rhythm() -> None:
    noise = np.round(np.random.default_rng(3).normal(0, 0.05, (12, SAMPLES)) * 1000) / 1000
    measures = measure_window(noise)
    assert measures["agree_detectors"] < 0.5 or measures["n_beats_multi"] < 3


def test_match_f1() -> None:
    assert match_f1(np.array([100, 500]), np.array([110, 900])) == pytest.approx(0.5)
    assert np.isnan(match_f1(np.array([]), np.array([])))


def test_full_record_finds_late_flat_lead(clean: np.ndarray) -> None:
    signal = np.hstack([clean, clean])
    signal[INDEX["V6"], SAMPLES:] = 0.25
    measures = measure_full_record(signal)
    assert measures["full_samples"] == 2 * SAMPLES
    assert measures["after10_flat_leads"] == 1
    assert measures["full_flat_leads"] == 0


def test_measure_window_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected"):
        measure_window(np.zeros((12, 4000)))


def test_test_roles_stay_closed() -> None:
    from ecg_experiment.mlc03_signal_runs import check_roles, open_block, read_index

    with pytest.raises(PermissionError, match="closed"):
        check_roles(("train", "test_external"))
    with pytest.raises(PermissionError, match="closed"):
        read_index(("sph_calibration",))
    with pytest.raises(PermissionError, match="closed"):
        open_block("test_same_source__ptbxl", allow_test=False)
    assert check_roles(("test_external",), allow_test=True) == ("test_external",)


def test_cached_window_reproduces_manifest_hash() -> None:
    from ecg_experiment.mlc03_signal_runs import INDEX, open_block, read_checked, read_index

    if not INDEX.exists():
        pytest.skip("the local waveform cache is absent")
    index = read_index(("train",))
    row = index.iloc[0]
    window = read_checked(open_block(row["block"], allow_test=False), int(row["row"]), row["sha256_10s"],
                          row["record_key"])
    assert window.shape == (12, SAMPLES)
    with pytest.raises(ValueError, match="does not reproduce"):
        read_checked(open_block(row["block"], allow_test=False), int(row["row"]), "0" * 64, row["record_key"])
