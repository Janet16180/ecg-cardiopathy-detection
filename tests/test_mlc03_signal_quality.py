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


def test_dense_score_finds_rescaled_copy() -> None:
    from ecg_experiment.mlc03_near_duplicates import dense_pairs, dense_vector

    first, other = synthetic_ecg(seed=4), synthetic_ecg(heart_rate=61, seed=5)
    copy = 1.7 * first + 0.2 + np.random.default_rng(6).normal(0, 0.003, first.shape)
    vectors = np.stack([dense_vector(first), dense_vector(copy), dense_vector(other)])
    pairs, nearest, scores = dense_pairs(vectors, vectors, floor=0.95, same_set=True, neighbours=2)
    assert pairs[["query", "reference"]].values.tolist() == [[0, 1]]
    assert pairs["score"].iloc[0] > 0.99
    assert nearest[2, 0] in (0, 1)
    assert scores[2, 0] < 0.9


def test_shifted_copy_is_found_and_verified() -> None:
    from ecg_experiment.mlc03_near_duplicates import (
        fingerprint,
        r_peaks,
        rr_keys,
        shifted_candidates,
        verify_candidates,
    )

    rng = np.random.default_rng(7)
    rr = rng.normal(0.85, 0.06, 40)
    time = np.arange(3 * SAMPLES) / FS
    long = np.zeros((12, 3 * SAMPLES))
    for beat in np.cumsum(rr):
        long += np.outer(np.r_[1.0, 1.4, 0.4, -1.2, 0.3, 0.9, -0.6, 0.4, 1.2, 1.6, 1.4, 1.0],
                         np.exp(-0.5 * ((time - beat) / 0.012) ** 2))
    long += rng.normal(0, 0.005, long.shape)
    first, second = long[:, :SAMPLES], long[:, 3000 : 3000 + SAMPLES]
    other = synthetic_ecg(heart_rate=70, seed=8)
    keys = {number: rr_keys(r_peaks(signal)) for number, signal in enumerate((first, second, other))}
    candidates = shifted_candidates(keys)
    assert candidates[["first", "second"]].values.tolist() == [[0, 1]]
    assert candidates["lag"].iloc[0] == pytest.approx(-3000, abs=8)
    prints = {number: fingerprint(signal) for number, signal in enumerate((first, second, other))}
    verified = verify_candidates(candidates, prints)
    assert verified["correlation"].iloc[0] > 0.95
    assert verified["overlap_s"].iloc[0] == pytest.approx(4.0, abs=0.1)


def _assign(window: np.ndarray, lead: str, values: np.ndarray | float, part: slice = slice(None)) -> None:
    """Set part of one lead in place."""
    window[INDEX[lead], part] = values


def _noise(scale: float) -> np.ndarray:
    """Rounded white noise of one lead."""
    return np.round(np.random.default_rng(11).normal(0, scale, SAMPLES) * 1000) / 1000


TIME = np.arange(SAMPLES) / FS
PLANTS = {
    "missing_samples": lambda w, c: _assign(w, "V5", np.nan, slice(3000, 3010)),
    "edge_hold": lambda w, c: w.__setitem__((slice(None), slice(-100, None)), w[:, [-101]]),
    "adc_rail": lambda w, c: _assign(w, "V2", 32.767, slice(2400, 2410)),
    "flat_lead": lambda w, c: _assign(w, "V4", 0.0),
    "near_flat_lead": lambda w, c: _assign(w, "V4", _noise(0.004)),
    "duplicate_lead": lambda w, c: _assign(w, "V3", c[INDEX["V2"]]),
    "identity_violation": lambda w, c: _assign(w, "aVL", c[INDEX["aVL"]] + 0.2),
    "reversal_la_ra": lambda w, c: w.__setitem__([0, 1, 2, 3, 4], np.vstack([-c[0], c[2], c[1], c[4], c[3]])),
    "reversal_ra_ll": lambda w, c: w.__setitem__([0, 1, 2, 3, 5],
                                                 np.vstack([-c[2], -c[1], -c[0], c[5], c[3]])),
    "reversal_neutral": lambda w, c: _assign(w, "III", _noise(0.005)),
    "amplitude_outlier": lambda w, c: w.__imul__(5.0),
    "baseline_wander": lambda w, c: _assign(w, "V1", c[INDEX["V1"]] + 1.5 * np.sin(2 * np.pi * 0.2 * TIME)),
    "mains_interference": lambda w, c: w.__iadd__(0.1 * np.sin(2 * np.pi * 50.0 * TIME)),
    "hf_noise": lambda w, c: w.__iadd__(np.random.default_rng(11).normal(0, 0.1, w.shape)),
    "baseline_step": lambda w, c: _assign(w, "V2", c[INDEX["V2"], 2500:] + 1.5, slice(2500, None)),
    "isolated_spike": lambda w, c: _assign(w, "V3", c[INDEX["V3"], 2000] + 3.0, slice(2000, 2001)),
    "contact_loss": lambda w, c: _assign(w, "V4", c[INDEX["V4"], 1500], slice(1500, None)),
    "rhythm_unmeasurable": lambda w, c: w.__setitem__(
        slice(None), np.round(np.random.default_rng(11).normal(0, 0.05, w.shape) * 1000) / 1000),
}


def plant(defect: str, clean: np.ndarray) -> np.ndarray:
    """
    Plant one defect in a copy of the clean synthetic ECG.

    Parameters
    ----------
    defect : str
        Flag name whose defect is planted.
    clean : np.ndarray
        Output of ``synthetic_ecg``.

    Returns
    -------
    np.ndarray
        The defective copy.
    """
    window = clean.copy()
    PLANTS[defect](window, clean)
    return window


def flags_of(window: np.ndarray, key: str = "synthetic") -> dict:
    """
    Flags of one window.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` mV.
    key : str
        Record key.

    Returns
    -------
    dict
        The single row of ``compute_flags``.
    """
    import pandas as pd

    from ecg_experiment.mlc03_signal_flags import compute_flags

    return compute_flags(pd.DataFrame([{"record_key": key, **measure_window(window)}])).iloc[0].to_dict()


def test_clean_ecg_raises_no_flag(clean: np.ndarray) -> None:
    from ecg_experiment.mlc03_signal_flags import FLAGS

    row = flags_of(clean)
    assert [name for name in FLAGS if row[name]] == []


@pytest.mark.parametrize("defect", list(PLANTS))
def test_each_flag_detects_its_planted_defect(defect: str, clean: np.ndarray) -> None:
    assert flags_of(plant(defect, clean))[defect]


def test_every_flag_has_a_test_and_a_definition() -> None:
    from ecg_experiment.mlc03_signal_flags import FLAGS

    assert set(FLAGS) == set(PLANTS)
    for spec in FLAGS.values():
        assert set(spec) == {"item", "definition", "threshold", "reason"}
