"""Scientific invariants of the independent expert beat benchmark."""

import numpy as np
import pandas as pd
import wfdb

from ecg_experiment.fragment_localization import r_peaks
from ecg_experiment.incart_localization056 import (
    METHODS,
    beat_scores,
    benchmark_summary,
    match_beats,
    patient_metrics,
    pieces_at_peaks,
    read_record,
)
from ecg_experiment.lead_wave_maps import UnitMap, beat_pieces


def test_physical_gains_canonical_leads_patient_and_resampling(tmp_path):
    names = ["V6", "V5", "V4", "V3", "V2", "V1", "AVF", "AVL", "AVR", "III", "II", "I"]
    values = np.tile(np.arange(12, 0, -1) / 10, (257, 1))
    wfdb.wrsamp(
        "test",
        fs=257,
        units=["mV"] * 12,
        sig_name=names,
        p_signal=values,
        fmt=["16"] * 12,
        adc_gain=list(np.linspace(240, 1063, 12)),
        baseline=[0] * 12,
        comments=["patient 7"],
        write_dir=str(tmp_path),
    )
    signal, patient = read_record(tmp_path / "test")
    assert patient == 7
    assert signal.shape == (12, 500)
    assert np.allclose(np.median(signal[:, 50:450], axis=1), np.arange(1, 13) / 10, atol=0.005)


def test_matching_is_one_to_one_deterministic_and_bounded():
    detected = np.array([1.0, 1.25, 2.0, 3.0])
    reference = np.array([1.125, 2.01, 3.2])
    assert np.array_equal(match_beats(detected, reference), [0, -1, 1, -1])
    assert np.array_equal(match_beats(detected, reference), match_beats(detected, reference))


def test_explicit_anchors_factor_historical_wave_extraction_exactly():
    signal = np.zeros((12, 5000))
    for peak in range(300, 4800, 500):
        signal[:, peak - 40 : peak + 41] = np.exp(-(np.arange(-40, 41) ** 2) / 100)
    historical_times, historical = beat_pieces(signal, 500)
    times, factored = pieces_at_peaks(signal, r_peaks(signal, 500))
    assert np.array_equal(times, historical_times)
    for name, values in historical.items():
        assert np.array_equal(values, factored[name])


def test_midpoint_window_ownership_preserves_original_beat_identity():
    units = UnitMap(
        np.array([1.0, 2.0, 9.0]),
        np.array([0, 1, 2]),
        np.array([0.53, 0.83, 1.43]),
        np.array([0.67, 0.97, 1.57]),
    )
    scores, starts, leads = beat_scores(units, np.array([300, 750]))
    assert np.array_equal(scores, [2, 9])
    assert np.allclose(starts, [0.83, 1.43])
    assert np.array_equal(leads, [1, 2])


def test_patient_macro_weight_and_missed_v_denominator():
    records = []
    for patient, size in ((1, 1), (2, 100)):
        for symbol in ("N", "V"):
            for _ in range(size):
                candidate = float((symbol == "V") == (patient == 1))
                records.append(
                    {"patient": patient, "symbol": symbol, METHODS[0]: candidate, METHODS[1]: 1 - candidate}
                )
    beats = pd.DataFrame(records)
    reference = pd.DataFrame(records + [{"patient": 1, "symbol": "V"}])
    chunks = pd.DataFrame(
        [
            {"patient": patient, "map": name, "excess": 0.0, "hit": 0.5, "chance": 0.5}
            for patient in (1, 2)
            for name in METHODS
        ]
    )
    patients = patient_metrics(beats, chunks, reference, dict.fromkeys(METHODS, 0.5))
    summary = benchmark_summary(patients)
    assert summary["methods"][METHODS[0]]["auroc"] == 0.5
    assert summary["methods"][METHODS[1]]["auroc"] == 0.5
    assert summary["paired"]["auroc"]["value"] == 0
    selected = patients[(patients["patient"] == 1) & (patients["map"] == METHODS[0])].iloc[0]
    assert selected["v_end_to_end_sensitivity"] == 0.5
