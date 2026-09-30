"""Checks of the Challenge feature extraction helpers."""

import hashlib

import numpy as np

from ecg_experiment.challenge_features import (
    canonical_window,
    encoder_inputs,
    files_digest,
    record_stems,
    skip_reasons,
    window_start,
)
from ecg_experiment.waveforms import LEADS


def test_record_stems_lists_one_source_sorted():
    checksums = {
        "training/georgia/g2/E10001.hea": "a", "training/georgia/g2/E10001.mat": "b",
        "training/georgia/g1/E00001.hea": "c", "training/georgia/g1/E00001.mat": "d",
        "training/cpsc_2018/g1/A0001.hea": "e", "README.md": "f",
    }
    assert record_stems(checksums, "georgia") == ["training/georgia/g1/E00001", "training/georgia/g2/E10001"]
    assert record_stems(checksums, "cpsc_2018") == ["training/cpsc_2018/g1/A0001"]


def test_files_digest_hashes_header_and_signal_lines_in_record_order():
    checksums = {"r1.hea": "a", "r1.mat": "b", "r2.hea": "c", "r2.mat": "d"}
    expected = hashlib.sha256(b"a  r1.hea\nb  r1.mat\nc  r2.hea\nd  r2.mat\n").hexdigest()
    assert files_digest(["r1", "r2"], checksums) == expected
    assert files_digest(["r2", "r1"], checksums) != expected


def test_window_start_centres_long_records():
    assert window_start(5000) == 0
    assert window_start(7500) == 1250
    assert window_start(5973) == 486


def test_skip_reasons():
    good = np.zeros((5000, 12))
    assert skip_reasons(500, good) == []
    assert skip_reasons(500, np.zeros((7500, 12))) == []
    assert skip_reasons(500, np.zeros((2500, 12))) == ["samples_2500"]
    assert skip_reasons(257, np.zeros((5000, 8))) == ["sampling_rate_257_hz", "leads_8"]
    bad = good.copy()
    bad[10, 3] = np.nan
    assert skip_reasons(500, bad) == ["nonfinite"]


def test_nonfinite_outside_the_window_is_not_a_skip():
    signal = np.zeros((7500, 12))
    signal[:1250] = np.nan
    assert skip_reasons(500, signal) == []
    signal[1250, 0] = np.nan
    assert skip_reasons(500, signal) == ["nonfinite"]


def test_canonical_window_reorders_leads_and_takes_the_window():
    names = list(reversed(LEADS))
    time = np.arange(7500, dtype=np.float64)[:, None]
    signal = time * 100 + np.arange(12, dtype=np.float64)[::-1]
    window = canonical_window(signal, names, 1250)
    assert window.shape == (12, 5000)
    assert window.dtype == np.float64
    np.testing.assert_array_equal(window[:, 0], 125000 + np.arange(12))
    np.testing.assert_array_equal(window[0, [0, -1]], [125000, 624900])


def test_encoder_inputs_shapes_and_dtypes():
    rng = np.random.default_rng(0)
    inputs = encoder_inputs(rng.normal(scale=0.3, size=(12, 5000)))
    assert {name: value.shape for name, value in inputs.items()} == {
        "cpc": (12, 2500), "jepa": (8, 2500), "xecg": (1000, 12)}
    assert all(value.dtype == np.float32 for value in inputs.values())
