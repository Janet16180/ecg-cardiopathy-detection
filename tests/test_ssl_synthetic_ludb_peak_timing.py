"""Artificial tests of explicit-landmark ranks without inferred boundaries or negative labels."""

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_ludb_peak_timing as peak


def event(sample, symbol, code, channel=0):
    return {"sample": sample, "symbol": symbol, "type_code": code, "channel": channel,
            "subtype": 0, "num": 0, "aux_length": 0, "symbol_lookup_status": "mapped"}


def test_midrank_ties_and_constant_reference():
    np.testing.assert_array_equal(peak.midranks(np.array([0., 1., 1., 3.]), np.array([0., 1., 3.])),
                                  [.125, .5, .875])
    np.testing.assert_array_equal(peak.midranks(np.ones(10), np.ones(3)), [.5, .5, .5])


def test_missing_onset_unknown_code_and_duplicate_landmarks_are_retained():
    rows = [event(100, "N", 1, 19), event(100, "N", 1, 19), event(150, None, 58), event(300, ")", 40)]
    result = peak.validate_events(rows)
    assert result["N_samples"] == [100, 100]
    assert result["duplicate_N_timestamp_excess_count"] == 1
    assert result["span_samples"] == 201
    assert result["events"] == rows
    assert result["N_without_immediately_preceding_onset_indices"] == [0, 1]


def test_invalid_times_and_missing_n_raise():
    with pytest.raises(ValueError, match="ordered integer"):
        peak.validate_events([event(100, "N", 1), event(99, ")", 40)])
    with pytest.raises(ValueError, match="explicit N"):
        peak.validate_events([event(100, "t", 27), event(300, ")", 40)])


def test_rank_once_control_is_exact_literal_shift_count_statistic():
    signal = np.sin(np.arange(5000) / 20)
    structure = peak.validate_events([event(100, "N", 1), event(300, "N", 1), event(800, ")", 40)])
    metrics, proxy = peak.score_lead(signal, structure)
    reference = proxy[100:801]
    points = [0, 200]
    for offset, actual in zip(metrics["shift_samples"], metrics["shift_N_midrank_values"], strict=True):
        shifted = np.roll(reference, offset)
        expected = [(np.count_nonzero(shifted < shifted[index])
                     + .5 * np.count_nonzero(shifted == shifted[index])) / len(shifted) for index in points]
        np.testing.assert_array_equal(actual, expected)
    flat, _ = peak.score_lead(np.zeros(5000), structure)
    assert flat["N_mean_midrank"] == flat["mean_shift_N_midrank"] == .5
    assert flat["gain"] == 0


def test_parser_role_has_no_proxy_or_rank_and_sanitizes_rdann_header(tmp_path, monkeypatch):
    values = np.tile(np.sin(np.arange(5000) / 20)[:, None], (1, 12))
    peak.wfdb.wrsamp("5", fs=500, units=["mV"] * 12, sig_name=list(peak.LEADS),
                     p_signal=values, fmt=["16"] * 12, write_dir=str(tmp_path))
    with (tmp_path / "5.hea").open("a") as stream:
        stream.write("# DO_NOT_READ_ORIGINAL_DIAGNOSIS\n")
    for lead in peak.LEADS:
        peak.wfdb.wrann("5", "test", np.array([100, 200, 300]),
                         symbol=["N", "t", ")"], write_dir=str(tmp_path))
        (tmp_path / "5.test").rename(tmp_path / ("5." + lead))
    original = peak.wfdb.io.record.rdheader

    def sanitized(record, *args, **kwargs):
        assert "sanitized" in str(record)
        assert "DO_NOT_READ_ORIGINAL_DIAGNOSIS" not in (tmp_path / "sanitized" / "5.hea").read_text()
        return original(record, *args, **kwargs)

    def forbidden_proxy(values):
        raise AssertionError("Parser-role records cannot calculate proxy or ranks")

    monkeypatch.setattr(peak.wfdb.io.record, "rdheader", sanitized)
    monkeypatch.setattr(peak.original, "activity_proxy", forbidden_proxy)
    row, _, scores = peak.evaluate_record(tmp_path / "5", tmp_path / "sanitized")
    assert scores is None
    assert row["role"] == "parser_qualification"
    assert "N_mean_midrank" not in row
    assert all(lead["independent_event_decode"]["all_numeric_fields_exact"] for lead in row["leads"])


def test_whole_subject_bootstrap_and_equal_role_weighting():
    rows = []
    for identifier in range(1, 201):
        row = {"record_id": identifier,
               "role": "parser_qualification" if identifier % 5 == 0 else "measurement",
               "leads": [{"span_samples": 3000, "N_samples": [100, 200], "N_mean_midrank": .8,
                           "mean_shift_N_midrank": .4, "gain": .4, "constant_score": False}] * 12}
        if identifier % 5:
            row.update(N_mean_midrank=.8, mean_shift_N_midrank=.4, shift_N_mean_midrank=[.4] * 6, gain=.4)
        rows.append(row)
    result = peak.aggregate(rows)
    assert result["measurement_records"] == 160
    np.testing.assert_allclose(result["bootstrap_gains"], .4, atol=1e-15)
    assert result["scientific_passed"]
    assert "AP" not in result
