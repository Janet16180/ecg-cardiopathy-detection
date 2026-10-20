"""Artificial-only checks of annotation semantics and frozen timing measurements."""

from pathlib import Path

import numpy as np
import pytest
import wfdb

from ecg_experiment import ssl_synthetic_ludb_qualification as ludb


def test_inclusive_spans_and_unlabelled_margins():
    waves = ludb.parse_waves(np.array([100, 110, 120, 140, 150, 160, 180, 190, 200]),
                             ["(", "p", ")", "(", "N", ")", "(", "t", ")"])
    mask, envelope = ludb.targets(waves)
    assert envelope == (100, 201)
    assert mask.sum() == 21
    assert mask[140] == mask[160] == 1
    assert mask[139] == mask[161] == 0


@pytest.mark.parametrize(("samples", "symbols"), [
    ([1, 2], ["(", "N"]), ([1, 2, 3], ["(", "x", ")"]),
    ([1, 3, 2], ["(", "N", ")"]), ([1, 1, 1], ["(", "N", ")"]),
    ([-1, 2, 3], ["(", "N", ")"]), ([1, 2, 5000], ["(", "N", ")"]),
    ([1, 2, 3, 2, 4, 5], ["(", "N", ")", "(", "t", ")"]),
])
def test_malformed_annotations_raise(samples, symbols):
    with pytest.raises(ValueError, match="triplets|symbols|ordering|outside"):
        ludb.parse_waves(np.array(samples), symbols)


def test_missing_qrs_is_not_silent_negative():
    waves = ludb.parse_waves(np.array([100, 110, 120]), ["(", "p", ")"])
    with pytest.raises(ValueError, match="positives and negatives"):
        ludb.targets(waves)


def test_proxy_has_no_annotation_dependence_and_is_scale_free():
    signal = np.sin(np.arange(5000) / 17)
    score = ludb.activity_proxy(signal)
    np.testing.assert_allclose(score, ludb.activity_proxy(-3 * signal + 10), atol=2e-14)
    assert score.max() == 1
    assert not ludb.activity_proxy(np.zeros(5000)).any()
    derivative = np.diff(signal, prepend=signal[0])
    expected = np.convolve(derivative**2, np.ones(20) / 20, mode="same")
    np.testing.assert_array_equal(score, expected / expected.max())


def test_flat_tied_ap_equals_prevalence():
    waves = ludb.parse_waves(np.array([100, 110, 120, 140, 150, 160]),
                             ["(", "N", ")", "(", "t", ")"])
    row, score = ludb.lead_metrics(np.zeros(5000), waves)
    assert row["AP"] == 21 / 61
    np.testing.assert_allclose(row["shift_AP"], 21 / 61, atol=0)
    assert row["gain"] == pytest.approx(0)
    assert row["shift_samples"] == [int(61 * fraction) for fraction in ludb.SHIFTS]
    assert score.shape == (5000,)


def test_reader_canonicalizes_without_reading_header_comments(tmp_path: Path, monkeypatch):
    values = np.tile(np.arange(12), (5000, 1)).astype(float)
    wfdb.wrsamp("1", fs=500, units=["mV"] * 12, sig_name=list(reversed(ludb.LEADS)),
                p_signal=values, fmt=["16"] * 12, write_dir=str(tmp_path))
    header = tmp_path / "1.hea"
    with header.open("a") as stream:
        stream.write("#Diagnosis_DO_NOT_PARSE\n")
    original = wfdb.rdrecord

    def checked_read(path, **kwargs):
        assert "Diagnosis_DO_NOT_PARSE" not in Path(path + ".hea").read_text()
        return original(path, **kwargs)

    monkeypatch.setattr(wfdb, "rdrecord", checked_read)
    signal, metadata = ludb.read_signal(tmp_path / "1", tmp_path / "sanitized")
    np.testing.assert_allclose(signal[:, 0], np.arange(12)[::-1], atol=.001)
    assert metadata["canonical_lead_order"] == list(ludb.LEADS)
    assert metadata["source_lead_order"] == list(reversed(ludb.LEADS))


def test_bootstrap_keeps_patient_unit_and_exact_roles():
    rows = [{"record_id": index, "role": "parser_qualification" if index % 5 == 0 else "measurement",
             "AP": .6, "mean_shift_AP": .2, "gain": .4, "shift_AP": [.2] * 6, "prevalence": .1,
             "leads": [{"AP": .6, "mean_shift_AP": .2, "gain": .4, "prevalence": .1,
                        "constant_score": False}] * 12}
            for index in range(1, 201)]
    result = ludb.aggregate(rows)
    assert result["measurement_records"] == 160
    assert result["gain"] == pytest.approx(.4)
    np.testing.assert_allclose(result["gain_interval"], [.4, .4])
    assert result["scientific_passed"]
    with pytest.raises(ValueError, match="all200"):
        ludb.aggregate(rows[:-1])


def test_parser_role_never_computes_activity_or_ap(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(ludb, "read_signal", lambda record, scratch: (np.zeros((12, 5000)), {}))
    annotation = SimpleNamespace(sample=np.array([100, 110, 120, 140, 150, 160]),
                                 symbol=["(", "N", ")", "(", "t", ")"], chan=np.zeros(6, dtype=int))
    monkeypatch.setattr(ludb.wfdb, "rdann", lambda record, lead: annotation)

    def forbidden_activity(signal):
        raise AssertionError("Parser qualification cannot compute activity or AP")

    monkeypatch.setattr(ludb, "activity_proxy", forbidden_activity)
    row, _, proxies = ludb.evaluate_record(tmp_path / "5", tmp_path / "scratch")
    assert row["role"] == "parser_qualification"
    assert proxies is None
    assert "AP" not in row
    assert all("AP" not in lead for lead in row["leads"])


def test_shared_annotation_boundary_allowed_without_overlap_repair():
    waves = ludb.parse_waves(np.array([100, 110, 120, 120, 130, 140]),
                             ["(", "p", ")", "(", "N", ")"])
    assert len(waves) == 2
    mask, _ = ludb.targets(waves)
    assert mask[120] == 1
