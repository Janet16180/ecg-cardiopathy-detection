import struct

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from ecg_experiment.ssl_synthetic_ludb_audit import (
    LEADS,
    annotation_events,
    average_precision,
    coverage_row,
    lead_row,
    record_row,
    summary,
    wave_triples,
    waveform,
)


def word(code, delta):
    return struct.pack("<H", (code << 10) | delta)


def synthetic_record(tmp_path, number=1, names=LEADS):
    record = tmp_path / str(number)
    digital = np.arange(60000, dtype=np.int32).reshape(5000, 12) % 2000
    record.with_suffix(".dat").write_bytes(digital.astype("<i2").tobytes())
    lines = [f"{number} 12 500 5000"]
    lines.extend(f"{number}.dat 16 1000(10)/mV 16 0 0 0 0 {lead}" for lead in names)
    record.with_suffix(".hea").write_text("\n".join([*lines, "# DO NOT PARSE CLINICAL CONTENT"]) + "\n")
    events = [(39, 100), (24, 10), (40, 10), (39, 80), (1, 10), (40, 10),
              (39, 80), (27, 10), (40, 10)]
    payload = b"".join(word(code, delta) for code, delta in events) + b"\0\0"
    for lead in LEADS:
        record.with_suffix("." + lead).write_bytes(payload)
    return record, digital


def test_binary_annotation_skip_auxiliary_and_persistent_fields():
    payload = (word(59, 0) + struct.pack("<HH", 0, 2000) + word(39, 5)
               + word(60, 1) + word(61, 2) + word(62, 3) + word(63, 3) + b"abc\0"
               + word(1, 10) + word(40, 10) + b"\0\0")
    assert annotation_events(payload) == [(2005, "("), (2015, "N"), (2025, ")")]


@pytest.mark.parametrize("payload", [b"\x01", word(2, 1) + b"\0\0", word(39, 2),
                                    word(59, 0) + b"\0\0", word(63, 5) + b"a"])
def test_invalid_binary_annotation_rejected(payload):
    with pytest.raises(ValueError, match="Annotation|annotation|Unexpected|wave|AP|timing"
    ):
        annotation_events(payload)


def test_wave_triples_shared_endpoint_and_strict_overlap():
    events = [(10, "("), (15, "p"), (20, ")"), (20, "("), (25, "N"), (30, ")")]
    waves = wave_triples(events)
    assert waves[1] == {"wave": "QRS", "onset": 20, "peak": 25, "offset": 30}
    with pytest.raises(ValueError, match="overlapping|timing"):
        wave_triples([*events[:3], (19, "("), (25, "N"), (30, ")")])


@pytest.mark.parametrize("events", [[(1, "(")], [(0, "("), (1, "N"), (5000, ")")],
                                   [(1, "("), (1, "N"), (1, ")")],
                                   [(1, "("), (2, "x"), (3, ")")]])
def test_invalid_triples_rejected(events):
    with pytest.raises(ValueError, match="Annotation|annotation|Unexpected|wave|AP|timing"
    ):
        wave_triples(events)


def test_independent_waveform_scaling_and_lead_mapping(tmp_path):
    names = LEADS[::-1]
    record, raw = synthetic_record(tmp_path, names=names)
    values, metadata = waveform(record)
    assert values.shape == (12, 5000)
    np.testing.assert_array_equal(values[0], (raw[:, 11] - 10) / 1000)
    assert metadata["source_lead_order"] == list(names)
    assert metadata["baseline"] == [10] * 12
    assert "CLINICAL" not in str(metadata)


def test_duplicate_header_lead_rejected(tmp_path):
    record, _ = synthetic_record(tmp_path, names=(*LEADS[:-1], "i"))
    with pytest.raises(ValueError, match="duplicate"):
        waveform(record)


def test_tie_grouped_ap_matches_library_and_constant_prevalence():
    target = np.array([0, 1, 1, 0, 1, 0])
    score = np.array([.8, .8, .8, .3, .3, .1])
    assert average_precision(target, score) == pytest.approx(average_precision_score(target, score))
    assert average_precision(target, np.zeros(6)) == .5
    with pytest.raises(ValueError, match="Annotation|annotation|Unexpected|wave|AP|timing"
    ):
        average_precision(np.zeros(6), score)


def test_inclusive_target_and_fixed_shift_controls():
    waves = wave_triples([(100, "("), (110, "p"), (120, ")"),
                          (200, "("), (210, "N"), (220, ")"),
                          (300, "("), (310, "t"), (320, ")")])
    values = np.zeros(5000)
    metadata, target = coverage_row(values, waves)
    assert metadata["coverage_samples"] == 221
    assert metadata["qrs_samples"] == 21
    assert target.sum() == 21
    row = lead_row(values, waves)
    assert row["AP"] == 21 / 221
    assert row["gain"] == pytest.approx(0)
    assert row["shift_samples"] == [37, 68, 95, 130, 156, 183]


def test_parser_records_have_no_scores(tmp_path):
    record, _ = synthetic_record(tmp_path, number=5)
    row = record_row(record)
    assert row["role"] == "parser_qualification"
    assert "AP" not in row
    assert "gain" not in row
    assert len(row["leads"]) == 12
    assert all("AP" not in lead and "shift_AP" not in lead for lead in row["leads"])


def test_whole_record_bootstrap_and_decision():
    rows = []
    for number in range(1, 201):
        metric = {"AP": .5, "mean_shift_AP": .3, "gain": .2, "prevalence": .1}
        rows.append({"record_id": number, **metric, "leads": [metric.copy() for _ in LEADS]})
    result = summary(rows)
    assert result["scientific_passed"]
    assert result["measurement_records"] == 160
    assert result["bootstrap_draws"] == 2000
    assert result["gain_interval"] == pytest.approx([.2, .2])
