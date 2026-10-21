import json
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
from scripts.validation import audit_ssl_synthetic_ludb_qualification as checker


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
                                    word(59, 0) + b"\0\0", word(63, 5) + b"a",
                                    word(62, 12) + b"\0\0"])
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
    assert metadata["source_baseline"] == [10] * 12
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
        metric = {"AP": .5, "mean_shift_AP": .3, "gain": .2, "prevalence": .1,
                  "shift_AP": [.3] * 6, "constant_score": False}
        rows.append({"record_id": number, **metric, "leads": [metric.copy() for _ in LEADS]})
    result = summary(rows)
    assert result["scientific_passed"]
    assert result["measurement_records"] == 160
    assert result["bootstrap_draws"] == 2000
    assert result["gain_interval"] == pytest.approx([.2, .2])


def cost_fixture(tmp_path, monkeypatch, transfers=2, versions=None):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return checker.file_sha256(path)

    temporal = tmp_path / "outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v2"
    base = {"current_machine_cumulative_cpu_seconds": 429.49696094999996,
            "current_machine_cumulative_gpu_wall_seconds": 447.98139213304967}
    prior_hash = save(temporal / "independent_audit_v1/prospective_ledger.json", base)
    monkeypatch.setattr(checker, "PRIOR_SHA", prior_hash)
    prefix = tmp_path / "outputs/ssl_synthetic_localization"
    parent = prefix / "ludb_acquisition_v1"
    steps = [(parent / "prospective_ledger.json", 19, .1),
             (parent / "independent_failure_audit_v1/prospective_ledger.json", 20, .02)]
    actual_versions = list(range(2, transfers + 1)) if versions is None else versions
    steps.extend((prefix / f"ludb_acquisition_v{version}/prospective_ledger.json", 21 + offset, .3)
                 for offset, version in enumerate(actual_versions))
    chain = []
    for path, stage, cpu in steps:
        receipt = {"status": "PASS" if stage == steps[-1][1] else "FAIL"}
        if stage == 20:
            receipt = {"all_passed": True, "mismatch_count": 0}
        receipt_sha = save(path.parent / "summary.json", receipt)
        base = {"current_machine_stages": stage, "prior_ledger_sha256": prior_hash,
                "cpu_seconds": cpu, "gpu_wall_seconds": 0., "sha256": receipt_sha,
                "current_machine_cumulative_cpu_seconds": (
                    base["current_machine_cumulative_cpu_seconds"] + cpu),
                "current_machine_cumulative_gpu_wall_seconds": (
                    base["current_machine_cumulative_gpu_wall_seconds"])}
        prior_hash = save(path, base)
        chain.append({"path": path.relative_to(tmp_path).as_posix(), "sha256": prior_hash})
        if stage == 20:
            monkeypatch.setattr(checker, "FAILURE_AUDIT_SHA", prior_hash)
    admission = {"transport_chain": chain,
                 "acquisition_ledger": {"path": chain[-1]["path"], "sha256": prior_hash},
                 "prior_ledger": {"path": chain[-1]["path"], "sha256": prior_hash,
                                  **{key: value for key, value in base.items()
                                     if key.startswith("current_machine_cumulative_")}}}
    study = tmp_path / "study"
    main = {"cpu_seconds_before_summary": .2, "wall_seconds_before_summary": .3}
    receipt_hash = save(study / "summary.json", main)
    ledger = {"receipt": "summary.json", "receipt_sha256": receipt_hash,
              "prior_ledger_sha256": prior_hash, "cpu_seconds": .4, "wall_seconds": .5,
              "gpu_wall_seconds": 0., "peak_rss_bytes": 200000,
              "prior_accounting_verified": True, "resource_gates": {"cpu": True},
              "current_machine_cumulative_cpu_seconds": base["current_machine_cumulative_cpu_seconds"] + .4,
              "current_machine_cumulative_gpu_wall_seconds": (
                    base["current_machine_cumulative_gpu_wall_seconds"])}
    save(study / "prospective_ledger.json", ledger)
    return study, main, admission, ledger


def test_full_failed_recovery_cost_lineage(tmp_path, monkeypatch):
    study, main, admission, _ = cost_fixture(tmp_path, monkeypatch)
    costs = checker.accounting(tmp_path, study, main, admission)
    assert costs["stages_through_main"] == 22
    assert costs["main_cpu_seconds"] == .4


def test_full_audit_rejects_duplicated_main_cost(tmp_path, monkeypatch):
    study, main, admission, ledger = cost_fixture(tmp_path, monkeypatch)
    ledger["current_machine_cumulative_cpu_seconds"] += .4
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="once"):
        checker.accounting(tmp_path, study, main, admission)


def test_second_failed_transfer_is_charged_without_reset(tmp_path, monkeypatch):
    study, main, admission, _ = cost_fixture(tmp_path, monkeypatch, transfers=3)
    costs = checker.accounting(tmp_path, study, main, admission)
    assert costs["stages_through_main"] == 23
    assert len(admission["transport_chain"]) == 4


def test_omitted_failed_transfer_is_rejected(tmp_path, monkeypatch):
    study, main, admission, _ = cost_fixture(tmp_path, monkeypatch, transfers=3)
    admission["transport_chain"].pop(2)
    with pytest.raises(ValueError, match="sequential|lineage"):
        checker.accounting(tmp_path, study, main, admission)


def test_unexecuted_protocol_version_has_no_fabricated_cost_stage(tmp_path, monkeypatch):
    study, main, admission, _ = cost_fixture(tmp_path, monkeypatch, versions=[2, 4])
    costs = checker.accounting(tmp_path, study, main, admission)
    assert costs["stages_through_main"] == 23
    assert len(admission["transport_chain"]) == 4


def split_fixture(tmp_path):
    text = ("User time (seconds): 0.10\nSystem time (seconds): 0.20\n"
            "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:03.50\n"
            "Maximum resident set size (kbytes): 2\nExit status: 0\n")
    (tmp_path / "child.txt").write_text(text)
    (tmp_path / "headers.txt").write_text('HTTP/1.1 200 Connection established\n\n'
                                         'HTTP/1.1 206 Partial Content\nContent-Range: bytes 3-5/6\n'
                                         'Content-Length: 3\nETag: "fixed"\n\n')
    (tmp_path / "suffix").write_bytes(b"def")
    (tmp_path / "prepare.json").write_text('{"status":"PASS"}')
    prepare = {"cpu_seconds": .25, "wall_seconds": .5, "peak_rss_bytes": 1000,
               "prepare_sha256": checker.file_sha256(tmp_path / "prepare.json")}
    (tmp_path / "prepare_costs.json").write_text(json.dumps(prepare))
    child = checker.child_measurements(text)
    receipt = {"phase_measurements": {"prepare": prepare, "curl": child,
                                      "finish_before_receipts": {"cpu_seconds": .7, "wall_seconds": 1.8,
                                                                 "peak_rss_bytes": 1500}},
               "peak_rss_bytes": 2048, "uninterrupted_whole_wall_claim": False,
               "external_sha256": {name: checker.file_sha256(tmp_path / name)
                                   for name in ("child.txt", "headers.txt", "suffix")}}
    ledger = {"finish_cpu_seconds": .75, "finish_wall_seconds": 2.,
              "cpu_seconds": (.25 + .75) + (.1 + .2), "wall_seconds": (.5 + 2.) + 3.5}
    (tmp_path / "prospective_ledger.json").write_text(json.dumps(ledger))
    config = {"curl": {"resource_log": "child.txt", "response_headers": "headers.txt",
                       "suffix_file": "suffix"},
              "resume": {"expected_complete_bytes": 6, "prefix_bytes": 3,
                         "required_content_range": "bytes 3-5/6", "if_range_etag": "fixed"},
              "resources": {"offline_prepare_and_finish_combined": {"cpu_seconds": 60,
                              "active_wall_seconds": 600},
                            "external_curl": {"cpu_seconds": 60, "wall_seconds": 3600},
                            "total": {"cpu_seconds": 120, "summed_active_wall_seconds": 4200,
                                      "rss_bytes": 2 * 1024**3}}}
    return receipt, config, ledger


def test_split_transfer_sums_actual_phases_without_approval_wait(tmp_path):
    receipt, config, _ = split_fixture(tmp_path)
    result = checker.split_phase_accounting(tmp_path, receipt, config)
    assert result["summed_cpu_seconds"] == 1.3
    assert result["summed_active_wall_seconds"] == 6.
    assert result["approval_wait_included"] is False


def test_split_transfer_rejects_fabricated_total_cpu(tmp_path):
    receipt, config, ledger = split_fixture(tmp_path)
    ledger["cpu_seconds"] = 1.
    (tmp_path / "prospective_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="terminal CPU"):
        checker.split_phase_accounting(tmp_path, receipt, config)


def test_split_transfer_rejects_unbound_child_log(tmp_path):
    receipt, config, _ = split_fixture(tmp_path)
    with (tmp_path / "child.txt").open("a") as stream:
        stream.write("modified\n")
    with pytest.raises(ValueError, match="binding"):
        checker.split_phase_accounting(tmp_path, receipt, config)


def test_split_transfer_rejects_duplicate_identity_header(tmp_path):
    receipt, config, _ = split_fixture(tmp_path)
    path = tmp_path / "headers.txt"
    path.write_text(path.read_text().replace('ETag: "fixed"', 'ETag: "fixed"\nETag: "changed"'))
    receipt["external_sha256"]["headers.txt"] = checker.file_sha256(path)
    with pytest.raises(ValueError, match="identity headers"):
        checker.split_phase_accounting(tmp_path, receipt, config)
