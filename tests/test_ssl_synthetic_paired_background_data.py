"""Artificial-only selection, role, rendering and admission tests for paired real backgrounds."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import wfdb

from ecg_experiment import ssl_synthetic_paired_background_data as data
from ecg_experiment import ssl_synthetic_paired_background_preparation as prep
from ecg_experiment import ssl_synthetic_real_background as old
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.resample import resample_full
from ecg_experiment.waveforms import LEADS


@pytest.fixture
def config():
    return json.loads(Path("configs/ssl_synthetic_paired_background_v1.json").read_bytes())


def record(patient, ecg, role):
    return {
        "patient_id": str(patient),
        "ecg_id": str(ecg),
        "role": role,
        "filename_hr": "records500/artificial",
    }


def audit(rows):
    return [
        dict(
            row,
            train_optimization_eligible=row["role"] == "train",
            quality_exclusions=[],
            review_flags=[],
            payload_sha256={".hea": "header", ".dat": "payload"},
        )
        for row in rows
    ]


def artificial_selection():
    reserved = [
        dict(
            record(i, i, "train"),
            study_role="train_reserved",
            cache_index=None,
            subject_index=None,
            audit_payload_sha256={},
            quality_exclusions=[],
            review_flags=[],
        )
        for i in range(1, 501)
    ]
    spent = [dict(record(i, i, "validation"), study_role="calibration") for i in range(1000, 1192)]
    remaining = [record(i, i, "validation") for i in range(2000, 2300)]
    rows = (
        [record(i, i, "train") for i in range(1, 501)]
        + [record(i, i, "validation") for i in range(1000, 1192)]
        + remaining
    )
    rows += [record(2000, 9999, "validation"), record(99999, 99999, "engineering_test")]
    return rows, {"records": reserved + spent}


def study_row(role="train", index=None):
    index = {"train": 0, "calibration": 500, "evaluation": 564}[role] if index is None else index
    return dict(
        record(900001, 1, "train" if role == "train" else "validation"),
        study_role=role,
        cache_index=index,
        subject_index=index,
        role_subject_index=0,
    )


def background():
    t = np.arange(2500) / 250
    signal = sum(np.exp(-0.5 * ((t - center) / 0.012) ** 2) for center in np.arange(0.6, 9.6, 0.5))
    return (signal[None] * np.linspace(0.4, 1.5, 12)[:, None] + 0.03 * np.sin(2 * np.pi * t)[None]).astype(
        np.float32
    )


def test_exact_reservation_and_unfiltered_new_validation_selection():
    rows, previous = artificial_selection()
    audited = audit(rows)
    for item in audited:
        if item["role"] == "validation":
            item["quality_exclusions"] = ["retained_quality_flag"]
    first = data.select_rows(rows, audited, previous)
    assert first == data.select_rows(list(reversed(rows)), audited, previous)
    assert first["original_train_reservation_rows"] == previous["records"][:500]
    assert first["original_train_reservation_sha256"] == sha256_json(previous["records"][:500])
    selected = first["records"]
    assert [r["cache_index"] for r in selected] == list(range(692))
    assert [r["patient_id"] for r in selected[:500]] == [str(i) for i in range(1, 501)]
    assert len({r["patient_id"] for r in selected}) == 692
    assert not {r["patient_id"] for r in selected} & {str(i) for i in range(1000, 1192)}
    expected = sorted(
        range(2000, 2300),
        key=lambda i: (hashlib.sha256(f"ssl_paired_background_v1:validation:{i}".encode()).hexdigest(), i),
    )[:192]
    assert [r["patient_id"] for r in selected[500:]] == list(map(str, expected))
    assert all(r["ecg_id"] == r["patient_id"] for r in selected[500:])
    assert all(r["quality_exclusions"] == ["retained_quality_flag"] for r in selected[500:])
    assert first["waveforms_opened_when_written"] == 0


def test_reserved_patient_cannot_be_changed_or_excluded():
    rows, previous = artificial_selection()
    bad = copy.deepcopy(previous)
    bad["records"][0]["patient_id"] = "new"
    with pytest.raises(ValueError, match="reservation changed"):
        data.select_rows(rows, audit(rows), bad)
    audited = audit(rows)
    audited[0]["train_optimization_eligible"] = False
    with pytest.raises(ValueError, match="reservation changed"):
        data.select_rows(rows, audited, previous)


@pytest.mark.parametrize(
    ("source", "role", "index"),
    [
        ("engineering_test", "evaluation", 564),
        ("excluded", "train", 0),
        ("validation", "train", 0),
        ("train", "calibration", 500),
        ("validation", "evaluation", 563),
        ("train", "train", 500),
        ("train", "other", 0),
    ],
)
def test_protected_roles_and_global_mapping_rejected_before_file_access(source, role, index, monkeypatch):
    row = dict(record(1, 1, source), study_role=role, cache_index=index)
    monkeypatch.setattr(data, "checked_relative", lambda *_: pytest.fail("Protected path constructed"))
    monkeypatch.setattr(data, "verify_files", lambda *_: pytest.fail("Protected payload opened"))
    with pytest.raises(ValueError, match="Protected/unselected"):
        data.prepare_background(row, {})


@pytest.mark.parametrize("role", ["train", "calibration", "evaluation"])
def test_role_seed_adapter_preserves_frozen_math_and_global_cases(config, role):
    source = background()
    row = study_row(role)
    anchors = old.find_anchors(source, config)["anchors"]
    cases = data.cases_for_background(row, anchors, old.array_sha256(source), config)
    expected_seed = int.from_bytes(
        hashlib.sha256(f"ssl_paired_background_v1:{role}:900001".encode()).digest()[:8], "big"
    )
    assert [case["case_index"] for case in cases] == list(
        range(4 * row["cache_index"], 4 * row["cache_index"] + 4)
    )
    assert [case["variant"] for case in cases] == ["focal", "repeated", "sham", "nuisance"]
    for case in cases:
        assert case["role"] == role
        assert case["source_partition_role"] == row["role"]
        assert case["seed"] == expected_seed
        assert case["profile"] == ("B" if role == "evaluation" else "A")
        actual = data.render_case(source, case)
        legacy = old.render_case(
            source, dict(case, role="evaluation" if role == "evaluation" else "calibration")
        )
        assert not old.qualify_pair(actual, case, config)["failures"]
        assert all(
            np.array_equal(actual[k], legacy[k])
            for k in ("edited", "baseline", "support", "candidate", "analytic_delta")
        )
        assert case["role"] == role
    supports, candidates = {}, {}
    qualified = prep.qualify_cases(source, row, anchors, config, supports, candidates)
    assert len(supports) == len(candidates) == 4
    for case in qualified:
        assert case["deterministic_replay_passed"]
        assert not case["qualification"]["failures"]
        for key in ("edited", "baseline", "support", "candidate"):
            assert len(case[key + "_sha256"]) == 64
    for case in qualified[:2]:
        control = qualified[3 if case["nuisance"] else 2]
        assert case["matched_control_case_index"] == control["case_index"]
        assert case["baseline_sha256"] == control["baseline_sha256"]


def test_native_train_decode_checksum_units_and_millivolts(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    raw = tmp_path / "data/raw/ptb-xl/1.0.3"
    folder = raw / "records500"
    folder.mkdir(parents=True)
    t = np.arange(5000) / 500
    signal = np.sin(2 * np.pi * 2 * t)[:, None] * np.linspace(0.1, 0.9, 12)[None]
    wfdb.wrsamp(
        "artificial",
        fs=500,
        units=["mV"] * 12,
        sig_name=list(LEADS),
        p_signal=signal,
        fmt=["16"] * 12,
        write_dir=str(folder),
    )
    checks = {p.relative_to(raw).as_posix(): sha256_file(p) for p in folder.iterdir()}
    row = study_row()
    row["audit_payload_sha256"] = {ext: checks["records500/artificial" + ext] for ext in (".hea", ".dat")}
    decoded, _ = data.prepare_background(row, checks)
    assert decoded.dtype == np.float32
    assert decoded.shape == (12, 2500)
    assert np.allclose(decoded, resample_full(signal.T, 1, 2), atol=1e-4)
    header = folder / "artificial.hea"
    header.write_text(header.read_text().replace("/mV", "/uV"))
    checks["records500/artificial.hea"] = sha256_file(header)
    row["audit_payload_sha256"][".hea"] = checks["records500/artificial.hea"]
    with pytest.raises(ValueError, match="units"):
        data.prepare_background(row, checks)
    with (folder / "artificial.dat").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum_mismatch"):
        data.prepare_background(row, checks)


def test_forecast_overwrite_and_cumulative_cpu_guard(tmp_path, monkeypatch, config):
    assert prep.storage_forecast() == {
        "array_payload_bytes": 433440000,
        "array_header_bytes": 640,
        "metadata_reserve_bytes": 67108864,
        "complete_study_new_bytes": 500549504,
    }
    monkeypatch.setattr(prep, "OUTPUT", tmp_path)
    monkeypatch.setattr(
        prep, "metadata_inputs", lambda: pytest.fail("Metadata read before overwrite refusal")
    )
    with pytest.raises(FileExistsError):
        prep.prepare()
    monkeypatch.setattr(prep.time, "process_time", lambda: 100)
    with pytest.raises(ValueError, match="CPU ceiling"):
        prep.check_budget(99, 14399.5, config)


def test_complete_selection_frozen_before_processing(tmp_path, monkeypatch, config):
    rows, previous = artificial_selection()
    selected = data.select_rows(rows, audit(rows), previous)
    output = tmp_path / "new"
    monkeypatch.setattr(prep, "OUTPUT", output)
    monkeypatch.setattr(prep, "SOURCE_FILES", ())
    monkeypatch.setattr(prep, "stream_bytes", lambda: 0)
    monkeypatch.setattr(
        prep,
        "metadata_inputs",
        lambda: (selected, {}, {"input_sha256": {}, "prior_cumulative_cpu_seconds": 0}, config),
    )

    def inspect_first(*_):
        frozen = json.loads((output / "selection.json").read_bytes())
        assert len(frozen["records"]) == 692
        assert len(frozen["original_train_reservation_rows"]) == 500
        assert frozen["waveforms_opened_when_written"] == 0
        raise RuntimeError("artificial stop before payload")

    monkeypatch.setattr(prep, "process_rows", inspect_first)
    with pytest.raises(RuntimeError, match="artificial stop"):
        prep.prepare()


def test_reproduction_values_and_conflicting_digest_rejected():
    assert prep.compare_reproduction({"value": [0.1, True, None]}, {"value": [0.1 + 1e-13, True, None]}) == 3
    with pytest.raises(ValueError, match="aggregate changed"):
        prep.compare_reproduction({"value": 0.1}, {"value": 0.101})
    with pytest.raises(ValueError, match="Conflicting"):
        prep.merge_unique({"path": "old"}, {"path": "new"})


def test_first_integrity_failure_retained_stops_further_payload_reads(monkeypatch, config):
    class EmptyMap:
        def __setitem__(self, *_):
            pass

        def __getitem__(self, _):
            return np.full((12, 2500), np.nan, dtype=np.float32)

        def flush(self):
            pass

    opened = []

    def fail_first(row, _):
        opened.append(row["cache_index"])
        raise ValueError("artificial integrity failure")

    monkeypatch.setattr(prep.np.lib.format, "open_memmap", lambda *_a, **_kw: EmptyMap())
    monkeypatch.setattr(prep, "prepare_background", fail_first)
    metadata, cases = prep.process_rows([study_row(index=0), study_row(index=1)], {}, config, 0, 0)
    assert opened == [0]
    assert len(metadata) == 2
    assert all(row["status"] == "processing_failed" for row in metadata)
    assert len(cases) == 8
    assert all(case["qualification"]["failures"] for case in cases)
