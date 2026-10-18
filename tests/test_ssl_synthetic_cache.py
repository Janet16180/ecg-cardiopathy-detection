"""Artificial-input tests for the fixed patient subset and physical TRAIN cache."""

import hashlib
import json

import numpy as np
import pytest
import wfdb

from ecg_experiment import ssl_synthetic_cache as cache
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.resample import resample_full
from ecg_experiment.waveforms import LEADS
from scripts.data import prepare_ssl_synthetic_real_cache as command


def record(patient, ecg, role="train"):
    return {
        "patient_id": str(patient),
        "ecg_id": str(ecg),
        "role": role,
        "filename_hr": "records500/artificial",
    }


def audited(rows, excluded_ids=()):
    return [
        dict(
            row,
            status="decoded",
            train_optimization_eligible=row["ecg_id"] not in excluded_ids,
            quality_exclusions=["flat_segment"] if row["ecg_id"] in excluded_ids else [],
            review_flags=[],
            payload_sha256={},
        )
        for row in rows
    ]


def artificial_source(tmp_path):
    raw = tmp_path / "raw"
    destination = raw / "records500"
    destination.mkdir(parents=True)
    time = np.arange(5000) / 500
    signal = 0.5 + np.sin(time * 2 * np.pi * 2)[:, None] * np.arange(1, 13)[None, :] * 0.1
    wfdb.wrsamp(
        "artificial",
        fs=500,
        units=["mV"] * 12,
        sig_name=list(LEADS),
        p_signal=signal,
        fmt=["16"] * 12,
        write_dir=str(destination),
    )
    hashes = {str(path.relative_to(raw)): sha256_file(path) for path in destination.iterdir()}
    row = audited([record(1, 1)])[0]
    row["payload_sha256"] = {
        extension: hashes["records500/artificial" + extension] for extension in (".hea", ".dat")
    }
    selection = cache.join_eligibility([record(1, 1)], [row])
    return raw, hashes, selection, signal


def test_subset_is_stable_and_selects_lowest_record_before_eligibility():
    rows = [record(patient, 100 + patient) for patient in range(1, 8)]
    rows += [record(3, 3), record(99, 99, "validation")]
    first = cache.select_records(rows, "frozen", 3)
    reversed_selection = cache.select_records(list(reversed(rows)), "frozen", 3)
    assert first == reversed_selection
    expected = sorted(range(1, 8), key=lambda p: (hashlib.sha256(f"frozen:{p}".encode()).hexdigest(), p))[:3]
    assert [int(row["patient_id"]) for row in first] == expected
    for row in first:
        if row["patient_id"] == "3":
            assert row["ecg_id"] == "3"
    assert all(row["role"] == "train" for row in first)


def test_failed_selected_record_is_skipped_without_replacement_or_alternate_ecg():
    rows = [record(patient, patient) for patient in range(1, 9)]
    selected = cache.select_records(rows, "frozen", 3)
    failed = selected[0]
    extra_same_patient = record(failed["patient_id"], 1000)
    rows.append(extra_same_patient)
    selected_again = cache.select_records(rows, "frozen", 3)
    assert selected_again == selected
    plan = cache.join_eligibility(selected_again, audited(rows, [failed["ecg_id"]]))
    assert len(plan) == 3
    assert plan[0]["status"] == "quality_skipped"
    assert plan[0]["ecg_id"] == failed["ecg_id"]
    assert sum(row["status"] == "pending" for row in plan) == 2


def test_selection_hash_is_original_ordered_identity():
    rows = [record(patient, patient) for patient in range(1, 5)]
    selected = cache.select_records(rows, "frozen", 3)
    original_hash = cache.selection_identity(selected)
    plan = cache.join_eligibility(selected, audited(rows, [selected[0]["ecg_id"]]))
    assert original_hash == sha256_json(selected)
    assert cache.selection_identity(selected) == original_hash
    assert sha256_json(plan) != original_hash


@pytest.mark.parametrize("role", ["validation", "engineering_test", "excluded"])
def test_protected_and_nontraining_roles_rejected_before_payload_access(role, monkeypatch, tmp_path):
    row = dict(record(1, 1, role), status="pending", quality_exclusions=[])
    monkeypatch.setattr(cache, "checked_relative", lambda *_: pytest.fail("Constructed protected path"))
    monkeypatch.setattr(cache, "verify_files", lambda *_: pytest.fail("Opened protected payload"))
    with pytest.raises(ValueError, match="Only selected eligible TRAIN"):
        cache.prepare_signal(row, tmp_path, {})
    with pytest.raises(ValueError, match="nontraining record"):
        cache.prepare_cache([row], tmp_path, {}, tmp_path / "output", 1)
    assert not (tmp_path / "output").exists()


def test_native_cache_shape_and_physical_mv_preserved(tmp_path):
    raw, hashes, selection, signal = artificial_source(tmp_path)
    output = tmp_path / "cache"
    summary = cache.prepare_cache(selection, raw, hashes, output, 1)
    assert summary["cache_gate_passed"]
    assert not summary["training_admitted"]
    assert summary["usable_patients"] == 1
    assert summary["units"] == "mV"
    assert not summary["physical_scale_applied"]
    result = np.load(output / "signals.npy", mmap_mode="r")
    assert result.dtype == np.float32
    assert result.shape == (1, 12, 2500)
    assert np.allclose(result[0], resample_full(signal.T, 1, 2), atol=0.0001)
    assert selection[0]["cache_index"] == 0
    assert selection[0]["payload_sha256"] == selection[0]["audit_payload_sha256"]
    assert (
        sha256_file(output / "signals.npy")
        == hashlib.sha256((output / "signals.npy").read_bytes()).hexdigest()
    )


def test_fresh_corruption_rejects_cache_and_retains_failure(tmp_path):
    raw, hashes, selection, _ = artificial_source(tmp_path)
    with (raw / "records500/artificial.dat").open("ab") as stream:
        stream.write(b"corrupt")
    summary = cache.prepare_cache(selection, raw, hashes, tmp_path / "cache", 1)
    assert not summary["cache_gate_passed"]
    assert summary["integrity_failed_patients"] == 1
    assert selection[0]["status"] == "integrity_failed"
    assert "checksum_mismatch" in selection[0]["failure_detail"]
    assert selection[0]["cache_index"] is None


def test_original_audit_hash_mismatch_rejects_before_decode(tmp_path, monkeypatch):
    raw, hashes, selection, _ = artificial_source(tmp_path)
    selection[0]["audit_payload_sha256"][".dat"] = "0" * 64
    monkeypatch.setattr(cache, "read_record", lambda *_: pytest.fail("Decoded stale audit input"))
    summary = cache.prepare_cache(selection, raw, hashes, tmp_path / "cache", 1)
    assert not summary["cache_gate_passed"]
    assert "original input audit" in selection[0]["failure_detail"]


def test_quality_skip_never_opens_and_insufficient_usable_rejects(tmp_path, monkeypatch):
    raw, hashes, selection, _ = artificial_source(tmp_path)
    skipped = dict(
        selection[0],
        ecg_id="2",
        patient_id="2",
        status="quality_skipped",
        quality_exclusions=["flat_segment"],
    )
    original = cache.prepare_signal

    def guard(row, *args):
        assert row["ecg_id"] != "2"
        return original(row, *args)

    monkeypatch.setattr(cache, "prepare_signal", guard)
    summary = cache.prepare_cache([selection[0], skipped], raw, hashes, tmp_path / "cache", 2)
    assert not summary["cache_gate_passed"]
    assert summary["usable_patients"] == 1
    assert summary["quality_skipped_patients"] == 1
    assert summary["quality_skip_reasons"] == {"flat_segment": 1}
    assert skipped["cache_index"] is None


def test_missing_units_rejected_even_with_current_byte_hashes(tmp_path):
    raw, hashes, selection, _ = artificial_source(tmp_path)
    path = raw / "records500/artificial.hea"
    path.write_text(path.read_text().replace("/mV", ""))
    hashes["records500/artificial.hea"] = sha256_file(path)
    selection[0]["audit_payload_sha256"][".hea"] = sha256_file(path)
    summary = cache.prepare_cache(selection, raw, hashes, tmp_path / "cache", 1)
    assert not summary["cache_gate_passed"]
    assert "invalid_units" in selection[0]["failure_detail"]


def test_existing_cache_never_overwritten_before_access(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "prepare_signal", lambda *_: pytest.fail("Read source during overwrite"))
    with pytest.raises(FileExistsError, match="must be new"):
        cache.prepare_cache([], tmp_path, {}, tmp_path, 1)


def test_existing_runner_destination_rejected_before_binding_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "OUTPUT", tmp_path)
    monkeypatch.setattr("sys.argv", ["prepare"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        command, "load_audited_inputs", lambda *_: pytest.fail("Read inputs before overwrite")
    )
    with pytest.raises(FileExistsError, match="must be new"):
        command.main()


def test_input_audit_summary_hash_is_checked_before_payload_parsing(tmp_path):
    directory = tmp_path / "outputs/ssl_synthetic_localization/input_audit_v1"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps({"input_integrity_gate_passed": True}))
    with pytest.raises(ValueError, match="differs from the frozen cache input"):
        cache.load_audited_inputs(tmp_path)


def test_selection_requires_canonical_patient_and_full_original_count():
    with pytest.raises(ValueError, match="canonical"):
        cache.select_records([record("01", 1)], "frozen", 1)
    with pytest.raises(ValueError, match="Insufficient"):
        cache.select_records([record(1, 1)], "frozen", 2)
