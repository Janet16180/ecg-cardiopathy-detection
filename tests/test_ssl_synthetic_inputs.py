"""Synthetic-only checks for permitted waveform admission and protected-role isolation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import wfdb

from ecg_experiment import ssl_synthetic_inputs as audit
from ecg_experiment.waveforms import LEADS
from scripts.validation import audit_ssl_synthetic_inputs as command


def row(role="train"):
    return {
        "ecg_id": "artificial",
        "patient_id": "artificial",
        "role": role,
        "filename_hr": "records500/artificial",
    }


def source(tmp_path, *, units=None, names=None, fs=500, samples=5000, signal=None):
    target = tmp_path / "records500"
    target.mkdir()
    if signal is None:
        time = np.arange(samples) / fs
        signal = np.sin(time * 2 * np.pi * 2)[:, None] * np.ones((1, 12)) * 0.2
    wfdb.wrsamp(
        "artificial",
        fs=fs,
        units=units or ["mV"] * 12,
        sig_name=names or list(LEADS),
        p_signal=signal,
        fmt=["16"] * 12,
        write_dir=str(target),
    )
    return {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in target.iterdir()
    }


@pytest.mark.parametrize("role", audit.PROTECTED_ROLES)
def test_protected_roles_return_before_access(role, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A protected role attempted payload access")

    monkeypatch.setattr(audit, "verify_files", forbidden)
    monkeypatch.setattr(audit, "checked_relative", forbidden)
    result = audit.audit_record(row(role), Path("/absent"), {})
    assert result["status"] == "protected_unread"
    assert result["payload_sha256"] == {}
    assert result["train_optimization_eligible"] is None


@pytest.mark.parametrize("role", ["typo", "test"])
def test_unknown_role_rejected_without_access(role):
    with pytest.raises(ValueError, match="Unknown partition role"):
        audit.audit_record(row(role), Path("/absent"), {})


def test_checksum_corruption_is_hard_failure_before_decode(tmp_path, monkeypatch):
    checksums = source(tmp_path)
    with (tmp_path / "records500/artificial.dat").open("ab") as stream:
        stream.write(b"corrupt")
    monkeypatch.setattr(audit, "read_record", lambda *_: pytest.fail("Decoded corrupted payload"))
    result = audit.audit_record(row(), tmp_path, checksums)
    assert result["status"] == "integrity_failed"
    assert "checksum_mismatch" in result["failure_detail"]
    assert ".dat" in result["payload_sha256"]
    assert not result["train_optimization_eligible"]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"units": ["uV"] * 12}, "invalid_units"),
        ({"names": ["other", *LEADS[1:]]}, "invalid_leads"),
        ({"fs": 250}, "invalid_header_geometry"),
        ({"samples": 2500}, "invalid_header_geometry"),
    ],
)
def test_header_integrity_failures(tmp_path, kwargs, reason):
    result = audit.audit_record(row(), tmp_path, source(tmp_path, **kwargs))
    assert result["status"] == "integrity_failed"
    assert reason in result["failure_detail"]


def test_header_cannot_redirect_unverified_payload(tmp_path):
    checksums = source(tmp_path)
    header = tmp_path / "records500/artificial.hea"
    header.write_text(header.read_text().replace("artificial.dat", "elsewhere.dat"))
    checksums["records500/artificial.hea"] = hashlib.sha256(header.read_bytes()).hexdigest()
    result = audit.audit_record(row(), tmp_path, checksums)
    assert "unverified_payload_reference" in result["failure_detail"]


def test_quality_exclusion_only_removes_training_eligibility(tmp_path):
    checksums = source(tmp_path, signal=np.zeros((5000, 12)))
    train = audit.audit_record(row(), tmp_path, checksums)
    validation = audit.audit_record(row("validation"), tmp_path, checksums)
    assert train["status"] == validation["status"] == "decoded"
    assert "constant_lead" in train["quality_exclusions"]
    assert not train["train_optimization_eligible"]
    assert validation["validation_retained"]
    assert validation["train_optimization_eligible"] is None
    aggregate = audit.aggregate_records([train, validation])
    assert aggregate["input_integrity_gate_passed"]
    assert aggregate["by_role"]["validation"]["validation_retained"] == 1
    assert not aggregate["training_admitted"]


def test_review_only_flags_preserve_training_eligibility(tmp_path):
    result = audit.audit_record(row(), tmp_path, source(tmp_path))
    assert result["quality_exclusions"] == []
    assert "limb_identity_violated" in result["review_flags"]
    assert result["train_optimization_eligible"]


def test_nonfinite_decode_rejects_gate(tmp_path, monkeypatch):
    checksums = source(tmp_path)

    def fail(*_):
        raise ValueError("Nonfinite waveform")

    monkeypatch.setattr(audit, "read_record", fail)
    result = audit.audit_record(row("validation"), tmp_path, checksums)
    assert result["validation_retained"]
    assert not audit.aggregate_records([result])["input_integrity_gate_passed"]


def test_missing_file_retained(tmp_path):
    checksums = {"records500/artificial.hea": "0" * 64}
    result = audit.audit_record(row(), tmp_path, checksums)
    assert result["integrity_failures"] == ["integrity:FileNotFoundError"]


def test_output_manifest_preserves_all_roles_and_hashes(tmp_path):
    records = [audit.audit_record(row(role), Path("/absent"), {}) for role in audit.PROTECTED_ROLES]
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = audit.write_audit(records, tmp_path, {"manifest_sha256": "frozen"})
    manifest = json.loads((tmp_path / "eligibility.json").read_text())
    assert manifest["records"] == records
    assert not manifest["base_partition_changed"]
    assert set(summary["output_sha256"]) == {"eligibility.json", "records.jsonl"}
    with pytest.raises(FileExistsError):
        audit.write_audit(records, tmp_path, {})


def test_existing_destination_refused_before_input_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "OUTPUT", tmp_path)
    monkeypatch.setattr("sys.argv", ["audit"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        command, "load_bound_inputs", lambda: pytest.fail("Read inputs before overwrite check")
    )
    with pytest.raises(FileExistsError, match="must be new"):
        command.main()


def test_units_must_be_explicit_even_when_parser_defaults_to_mv(tmp_path):
    checksums = source(tmp_path)
    header = tmp_path / "records500/artificial.hea"
    header.write_text(header.read_text().replace("/mV", ""))
    checksums["records500/artificial.hea"] = hashlib.sha256(header.read_bytes()).hexdigest()
    result = audit.audit_record(row(), tmp_path, checksums)
    assert result["status"] == "integrity_failed"
    assert "invalid_units" in result["failure_detail"]


def test_leads_are_reordered_to_canonical_before_quality(tmp_path, monkeypatch):
    order = list(reversed(LEADS))
    time = np.arange(5000) / 500
    signal = np.sin(time * 2 * np.pi * 2)[:, None] * np.arange(1, 13)[None, :] * 0.1
    checksums = source(tmp_path, names=order, signal=signal)
    observed = []

    def capture(waveform):
        observed.append(waveform)
        return [], []

    monkeypatch.setattr(audit, "assess", capture)
    result = audit.audit_record(row(), tmp_path, checksums)
    assert result["status"] == "decoded"
    assert np.allclose(observed[0], signal[:, ::-1].T, atol=0.0001)


def test_malformed_header_and_missing_checksum_are_retained(tmp_path):
    checksums = source(tmp_path)
    header = tmp_path / "records500/artificial.hea"
    header.write_text("invalid\n")
    checksums["records500/artificial.hea"] = hashlib.sha256(header.read_bytes()).hexdigest()
    result = audit.audit_record(row(), tmp_path, checksums)
    assert "invalid_header_layout" in result["failure_detail"]
    result = audit.audit_record(row(), tmp_path, {})
    assert "missing_official_checksum" in result["failure_detail"]


def test_path_traversal_rejected_before_access(tmp_path, monkeypatch):
    record = dict(row(), filename_hr="../protected")
    monkeypatch.setattr(audit, "verify_files", lambda *_: pytest.fail("Accessed invalid path"))
    result = audit.audit_record(record, tmp_path, {})
    assert result["status"] == "integrity_failed"
