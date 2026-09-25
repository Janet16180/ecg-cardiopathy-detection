"""Adversarial checks of held-out raw integrity and exact signal isolation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment import clean_rerun
from ecg_experiment.public_sources import signal_sha256


def _raw_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, str]], np.ndarray]:
    """Create checksum-pinned toy raw pairs and a canonical decoded signal."""
    raw = root / "data/raw/ptb-xl/1.0.3"
    raw.mkdir(parents=True)
    lines = []
    references = []
    (raw / "records500").mkdir()
    for index in (1, 2):
        stem = f"records500/{index}_hr"
        for suffix in (".hea", ".dat"):
            content = f"record-{index}-{suffix}".encode()
            (raw / f"{stem}{suffix}").write_bytes(content)
            lines.append(f"{hashlib.sha256(content).hexdigest()} {stem}{suffix}")
        references.append({"record_id": f"ptbxl:{index}", "split": "development" if index == 1 else "test",
                           "raw_path": f"data/raw/ptb-xl/1.0.3/{stem}"})
    (raw / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    signal = np.tile(np.arange(5000, dtype=np.float32), (12, 1))
    monkeypatch.setattr(clean_rerun.wfdb, "rdheader", lambda _: type("Header", (), {"units": ["mV"] * 12})())
    monkeypatch.setattr(clean_rerun, "read_record", lambda _raw, stem: signal + ("2_hr" in stem))
    return references, signal


def test_heldout_flags_are_descriptive_and_all_rows_remain(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """A constant held-out lead is retained and marked for review."""
    references, signal = _raw_fixture(tmp_path, monkeypatch)
    signal[0] = 0
    audit = clean_rerun.audit_heldout(references, set(), tmp_path)
    assert len(audit) == 2
    assert "constant_I" in audit[0]["review_flags"]
    assert audit[0]["signal_sha256"] != audit[1]["signal_sha256"]


def test_heldout_rejects_training_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared verified train hash blocks exact held-out reuse."""
    references, signal = _raw_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicates training"):
        clean_rerun.audit_heldout(references, {signal_sha256(signal)}, tmp_path)


def test_heldout_rejects_raw_byte_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An altered official data file fails before any waveform is accepted."""
    references, _ = _raw_fixture(tmp_path, monkeypatch)
    (tmp_path / "data/raw/ptb-xl/1.0.3/records500/1_hr.dat").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Official PTB checksum mismatch"):
        clean_rerun.audit_heldout(references, set(), tmp_path)


def test_heldout_rejects_wrong_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Physical units are checked before canonical decoding."""
    references, _ = _raw_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(clean_rerun.wfdb, "rdheader", lambda _: type("Header", (), {"units": ["uV"] * 12})())
    with pytest.raises(ValueError, match="units"):
        clean_rerun.audit_heldout(references, set(), tmp_path)


def test_heldout_rejects_between_split_duplicate(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Two different held-out partitions cannot share an exact signal."""
    references, signal = _raw_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(clean_rerun, "read_record", lambda _raw, _stem: signal)
    with pytest.raises(ValueError, match="another held-out"):
        clean_rerun.audit_heldout(references, set(), tmp_path)


def test_limited_label_patient_and_target_must_match_full() -> None:
    """Matching record IDs alone cannot hide a changed patient or target."""
    full = [{"record_id": "ptbxl:1", "patient_id": "ptbxl:p1", "target": "1"}]
    clean_rerun.verify_label_nesting(full, full.copy())
    for changed in ({"record_id": "ptbxl:1", "patient_id": "ptbxl:p2", "target": "1"},
                    {"record_id": "ptbxl:1", "patient_id": "ptbxl:p1", "target": "0"}):
        with pytest.raises(ValueError, match="patient/target"):
            clean_rerun.verify_label_nesting(full, [changed])


def test_known_patient_cannot_cross_heldout_partitions() -> None:
    """Patient leakage across train and held-out or two held-out splits fails."""
    selected = [{"source": "ptbxl", "patient_id": "ptbxl:p1"},
                {"source": "mimic", "patient_id": "mimic:m1"}]
    clean_rerun.verify_patient_partitions(selected, [{"split": "development", "patient_id": "ptbxl:p2"}])
    with pytest.raises(ValueError, match="overlap"):
        clean_rerun.verify_patient_partitions(selected,
            [{"split": "test", "patient_id": "ptbxl:p1"}])
    with pytest.raises(ValueError, match="overlap"):
        clean_rerun.verify_patient_partitions(selected,
            [{"split": "development", "patient_id": "ptbxl:p2"},
             {"split": "calibration", "patient_id": "ptbxl:p2"}])


def test_wrong_union_lead_order_is_rejected(tmp_path: Path) -> None:
    """A receipt-pinned table set cannot make incorrect lead metadata canonical."""
    import json

    from ecg_experiment.files import sha256_file

    dataset = tmp_path / "dataset"
    release = tmp_path / "release"
    dataset.mkdir()
    release.mkdir()
    hashes = {}
    for name in clean_rerun.TABLES:
        path = dataset / name
        path.write_text("record_id\n")
        hashes[name] = sha256_file(path)
    metadata = {"complete": True, "schema_version": 1, "record_count": 76598,
                "shape_per_record": [12, 5000], "dtype": "float32", "sampling_rate_hz": 500,
                "units": "mV", "lead_order": list(reversed(clean_rerun.LEADS)),
                "source_counts": {"ptbxl": 17417, "mimic": 39392, "georgia": 10187,
                                  "cpsc_2018": 6581, "cpsc_2018_extra": 3021},
                "new_exclusions_by_source": clean_rerun.EXCLUDED_COUNTS,
                "label_counts": {key: {"retained": value} for key, value in clean_rerun.LABEL_COUNTS.items()},
                "heldout_reference_counts": clean_rerun.HELDOUT_COUNTS,
                "table_sha256": hashes, "input_sha256": {}}
    (dataset / "metadata.json").write_text(json.dumps(metadata))
    digest = sha256_file(dataset / "metadata.json")
    (release / "verification.json").write_text(json.dumps({"metadata_sha256": digest}))
    (release / "receipt.json").write_text(json.dumps(
        {"metadata_sha256": digest, "verification_sha256": sha256_file(release / "verification.json")}))
    with pytest.raises(ValueError, match="canonical metadata"):
        clean_rerun.verify_release(dataset, release)


def test_output_cannot_enter_input_tree(tmp_path: Path) -> None:
    """The preflight refuses publication inside a protected input directory."""
    with pytest.raises(ValueError, match="outside raw and verified input trees"):
        clean_rerun.prepare(tmp_path / "data/raw/unsafe", root=tmp_path)
