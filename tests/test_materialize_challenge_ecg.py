"""Integration checks for immutable Challenge waveform materialization."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.data.materialize_challenge_ecg as materialization
from ecg_experiment.files import read_csv, sha256_file
from ecg_experiment.public_sources import signal_sha256
from scripts.data.materialize_challenge_ecg import materialize, verify_materialized


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def use_root(monkeypatch: pytest.MonkeyPatch, root: Path,
             view: tuple[np.ndarray, int, int, str] | None = None) -> None:
    """Point the module at a fixture repository and fake the canonical view reader."""
    monkeypatch.setattr(materialization, "ROOT", root)
    if view is not None:
        monkeypatch.setattr(materialization, "load_view", lambda *_: view)


def fixture(tmp_path: Path, *, crop: bool = False, conflict: bool = False):
    root = tmp_path
    raw = root / "data/raw/challenge-2020/1.0.2"
    raw.mkdir(parents=True)
    (root / "data/acquisition").mkdir(parents=True)
    (root / "data/acquisition/georgia.json").write_text('{"state":"complete"}')
    samples = 6000 if crop else 5000
    time = np.arange(5000, dtype=np.float32) / 1000
    signal = np.stack([time + lead / 10 for lead in range(12)]).astype(np.float32)
    stems = ["training/georgia/g1/E00001"]
    if conflict:
        stems.append("training/georgia/g1/E00002")
    checksum_lines = []
    for index, stem in enumerate(stems):
        path = raw / stem
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.with_suffix(".hea")).write_text("# Dx: " + ("1" if index == 0 else "2") + "\n")
        (path.with_suffix(".mat")).write_bytes(b"raw signal fixture")
        for suffix in (".hea", ".mat"):
            file = path.with_suffix(suffix)
            checksum_lines.append(f"{sha256_file(file)}  {stem}{suffix}")
    checksums = raw / "SHA256SUMS.txt"
    checksums.write_text("\n".join(checksum_lines) + "\n")
    prepared = root / "data/processed/audit"
    prepared.mkdir(parents=True)
    row = {"ecg_id": "georgia:E00001", "patient_id": "georgia:E00001",
           "patient_identity_known": "false", "source": "georgia", "raw_dir": str(raw),
           "filename_hr": stems[0], "source_samples": str(samples),
           "window_start": "500" if crop else "0", "label_codes": "1",
           "signal_sha256": signal_sha256(signal), "qc_flags": ""}
    write_csv(prepared / "manifest.csv", [row])
    excluded = [{"ecg_id": "georgia:E00002", "source": "georgia",
                 "reason": "exact_pool_duplicate", "detail": "exact_pool_duplicate"}] if conflict else []
    with (prepared / "exclusions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ecg_id", "source", "reason", "detail"])
        writer.writeheader()
        writer.writerows(excluded)
    meta = {"policy": "ssl_center_crop" if crop else "strict_10s",
            "sources": ["georgia"], "accepted_records": 1,
            "excluded_records": len(excluded), "candidate_records": 1 + len(excluded),
            "manifest_sha256": sha256_file(prepared / "manifest.csv"),
            "exclusions_sha256": sha256_file(prepared / "exclusions.csv"),
            "preparation_source_sha256": "historical-revision",
            "provenance": {"georgia": {"checksum_file": str(checksums.relative_to(root)),
                                        "checksum_sha256": sha256_file(checksums),
                                        "source_url": "https://physionet.org/files/challenge-2020/1.0.2",
                                        "candidate_records": 1 + len(excluded)}}}
    (prepared / "metadata.json").write_text(json.dumps(meta))
    comparisons = root / "comparisons.csv"
    if conflict:
        write_csv(comparisons, [{"excluded_ecg_id": "georgia:E00002",
                                 "retained_ecg_id": "georgia:E00001",
                                 "excluded_label_codes": "2", "retained_label_codes": "1",
                                 "same_label_set": "false"}])
    return root, raw, prepared, comparisons, signal


def test_strict_conflict_retained_copy_is_not_label_eligible(tmp_path: Path, monkeypatch) -> None:
    root, raw, prepared, comparisons, signal = fixture(tmp_path, conflict=True)
    output = root / "data/processed/views/strict"
    raw_before = {file: sha256_file(file) for file in raw.rglob("*") if file.is_file()}
    use_root(monkeypatch, root, (signal, 0, 5000, ""))
    result = materialize(prepared, output, comparisons=comparisons)
    row = read_csv(output / "manifest.csv")[0]
    assert result["counts"]["label_conflict_retained"] == 1
    assert row["label_scope"] == "conflicting_duplicate_annotations"
    assert row["record_annotation_available"] == "false"
    assert row["endpoint_supervised_eligible"] == "false"
    assert row["patient_independent_eval_eligible"] == "false"
    array = np.load(output / row["shard"], allow_pickle=False)
    assert array.shape == (1, 12, 5000)
    assert array.dtype == np.float32
    assert signal_sha256(array[0]) == row["signal_sha256"]
    assert verify_materialized(output)["verified_records"] == 1
    assert {file: sha256_file(file) for file in raw.rglob("*") if file.is_file()} == raw_before


def test_center_crop_is_machine_readable_ssl_only(tmp_path: Path, monkeypatch) -> None:
    root, _, prepared, _, signal = fixture(tmp_path, crop=True)
    overlap = root / "other.csv"
    write_csv(overlap, [{"ecg_id": "georgia:E00001", "signal_sha256": signal_sha256(signal)}])
    output = root / "data/processed/views/crop"
    use_root(monkeypatch, root, (signal, 500, 6000, ""))
    materialize(prepared, output, overlap_manifest=overlap)
    row = read_csv(output / "manifest.csv")[0]
    assert row["label_scope"] == "ssl_only_crop"
    assert row["record_annotation_available"] == "false"
    assert row["endpoint_supervised_eligible"] == "false"
    assert row["overlap_by_record"] == "true"
    assert row["overlap_by_signal"] == "true"


def test_changed_raw_file_fails_without_publishing_output(tmp_path: Path, monkeypatch) -> None:
    root, raw, prepared, _, signal = fixture(tmp_path)
    (raw / "training/georgia/g1/E00001.mat").write_bytes(b"changed")
    output = root / "data/processed/views/strict"
    use_root(monkeypatch, root, (signal, 0, 5000, ""))
    with pytest.raises(ValueError, match="Official raw file checksum mismatch"):
        materialize(prepared, output)
    assert not output.exists()


def test_strict_original_annotation_is_unmapped_and_not_endpoint_ready(tmp_path: Path, monkeypatch) -> None:
    root, _, prepared, _, signal = fixture(tmp_path)
    output = root / "data/processed/views/strict"
    use_root(monkeypatch, root, (signal, 0, 5000, ""))
    materialize(prepared, output)
    row = read_csv(output / "manifest.csv")[0]
    assert row["record_annotation_available"] == "true"
    assert row["label_scope"] == "original_record_annotation_unmapped"
    assert row["endpoint_supervised_eligible"] == "false"


def test_changed_canonical_view_fails_without_publishing_output(tmp_path: Path, monkeypatch) -> None:
    root, _, prepared, _, signal = fixture(tmp_path)
    output = root / "data/processed/views/strict"
    modified = signal.copy()
    modified[0, 0] += 0.5
    use_root(monkeypatch, root, (modified, 0, 5000, ""))
    with pytest.raises(ValueError, match="Materialized view differs"):
        materialize(prepared, output)
    assert not output.exists()


def test_output_inside_raw_is_rejected(tmp_path: Path, monkeypatch) -> None:
    root, raw, prepared, _, _ = fixture(tmp_path)
    use_root(monkeypatch, root)
    with pytest.raises(ValueError, match="outside data/raw"):
        materialize(prepared, raw / "bad")


def test_verifier_rejects_changed_shard(tmp_path: Path, monkeypatch) -> None:
    root, _, prepared, _, signal = fixture(tmp_path)
    output = root / "data/processed/views/strict"
    use_root(monkeypatch, root, (signal, 0, 5000, ""))
    materialize(prepared, output)
    shard = output / "shard_00000.npy"
    with shard.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="Materialized shard hash mismatch"):
        verify_materialized(output)


@pytest.mark.parametrize(("field", "value", "error"), [
    ("endpoint_supervised_eligible", "true", "eligibility"),
    ("patient_identity_known", "true", "eligibility"),
    ("window_start", "100", "duration contract"),
    ("units", "uV", "row signal contract"),
    ("min_lead_std_mv", "999", "QC statistics"),
])
def test_verifier_rejects_semantic_corruption_even_when_rehashed(
        tmp_path: Path, monkeypatch, field: str, value: str, error: str) -> None:
    root, _, prepared, _, signal = fixture(tmp_path)
    output = root / "data/processed/views/strict"
    use_root(monkeypatch, root, (signal, 0, 5000, ""))
    materialize(prepared, output)
    records = read_csv(output / "manifest.csv")
    records[0][field] = value
    write_csv(output / "manifest.csv", records)
    metadata_file = output / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["manifest_sha256"] = sha256_file(output / "manifest.csv")
    metadata_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        verify_materialized(output)


def test_verifier_rejects_incomplete_publication(tmp_path: Path, monkeypatch) -> None:
    root, _, prepared, _, signal = fixture(tmp_path)
    output = root / "data/processed/views/strict"
    use_root(monkeypatch, root, (signal, 0, 5000, ""))
    materialize(prepared, output)
    path = output / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["complete"] = False
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unpublished"):
        verify_materialized(output)


def test_strict_duplicates_require_comparison_table(tmp_path: Path, monkeypatch) -> None:
    root, _, prepared, _, signal = fixture(tmp_path, conflict=True)
    output = root / "data/processed/views/strict"
    use_root(monkeypatch, root, (signal, 0, 5000, ""))
    with pytest.raises(ValueError, match="comparison table required"):
        materialize(prepared, output)
    assert not output.exists()
    assert not list(output.parent.glob(".strict.staging-*"))
