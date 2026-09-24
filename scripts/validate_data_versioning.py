"""Check a completed source against its receipt and official hashes, plus PTB splits.

This is a read-only validation command. It does not download, preprocess, or train.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_hashes(path: Path, prefix: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition(" ")
        relative = relative.lstrip(" *")
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f"Invalid official checksum line: {line[:100]}")
        if relative.startswith(prefix):
            local = relative.removeprefix(prefix)
            if not local or local in selected or Path(local).is_absolute() or ".." in Path(local).parts:
                raise ValueError(f"Invalid or duplicate source path: {relative}")
            selected[local] = digest.lower()
    if not selected:
        raise ValueError(f"No official checksum entries under {prefix}")
    return selected


def validate_source(root: Path, spec: dict) -> dict:
    source = root / spec["path"]
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"{spec['id']}: source directory missing or symlinked")
    if spec.get("kind") == "ptbxl_records":
        manifest = root / spec["official_manifest"]
        manifest_digest = sha256(manifest)
        if manifest_digest != spec["official_manifest_sha256"]:
            raise ValueError(f"{spec['id']}: PTB official manifest hash differs from frozen reference")
        expected = official_hashes(manifest, spec["official_prefix"])
        actual = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
        if len(expected) != spec["expected_files"] or actual != expected.keys():
            raise ValueError(f"{spec['id']}: PTB file inventory differs from official manifest")
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError(f"{spec['id']}: source contains symlinks")
        headers = {path.removesuffix(".hea") for path in expected if path.endswith(".hea")}
        waveforms = {path.removesuffix(".dat") for path in expected if path.endswith(".dat")}
        if headers != waveforms or len(headers) != spec["expected_records"]:
            raise ValueError(f"{spec['id']}: PTB waveform/header pair count differs from config")
        for relative, digest in expected.items():
            if sha256(source / relative) != digest:
                raise ValueError(f"{spec['id']}: checksum mismatch: {relative}")
        return {"id": spec["id"], "source_path": spec["path"], "state": "verified",
                "records": len(headers), "files": len(expected), "official_manifest_sha256": manifest_digest}
    if spec.get("kind") not in (None, "challenge_pairs"):
        raise ValueError(f"Unsupported source kind: {spec['kind']}")
    receipt = json.loads((root / spec["receipt"]).read_text(encoding="utf-8"))
    manifest = root / spec["official_manifest"]
    expected_count = spec["expected_files"]
    if receipt.get("state") != "complete":
        raise ValueError(f"{spec['id']}: acquisition is not complete")
    for key, expected in (("expected_files", expected_count), ("verified_files", expected_count),
                          ("expected_records", spec["expected_records"])):
        if receipt.get(key) != expected:
            raise ValueError(f"{spec['id']}: receipt {key} differs from config")
    if receipt.get("source_manifest_sha256") != sha256(manifest):
        raise ValueError(f"{spec['id']}: official manifest hash differs from receipt")
    if receipt.get("source_manifest") != spec["official_manifest"]:
        raise ValueError(f"{spec['id']}: official manifest path differs from receipt")
    expected = official_hashes(manifest, spec["official_prefix"])
    waveform_entries = {path for path in expected if path.endswith((".hea", ".mat"))}
    if len(waveform_entries) != expected_count:
        raise ValueError(f"{spec['id']}: official waveform file count differs from config")
    actual = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
    if actual != waveform_entries:
        raise ValueError(f"{spec['id']}: missing {len(waveform_entries - actual)}, unexpected {len(actual - waveform_entries)} files")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError(f"{spec['id']}: source contains symlinks")
    headers = {path.removesuffix(".hea") for path in waveform_entries if path.endswith(".hea")}
    waveforms = {path.removesuffix(".mat") for path in waveform_entries if path.endswith(".mat")}
    if headers != waveforms or len(headers) != spec["expected_records"]:
        raise ValueError(f"{spec['id']}: waveform/header pair count differs from receipt")
    for relative in sorted(waveform_entries):
        digest = expected[relative]
        if sha256(source / relative) != digest:
            raise ValueError(f"{spec['id']}: checksum mismatch: {relative}")
    return {"id": spec["id"], "source_path": spec["path"], "state": "verified",
            "records": len(headers), "waveform_files": len(waveform_entries), "files": len(actual),
            "official_manifest_sha256": receipt["source_manifest_sha256"]}


def load_split(path: Path) -> dict[str, tuple[str, str | None]]:
    rows: dict[str, tuple[str, str | None]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"ecg_id", "patient_id"} <= set(reader.fieldnames):
            raise ValueError(f"Missing ECG or patient ID columns: {path}")
        for row in reader:
            record, patient = row["ecg_id"].strip(), row["patient_id"].strip()
            if not record or not patient or record in rows:
                raise ValueError(f"Missing or duplicate identity in {path}")
            rows[record] = (patient, row.get("target"))
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def validate_splits(root: Path, paths: dict[str, str]) -> dict:
    splits = {name: load_split(root / relative) for name, relative in paths.items()
              if name != "heldout_references"}
    for name in ("full_labeled", "limited_labeled", "validation", "test"):
        if any(target not in {"0", "1"} for _, target in splits[name].values()):
            raise ValueError(f"{name}: target must be binary and present")
    train = splits["full_train"]
    holdouts = [splits["validation"], splits["test"]]
    for name in ("full_labeled", "limited_labeled"):
        for record, identity in splits[name].items():
            if record not in train or train[record][0] != identity[0]:
                raise ValueError(f"{name}: record missing from full training split or patient mismatch")
    for record, identity in splits["limited_labeled"].items():
        if splits["full_labeled"].get(record) != identity:
            raise ValueError("Limited-label selection differs from full-label selection")
    groups = {name: {patient for patient, _ in rows.values()}
              for name, rows in (("full_train", train), ("validation", holdouts[0]), ("test", holdouts[1]))}
    for left, right in (("full_train", "validation"), ("full_train", "test"), ("validation", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"Patient leakage between {left} and {right}")
    for left, right in (("full_train", "validation"), ("full_train", "test"), ("validation", "test")):
        if splits[left].keys() & splits[right].keys():
            raise ValueError(f"Record overlap between {left} and {right}")
    if "heldout_references" in paths:
        heldout: dict[str, dict[str, str]] = {name: {} for name in ("development", "calibration", "test")}
        with (root / paths["heldout_references"]).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"record_id", "patient_id", "split"} <= set(reader.fieldnames):
                raise ValueError("Held-out reference columns are incomplete")
            for row in reader:
                name = row["split"]
                if name not in heldout:
                    raise ValueError(f"Unknown held-out split: {name}")
                record = row["record_id"].removeprefix("ptbxl:")
                patient = row["patient_id"].removeprefix("ptbxl:")
                if not record or not patient or record in heldout[name]:
                    raise ValueError("Missing or duplicate held-out identity")
                heldout[name][record] = patient
        if set(heldout["development"]) | set(heldout["calibration"]) != set(splits["validation"]):
            raise ValueError("Development/calibration do not partition PTB validation")
        if set(heldout["test"]) != set(splits["test"]):
            raise ValueError("Held-out test differs from PTB test")
        for left, right in (("development", "calibration"), ("development", "test"), ("calibration", "test")):
            if heldout[left].keys() & heldout[right].keys():
                raise ValueError(f"Record overlap between {left} and {right}")
        if any(not rows for rows in heldout.values()):
            raise ValueError("An expected held-out partition is empty")
        for name, rows in heldout.items():
            reference = splits["test"] if name == "test" else splits["validation"]
            if any(reference[record][0] != patient for record, patient in rows.items()):
                raise ValueError(f"{name}: held-out patient mapping differs from PTB split")
        heldout_patients = {name: set(rows.values()) for name, rows in heldout.items()}
        for left, right in (("development", "calibration"), ("development", "test"), ("calibration", "test")):
            if heldout_patients[left] & heldout_patients[right]:
                raise ValueError(f"Patient leakage between {left} and {right}")
        if groups["full_train"] & set().union(*heldout_patients.values()):
            raise ValueError("Patient leakage between train and held-out references")
        split_details = {"record_counts": {name: len(rows) for name, rows in heldout.items()},
                         "patient_counts": {name: len(rows) for name, rows in heldout_patients.items()}}
    else:
        split_details = None
    return {"state": "verified", "record_counts": {key: len(value) for key, value in splits.items()},
            "patient_counts": {key: len(value) for key, value in groups.items()},
            "heldout_partition": split_details, "patient_leakage": 0}


def validate(root: Path, config: Path) -> dict:
    settings = json.loads(config.read_text(encoding="utf-8"))
    if settings.get("schema_version") != 1:
        raise ValueError("Unsupported dataset config schema")
    ptb_manifest = root / "data/raw/ptb-xl/1.0.3/SHA256SUMS.txt"
    ptb_hashes = official_hashes(ptb_manifest, "")
    for relative in settings["ptb_metadata"]:
        path = root / relative
        if path.name not in ptb_hashes or sha256(path) != ptb_hashes[path.name]:
            raise ValueError(f"PTB metadata checksum mismatch: {relative}")
    split_paths = settings["ptb_split_audit"]
    if settings["ptb_split_sha256"].keys() != split_paths.keys():
        raise ValueError("PTB split hash inventory differs from split paths")
    for name, relative in split_paths.items():
        if sha256(root / relative) != settings["ptb_split_sha256"][name]:
            raise ValueError(f"Frozen PTB split checksum mismatch: {name}")
    return {"schema_version": 1,
            "sources": [validate_source(root, spec) for spec in settings["datasets"]],
            "ptb_metadata_files": len(settings["ptb_metadata"]),
            "ptb_split_audit": validate_splits(root, split_paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/data-validation.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = validate(root, root / args.config)
    destination = root / args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated {sum(source['files'] for source in report['sources'])} source files and PTB patient splits")


if __name__ == "__main__":
    main()
