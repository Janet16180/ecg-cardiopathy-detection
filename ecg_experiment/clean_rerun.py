"""CPU-only provenance and held-out audit for a clean original-cohort rerun."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

from .downloads import parse_checksums
from .evaluation import partition_validation
from .files import read_csv, sha256_file, write_csv_atomic, write_json_atomic
from .public_sources import signal_sha256
from .staging import published_directory
from .waveforms import LEADS, read_record

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/training_union_500hz_v1"
RELEASE = ROOT / "outputs/data_quality/training_union_v1"
PTB = ROOT / "data/processed/ptbxl/seed42_fraction1"
PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
MIMIC = ROOT / "data/processed/mimic_ssl_40k_cpc/ssl_manifest.csv"
BUDGETS = ("1", "0.1")
SPLITS = ("development", "calibration", "test")
COUNTS = {"ptbxl": 17417, "mimic": 39392}
HELDOUT_COUNTS = {"development": 1306, "calibration": 564, "test": 1896}
LABEL_COUNTS = {"1": 15359, "0.1": 1518}
EXCLUDED_COUNTS = {"ptbxl": 1, "mimic": 65}
TABLES = ("train_manifest.csv", "labels_fraction1.csv", "labels_fraction0.1.csv",
          "heldout_references.csv", "exclusions.csv")


def require(condition: bool, message: str) -> None:
    """Raise a data-integrity error when a required condition is false."""
    if not condition:
        raise ValueError(message)


def _unique(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    """Index rows by a unique, nonempty field."""
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows) and "" not in result, f"Duplicate or empty {key}")
    return result


def verify_release(dataset: Path, release: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify immutable union metadata, release receipts and every table hash."""
    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    verification_path = release / "verification.json"
    receipt = json.loads((release / "receipt.json").read_text())
    verification = json.loads(verification_path.read_text())
    digest = sha256_file(metadata_path)
    require(digest == verification["metadata_sha256"] == receipt["metadata_sha256"],
            "Union metadata is not pinned by release receipts")
    require(sha256_file(verification_path) == receipt["verification_sha256"],
            "Union verification receipt changed")
    require(metadata["complete"] is True and metadata["schema_version"] == 1 and
            metadata["record_count"] == 76598 and metadata["shape_per_record"] == [12, 5000] and
            metadata["dtype"] == "float32" and metadata["sampling_rate_hz"] == 500 and
            metadata["units"] == "mV" and metadata["lead_order"] == list(LEADS),
            "Union canonical metadata contract changed")
    require(metadata["source_counts"] == {"ptbxl": 17417, "mimic": 39392,
            "georgia": 10187, "cpsc_2018": 6581, "cpsc_2018_extra": 3021},
            "Union source counts changed")
    require(metadata["new_exclusions_by_source"] == EXCLUDED_COUNTS and
            {key: value["retained"] for key, value in metadata["label_counts"].items()} == LABEL_COUNTS and
            metadata["heldout_reference_counts"] == HELDOUT_COUNTS,
            "Union selection counts changed")
    require(set(TABLES) <= set(metadata["table_sha256"]), "Union table pins incomplete")
    pins = {str(metadata_path): digest, str(verification_path): sha256_file(verification_path),
            str(release / "receipt.json"): sha256_file(release / "receipt.json")}
    for name, expected in metadata["table_sha256"].items():
        path = dataset / name
        require(path.resolve().is_relative_to(dataset.resolve()) and sha256_file(path) == expected,
                f"Union table changed: {name}")
        pins[str(path)] = expected
    for name, expected in metadata["input_sha256"].items():
        path = Path(name)
        require(path.is_file() and sha256_file(path) == expected, f"Union source input changed: {name}")
        pins[str(path)] = expected
    return metadata, pins



def verify_label_nesting(full: list[dict[str, str]], limited: list[dict[str, str]]) -> None:
    """Require each limited record, patient and target tuple in the full selection."""
    full_tuples = {(r["record_id"], r["patient_id"], r["target"]) for r in full}
    limited_tuples = {(r["record_id"], r["patient_id"], r["target"]) for r in limited}
    require(len(full_tuples) == len(full) and len(limited_tuples) == len(limited) and
            limited_tuples <= full_tuples, "Limited labels differ from full patient/target tuples")


def verify_patient_partitions(selected: list[dict[str, str]],
                              refs: list[dict[str, str]]) -> None:
    """Require known PTB and MIMIC patients to stay disjoint across partitions."""
    patients = {"train": {r["patient_id"] for r in selected if r["source"] == "ptbxl"}}
    patients.update({split: {r["patient_id"] for r in refs if r["split"] == split} for split in SPLITS})
    for index, first in enumerate(patients):
        for second in list(patients)[index + 1:]:
            require(not patients[first] & patients[second], f"PTB patient overlap: {first}/{second}")
    require(not {r["patient_id"] for r in selected if r["source"] == "mimic"} &
            set.union(*patients.values()), "Namespaced MIMIC/PTB patient overlap")


def verify_selection(dataset: Path, metadata: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Match every selected original-pool row, exclusion, label and held-out reference."""
    rows = read_csv(dataset / "train_manifest.csv")
    selected = [row for row in rows if row["source"] in COUNTS]
    require(len(rows) == metadata["record_count"] and Counter(r["source"] for r in selected) == COUNTS,
            "Clean cohort counts changed")
    by_id = _unique(rows, "record_id")
    require(all(r["split"] == "train" for r in rows), "Nontraining row in union")
    require(len({(r["shard"], r["shard_index"]) for r in rows}) == len(rows),
            "Duplicate union shard pointer")
    require(len({r["signal_sha256"] for r in rows}) == len(rows), "Duplicate union signal hash")
    exclusions = read_csv(dataset / "exclusions.csv")
    excluded = _unique(exclusions, "record_id")
    require(Counter(r["source"] for r in exclusions) == EXCLUDED_COUNTS and
            all(r["reason"] == "full_constant_lead" for r in exclusions), "Exclusions changed")
    frozen_ptb = read_csv(root / "data/processed/ptbxl/seed42_fraction1/all_train_ssl.csv")
    frozen_mimic = read_csv(root / "data/processed/mimic_ssl_40k_cpc/ssl_manifest.csv")
    cpc_dir = root / "data/processed/cpc_pool_40k"
    cpc_complete = json.loads((cpc_dir / "complete.json").read_text())
    require(sha256_file(cpc_dir / "rows.csv") == cpc_complete["rows_sha256"],
            "Frozen CPC rows changed")
    cpc_train = [r for r in read_csv(cpc_dir / "rows.csv") if r["split"] == "train"]
    require(len(cpc_train) == 56875 and Counter(r["source"] for r in cpc_train) ==
            {"ptbxl": 17418, "mimic": 39457}, "Frozen CPC train composition changed")
    cpc_identity = {(r["source"], r["ecg_id"], r["patient_id"]) for r in cpc_train}
    source_identity = {("ptbxl", r["ecg_id"], r["patient_id"]) for r in frozen_ptb}
    source_identity.update(("mimic", r["ecg_id"], r["patient_id"]) for r in frozen_mimic)
    require(cpc_identity == source_identity, "Frozen CPC rows differ from original source manifests")
    expected = {"ptbxl:" + r["ecg_id"]: ("ptbxl:" + r["patient_id"], r["filename_hr"])
                for r in frozen_ptb}
    expected.update({r["ecg_id"]: (r["patient_id"], r["filename_hr"]) for r in frozen_mimic})
    require(len(expected) == 56875 and set(expected) == {r["record_id"] for r in selected} | set(excluded),
            "Clean cohort is not exactly the frozen CPC PTB/MIMIC pool minus exclusions")
    require(not set(excluded) & {r["record_id"] for r in selected},
            "Excluded original-pool row retained in clean cohort")
    for row in selected:
        patient, source_path = expected[row["record_id"]]
        require(row["patient_id"] == patient and row["source_record_id"] ==
                (row["record_id"].removeprefix("ptbxl:") if row["source"] == "ptbxl" else row["record_id"])
                and row["origin_path"].endswith(source_path) and row["view"] == "original_10s" and
                row["window_start"] == "0" and row["source_samples"] == "5000" and
                row["patient_identity_known"] == "true", "Original-pool row identity or view changed")
    labels = {}
    for budget in BUDGETS:
        name = f"labels_fraction{budget}.csv"
        actual = read_csv(dataset / name)
        frozen = read_csv(root / f"data/processed/ptbxl/seed42_fraction{budget}/labeled_train.csv")
        expected_labels = [{"record_id": "ptbxl:" + row["ecg_id"],
                            "patient_id": "ptbxl:" + row["patient_id"], "target": row["target"]}
                           for row in frozen if "ptbxl:" + row["ecg_id"] in by_id]
        require(actual == expected_labels and len(actual) == LABEL_COUNTS[budget],
                f"Frozen {budget} label selection differs")
        require(len(_unique(actual, "record_id")) == len(actual) and
                all(row["target"] in {"0", "1"} and
                    by_id[row["record_id"]]["patient_id"] == row["patient_id"] and
                    by_id[row["record_id"]]["source"] == "ptbxl" for row in actual),
                f"Invalid {budget} target or patient")
        labels[budget] = actual
    verify_label_nesting(labels["1"], labels["0.1"])
    refs = read_csv(dataset / "heldout_references.csv")
    validation_path = root / "data/processed/ptbxl/seed42_fraction1/validation.csv"
    development, calibration = partition_validation(read_csv(validation_path))
    originals = (("development", development), ("calibration", calibration),
                 ("test", read_csv(root / "data/processed/ptbxl/seed42_fraction1/test.csv")))
    raw_root = root / "data/raw/ptb-xl/1.0.3"
    expected_refs = [{"record_id": "ptbxl:" + row["ecg_id"],
                      "patient_id": "ptbxl:" + row["patient_id"], "split": split,
                      "target": row["target"],
                      "raw_path": str((raw_root / row["filename_hr"]).relative_to(root))}
                     for split, group in originals for row in group]
    require(refs == expected_refs and Counter(r["split"] for r in refs) == HELDOUT_COUNTS,
            "Held-out references differ from frozen PTB partitions")
    require(not set(_unique(refs, "record_id")) & set(by_id), "Held-out record in union")
    verify_patient_partitions(selected, refs)
    return {"selected": selected, "references": refs, "exclusions": exclusions,
            "labels": labels, "train_hashes": {r["signal_sha256"] for r in rows}}


def audit_heldout(references: list[dict[str, str]], train_hashes: set[str],
                  root: Path = ROOT) -> list[dict[str, str]]:
    """Verify official raw bytes and describe every held-out signal without dropping one."""
    raw = root / "data/raw/ptb-xl/1.0.3"
    checksums = parse_checksums((raw / "SHA256SUMS.txt").read_text())
    seen: dict[str, str] = {}
    audit = []
    for row in references:
        stem = (root / row["raw_path"]).resolve()
        require(stem.is_relative_to(raw.resolve()), "Held-out path escapes official PTB release")
        relative = str(stem.relative_to(raw.resolve()))
        files = [stem.with_suffix(suffix) for suffix in (".hea", ".dat")]
        for suffix, path in zip((".hea", ".dat"), files, strict=True):
            require(checksums[relative + suffix] == sha256_file(path), "Official PTB checksum mismatch")
        header = wfdb.rdheader(str(stem))
        require(header.units == ["mV"] * 12, "Held-out units are not mV")
        signal = read_record(raw, relative)
        for suffix, path in zip((".hea", ".dat"), files, strict=True):
            require(checksums[relative + suffix] == sha256_file(path), "PTB raw file changed during read")
        require(signal.dtype == np.float32 and signal.shape == (12, 5000) and
                np.isfinite(signal).all(), "Invalid held-out waveform")
        digest = signal_sha256(signal)
        require(digest not in train_hashes and digest not in seen,
                "Held-out exact signal duplicates training or another held-out record")
        seen[digest] = row["record_id"]
        flags = [f"constant_{LEADS[i]}" for i in np.flatnonzero(np.ptp(signal, axis=1) == 0)]
        flags += [f"near_flat_{LEADS[i]}" for i in np.flatnonzero(signal.std(axis=1) < 0.01)]
        if np.max(np.abs(signal)) > 10:
            flags.append("amplitude_over_10mV")
        audit.append({"record_id": row["record_id"], "split": row["split"],
                      "signal_sha256": digest, "review_flags": ";".join(flags)})
    return audit


def prepare(output: Path, dataset: Path = DATASET, release: Path = RELEASE,
            root: Path = ROOT) -> dict[str, Any]:
    """Publish a new immutable pointer-only clean-rerun manifest and audit receipt."""
    output = output.resolve()
    require(not output.exists(), "Refusing to overwrite clean-rerun output")
    protected = (root / "data", root / "outputs/data_quality/training_union_v1", dataset, release)
    require(not any(output == path.resolve() or output.is_relative_to(path.resolve())
                    for path in protected), "Output must be outside raw and verified input trees")
    source_files = ("ecg_experiment/clean_rerun.py", "scripts/validation/prepare_clean_rerun.py",
                    "ecg_experiment/evaluation.py", "ecg_experiment/waveforms.py",
                    "ecg_experiment/files.py", "ecg_experiment/downloads.py",
                    "ecg_experiment/public_sources.py", "ecg_experiment/staging.py")
    source_pins = {str(root / name): sha256_file(root / name) for name in source_files}
    dependencies = {name: version(name) for name in ("numpy", "scipy", "wfdb", "scikit-learn")}
    dependencies["python"] = sys.version.split()[0]
    started_at = datetime.now(UTC).isoformat()
    metadata, pins = verify_release(dataset, release)
    selection = verify_selection(dataset, metadata, root)
    cpc_rows = root / "data/processed/cpc_pool_40k/rows.csv"
    pins[str(cpc_rows)] = sha256_file(cpc_rows)
    audit = audit_heldout(selection["references"], selection["train_hashes"], root)
    require(len(audit) == sum(HELDOUT_COUNTS.values()), "Held-out audit incomplete")
    # Rehash every small pinned input after the long raw audit to detect concurrent changes.
    for name, expected in pins.items():
        require(sha256_file(name) == expected, f"Input changed during audit: {name}")
    for name, expected in source_pins.items():
        require(sha256_file(name) == expected, f"Source changed during audit: {name}")
    require({name: version(name) for name in ("numpy", "scipy", "wfdb", "scikit-learn")} ==
            {key: value for key, value in dependencies.items() if key != "python"},
            "Dependency version changed during audit")
    receipt = {"schema_version": 1, "status": "passed_cpu_preflight_no_training",
               "cohort": "original_frozen_ptb_mimic_minus_union_constant_lead_exclusions",
               "train_count": len(selection["selected"]),
               "train_source_counts": dict(Counter(r["source"] for r in selection["selected"])),
               "exclusion_counts": dict(Counter(r["source"] for r in selection["exclusions"])),
               "label_counts": {key: len(value) for key, value in selection["labels"].items()},
               "heldout_counts": dict(Counter(r["split"] for r in audit)),
               "heldout_review_flag_counts": dict(Counter(flag for row in audit
                   for flag in row["review_flags"].split(";") if flag)),
               "scope": "All held-out PTB raw pairs independently reread and official-checksummed; "
                        "all union training signal hashes read from pinned verified manifest; "
                        "union waveform shards were not reread; exact signal duplicates only",
               "waveform_processing": "none; source union pointers retained, held-out references unchanged",
               "union_metadata_sha256": pins[str(dataset / "metadata.json")],
               "input_sha256": pins,
               "source_sha256": {str(Path(name).relative_to(root)): digest
                                 for name, digest in source_pins.items()},
               "dependency_versions": dependencies, "command": sys.argv,
               "started_at_utc": started_at, "completed_at_utc": datetime.now(UTC).isoformat()}
    fields = list(read_csv(dataset / "train_manifest.csv")[0])
    with published_directory(output) as stage:
        write_csv_atomic(stage / "train_manifest.csv", selection["selected"], fields)
        for budget in BUDGETS:
            write_csv_atomic(stage / f"labels_fraction{budget}.csv", selection["labels"][budget],
                             ("record_id", "patient_id", "target"))
        (stage / "heldout_references.csv").write_bytes((dataset / "heldout_references.csv").read_bytes())
        write_csv_atomic(stage / "exclusions.csv", selection["exclusions"],
                         ("record_id", "source", "reason", "detail", "signal_sha256"))
        write_csv_atomic(stage / "heldout_audit.csv", audit,
                         ("record_id", "split", "signal_sha256", "review_flags"))
        receipt["output_sha256"] = {path.name: sha256_file(path) for path in stage.iterdir()}
        write_json_atomic(stage / "receipt.json", receipt, sort_keys=True)
    return receipt
