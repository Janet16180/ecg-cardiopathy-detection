"""Verify the declared supervised PTB cohort and prepare original-editor training banks."""

import os
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_background_editor_transfer_run as arrays
from ecg_experiment import ssl_synthetic_ludb_qualification as structural
from ecg_experiment import ssl_synthetic_ludb_run as shared
from ecg_experiment import ssl_synthetic_mixed_background as core
from ecg_experiment import ssl_synthetic_mixed_background_cohort as cohort
from ecg_experiment import ssl_synthetic_partition as partition
from ecg_experiment import ssl_synthetic_randomized_localization as original
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored

PERMITTED_RAW = Path("/home/janetrivera/ecg-cardiopathy-detection/data/raw/ptb-xl/1.0.3")


def checked_raw(root: Path, name: str) -> Path:
    """Restrict aliases to the explicitly permitted other-checkout raw source without recording targets.

    Parameters
    ----------
    root : Path
        Baseline-owned per-source alias directory beside tracked DVC pointers.
    name : str
        Metadata-validated source-relative filename.

    Returns
    -------
    Path
        Original lexical alias path, after read-only-use target containment validation.
    """
    path = shared.confined(root, name)
    target = Path(os.path.realpath(path))
    if not target.is_relative_to(PERMITTED_RAW) or not path.is_file():
        raise ValueError("PTB alias target escapes the permitted read-only raw source")
    return path


def selected_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct exactly the local40-patient declaration using only the four admitted CSV fields.

    Parameters
    ----------
    config : dict
        Frozen metadata/checksum identities, partition policy and selected-cohort rules.

    Returns
    -------
    list
        All40 local record declarations in their prospective geometry order40 through79.
    """
    raw = from_stored(config["raw_alias"])
    metadata = checked_raw(raw, "ptbxl_database.csv")
    table = checked_raw(raw, "SHA256SUMS.txt")
    if (
        sha256_file(metadata) != config["metadata_sha256"]
        or sha256_file(table) != config["official_table_sha256"]
    ):
        raise ValueError("PTB source metadata/checksum declaration changed")
    checksums = cohort.checksum_table(table)
    if checksums["ptbxl_database.csv"] != config["metadata_sha256"]:
        raise ValueError("Official metadata declaration differs")
    policy = shared.read_json(from_stored(config["partition_policy"]))
    partition.validate_policy(policy)
    assigned = partition.assign_records(partition.read_metadata(metadata.read_bytes()), policy)
    selected, _ = cohort.select_records(assigned, config["cohort"]["order_namespace"], 40)
    expected = [
        {
            **record,
            "geometry_index": slot,
            "header_sha256": checksums[record["filename_hr"] + ".hea"],
            "signal_sha256": checksums[record["filename_hr"] + ".dat"],
        }
        for slot, record in enumerate(selected, start=40)
    ]
    manifest = shared.read_json(from_stored(config["cohort_manifest"]))
    if (
        manifest["records"] != expected
        or manifest["cohort_policy"] != config["cohort"]
        or manifest["metadata_sha256"] != config["metadata_sha256"]
        or manifest["v4_membership_verified"] is not False
    ):
        raise ValueError("Admitted40-patient cohort differs from original-role/fold/hash reconstruction")
    if len({r["patient_id"] for r in expected}) != 40 or [r["geometry_index"] for r in expected] != list(
        range(40, 80)
    ):
        raise ValueError("Every fixed training patient/geometry must be unique and present")
    return expected


def verify_raw(config: dict[str, Any], records: list[dict[str, Any]], guard: Any) -> None:
    """Hash only the selected40 permitted header/payload pairs without changing raw ownership.

    Parameters
    ----------
    config : dict
        Frozen permitted source alias.
    records : list
        Fully reconstructed original-role/fold-qualified local declarations.
    guard : Any
        Live whole-stage CPU/wall/RSS guard.

    Returns
    -------
    None
        Raises on missing, changed or redirected files; no replacement is selected.
    """
    raw = from_stored(config["raw_alias"])
    for record in records:
        for suffix, key in [(".hea", "header_sha256"), (".dat", "signal_sha256")]:
            guard()
            if sha256_file(checked_raw(raw, record["filename_hr"] + suffix)) != record[key]:
                raise ValueError("Selected PTB source byte identity differs; no patient replacement")


def decode_real(record: dict[str, Any], config: dict[str, Any], destination: Path) -> tuple:
    """Retain each fixed selected-record attempt and stop on its first technical qualification failure.

    Parameters
    ----------
    record : dict
        One preselected original-role/fold-qualified PTB training declaration.
    config : dict
        Frozen permitted raw alias and original-editor policy.
    destination : Path
        Exclusive local study output root.

    Returns
    -------
    tuple
        Four canonical observations, strict/threshold masks and local source/editor metadata.
    """
    slot = record["geometry_index"]
    receipts = destination / "training_record_receipts"
    receipts.mkdir(exist_ok=True)
    shared.write_new(receipts / f"slot_{slot}_started.json", {"record": record, "status": "started"})
    try:
        path = checked_raw(from_stored(config["raw_alias"]), record["filename_hr"] + ".hea").with_suffix("")
        physical, header = structural.read_signal(
            path, destination / "sanitized_training_headers" / f"slot_{slot}"
        )
        if header["canonical_units"] != ["mV"] * 12:
            raise ValueError("Selected PTB record is not in declared physical mV units")
        if np.any(np.diff(np.quantile(physical, [0.05, 0.95], axis=1, method="linear"), axis=0) <= 0):
            raise ValueError("A physical training lead has zero robust range")
        values, exact, target, row = core.real_case(physical, slot)
        if np.any(np.diff(np.quantile(values[0], [0.05, 0.95], axis=1, method="linear"), axis=0) <= 0):
            raise ValueError("A derived canonical training lead has zero robust range")
        checks = row["checks"].values()
        if not all(
            c["strict_mask_exact"]
            and c["threshold_mask_exact"]
            and c["nonempty"]
            and c["threshold_subset_strict"]
            and c["outside_hull_cells"] == 0
            for c in checks
        ):
            raise ValueError("A fixed real training intervention failed final-mask/hull qualification")
        metadata = {
            **record,
            "signal_metadata": header,
            "original_editor": row,
            "clinical_labels_interpreted": 0,
            "generator_forwards": 0,
            "legacy_editor_seed_field_is_case_identity_only": True,
        }
        shared.write_new(receipts / f"slot_{slot}_passed.json", metadata)
        return values, exact, target, metadata
    except Exception as error:
        shared.write_new(
            receipts / f"slot_{slot}_failure.json",
            {
                "record": record,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "replacement_selected": False,
            },
        )
        raise


def prepare_banks(
    destination: Path, config: dict[str, Any], records: list[dict[str, Any]], guard: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Replace only slots40--79 with the fixed raw PTB cohort and retain exact rounded target truth.

    Parameters
    ----------
    destination : Path
        Exclusive new study directory.
    config : dict
        Original generated bank, physical source and fixed weight declarations.
    records : list
        All40 preselected local patient/record declarations.
    guard : Any
        Live stage guard called before every waveform decode or bank flush.

    Returns
    -------
    tuple
        Flattened generated/mixed training banks and aggregate data qualification evidence.
    """
    source = from_stored(config["generated_training_root"])
    generated = np.load(source / "observed.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(source / "support.npy", mmap_mode="r", allow_pickle=False)
    if (
        generated.shape != (128, 4, 12, 5000)
        or generated.dtype != np.float32
        or targets.shape != generated.shape
        or targets.dtype != np.bool_
    ):
        raise ValueError("Original generated training bank geometry differs")
    observed = arrays.new_array(destination / "mixed_observed.npy", (80, 4, 12, 5000), np.float32)
    support = arrays.new_array(destination / "mixed_support.npy", (80, 4, 12, 5000), bool)
    strict = arrays.new_array(destination / "mixed_strict_support.npy", (80, 2, 12, 5000), bool)
    observed[:40], support[:40] = generated[:40], targets[:40]
    strict[:40] = generated[:40, 2:].astype(np.float64) - generated[:40, 1:2].astype(np.float64) != 0
    metadata = []
    for record in records:
        guard()
        slot = record["geometry_index"]
        values, exact, target, row = decode_real(record, config, destination)
        observed[slot], strict[slot], support[slot] = values, exact, target
        metadata.append(row)
    observed.flush()
    support.flush()
    strict.flush()
    if not np.array_equal(observed[:40], generated[:40]) or not np.array_equal(support[:40], targets[:40]):
        raise ValueError("Retained generated0--39 inputs/targets changed")
    banks = {}
    qualification = {}
    for name, values, truth in [("generated", generated[:80], targets[:80]), ("mixed", observed, support)]:
        guard()
        difference = values[:, 2:].astype(np.float64) - values[:, 1:2].astype(np.float64)
        if np.any(truth[:, :2]) or not np.array_equal(np.abs(difference) > 0.01, truth[:, 2:]):
            raise ValueError("TRAIN control/final-input target identity differs")
        if not np.isfinite(values).all():
            raise ValueError("A TRAIN input is nonfinite")
        weight = original.positive_weight(truth)
        if weight != config["positive_weight"] or weight != 20.0:
            raise ValueError("Both independently derived TRAIN-only capped weights must equal20")
        qualification[name] = {
            "backgrounds": 80,
            "positive_weight": weight,
            "positive_cells": int(truth.sum()),
            "negative_cells": int(truth.size - truth.sum()),
            "zero_control_targets": True,
        }
        banks[name] = {
            "observed": values.reshape(320, 12, 5000),
            "support": truth.reshape(320, 12, 5000),
            "weight": weight,
        }
    shared.write_new(destination / "training_input_metadata.json", metadata)
    shared.write_new(destination / "training_bank_qualification.json", qualification)
    del strict
    return banks, qualification
