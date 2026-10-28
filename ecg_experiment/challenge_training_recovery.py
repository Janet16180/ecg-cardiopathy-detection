"""Stage Challenge training quality and label evidence behind the original split hash gate."""

from __future__ import annotations

import json
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from . import ROOT
from .downloads import parse_checksums
from .ecg_quality import assess
from .files import sha256_file, write_json_atomic
from .ningbo import LABEL_COLUMNS, clean_age, duplicate_status, label_key, male
from .paths import to_stored
from .public_sources import signal_sha256
from .recovery_views import read_verified_view

SPLIT_SHA256 = "5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8"
RECOVERY = Path("outputs/cohort25_recovery_20261028")
RECONSTRUCTION = RECOVERY / "challenge/full"
OUTPUT = RECOVERY / "challenge_training_quality"
RELEASES = {
    "ningbo": "challenge-2021/1.0.3", "chapman_shaoxing": "challenge-2021/1.0.3",
    "georgia": "challenge-2020/1.0.2", "cpsc_2018": "challenge-2020/1.0.2",
    "cpsc_2018_extra": "challenge-2020/1.0.2",
}
HEADER_COLUMNS = ("source", "path", "age", "sex", "fs", "samples")
SOURCE_FILES = (
    "ecg_experiment/challenge_training_recovery.py", "scripts/data/recover_challenge_training_quality.py",
    "ecg_experiment/public_sources.py", "ecg_experiment/ecg_quality.py", "ecg_experiment/ningbo.py",
    "ecg_experiment/downloads.py", "ecg_experiment/paths.py", "ecg_experiment/recovery_views.py",
)
SPLIT_REFERENCE_COLUMNS = ("source", "family", "record", "path", "duration_s", "signal_sha256",
                          "window_sha256", "duplicate_group", "duplicate_status", "split")
REFERENCE_ROOT = RECOVERY / "heldout_references"


def stored(path: Path) -> str:
    """Store a root-relative input name through the shared receipt path helper."""
    return to_stored(ROOT / path)


def original_split(input_root: Path) -> tuple[pd.DataFrame, dict, dict[str, str]]:
    """Require exact original split bytes and read its label-free reference columns only."""
    directory = input_root / RECONSTRUCTION
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    rows_path = directory / "rows.csv"
    if (not metadata.get("complete") or not metadata.get("original_rows_reproduced") or
            metadata.get("rows_sha256") != SPLIT_SHA256 or sha256_file(rows_path) != SPLIT_SHA256):
        raise ValueError("Original Challenge split SHA-256 gate has not passed")
    inputs = {stored(RECONSTRUCTION / name): sha256_file(directory / name)
              for name in ("metadata.json", "rows.csv")}
    split = pd.read_csv(rows_path, dtype=str, keep_default_na=False, usecols=list(SPLIT_REFERENCE_COLUMNS))
    if (split.groupby("duplicate_group")["split"].nunique() > 1).any():
        raise ValueError("Challenge duplicate group spans split partitions")
    return split, metadata, inputs


def source_checkpoint(
    input_root: Path, source: str, metadata: dict, columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read selected header evidence bound to the original split reconstruction receipt."""
    directory = input_root / RECONSTRUCTION
    table_name = f"{source}_verified.parquet"
    receipt_name = f"{source}_receipt.json"
    receipt = json.loads((directory / receipt_name).read_text())
    bound = metadata["sources"][source]
    digest = sha256_file(directory / table_name)
    if (digest != receipt["checkpoint_sha256"] or digest != bound["checkpoint_sha256"] or
            receipt["verified_records"] != bound["verified_records"]):
        raise ValueError(f"Source checkpoint is not bound to the original split: {source}")
    headers = pd.read_parquet(directory / table_name, columns=columns)
    if not headers.index.is_unique or len(headers) != receipt["verified_records"]:
        raise ValueError(f"Invalid source checkpoint record identity: {source}")
    inputs = {stored(RECONSTRUCTION / table_name): digest,
              stored(RECONSTRUCTION / receipt_name): sha256_file(directory / receipt_name)}
    return headers, inputs


def verified_training_rows(input_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Require the original split and join training rows to verified source-header checkpoints.

    Parameters
    ----------
    input_root : Path
        Checkout containing the completed local split reconstruction.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        Training-only rows with source evidence and every consumed artifact's SHA-256.
    """
    directory = input_root / RECONSTRUCTION
    split, metadata, inputs = original_split(input_root)
    rows_path = directory / "rows.csv"
    skip = list(split.index[split["split"] != "train"] + 1)
    train = pd.read_csv(rows_path, dtype=str, keep_default_na=False, skiprows=skip,
                        usecols=[*SPLIT_REFERENCE_COLUMNS, *LABEL_COLUMNS])
    expected_ids = split.loc[split["split"] == "train", ["source", "record"]].reset_index(drop=True)
    if not train[["source", "record"]].equals(expected_ids):
        raise ValueError("Training-only split label read lost row identity")
    if train["record"].duplicated().any():
        raise ValueError("Challenge training record names are not unique")
    joined = []
    for source, release in RELEASES.items():
        headers, source_inputs = source_checkpoint(input_root, source, metadata,
                                                    [*HEADER_COLUMNS, "signal_sha256", "window_sha256"])
        inputs.update(source_inputs)
        selected = train[train["source"] == source].set_index("record")
        evidence = headers.reindex(selected.index)
        identity = ("source", "path", "signal_sha256", "window_sha256")
        if not all((evidence[name].fillna("") == selected[name].fillna("")).all() for name in identity):
            raise ValueError(f"Training/header identity mismatch: {source}")
        selected = selected.join(evidence[["age", "sex", "fs", "samples"]])
        selected["age_raw"] = selected.pop("age")
        selected["age"] = clean_age(selected["age_raw"])
        selected["male"] = male(selected["sex"])
        selected["release"] = release
        selected["record_id"] = source + ":" + selected.index
        selected["split_record_id"] = selected["record_id"]
        selected["split_rows_sha256"] = SPLIT_SHA256
        selected["full_signal_sha256"] = selected["signal_sha256"]
        selected["patient_id"] = ""
        selected["patient_identity_known"] = False
        selected["view_policy"] = "ssl_center_crop" if source.startswith("cpsc_") else "strict_10s"
        selected["record_primary"] = selected["primary"]
        joined.append(selected.reset_index())
    result = pd.concat(joined, ignore_index=True).sort_values(["source", "record"], kind="stable")
    if len(result) != len(train):
        raise ValueError("Unexpected source or missing Challenge training rows")
    return result, inputs


def verified_view(input_root: str, row: dict, expected: dict[str, str]) -> list[str]:
    """
    Verify a raw training pair before and after the frozen view and quality functions.

    Parameters
    ----------
    input_root : str
        Local checkout root.
    row : dict
        Validated training evidence, updated with canonical view measurements.
    expected : dict[str, str]
        Official SHA-256 values keyed by .hea and .mat suffixes.

    Returns
    -------
    list[str]
        View-contract or quality-policy exclusions.
    """
    raw = Path(input_root) / "data/raw" / row["release"]
    signal, start, samples, flags, reasons = read_verified_view(
        raw, row["path"], row["view_policy"], expected,
    )
    if signal is not None:
        reasons, reviews = assess(signal)
        canonical_hash = signal_sha256(signal)
        if row["source"] == "ningbo" and canonical_hash != row["window_sha256"]:
            raise ValueError(f"Ningbo window differs from its frozen split: {row['record_id']}")
        row.update(window_start=start, samples=samples, signal_window_sha256=canonical_hash,
                   review_flags=";".join(reviews), view_review_flags=flags)
    return reasons


def assess_training_record(item: tuple[str, dict, dict[str, str]]) -> dict[str, object]:
    """
    Apply the original source view and frozen quality policy to one approved training recording.

    Parameters
    ----------
    item : tuple[str, dict, dict[str, str]]
        Local data root, validated training row and official release checksums.

    Returns
    -------
    dict[str, object]
        Source evidence, canonical hash and quality decisions; no cohort or age selection.
    """
    input_root, row, checksums = item
    if row["split"] != "train":
        raise ValueError("A non-training row reached waveform assessment")
    stem = Path(row["path"])
    if stem.is_absolute() or ".." in stem.parts or stem.parts[:2] != ("training", row["source"]):
        raise ValueError("Unsafe Challenge training waveform path")
    row.update(backend="challenge_wfdb", waveform_path=stored(Path("data/raw") / row["release"] / stem),
               window_start=0, signal_window_sha256="", exclusion_reasons="", review_flags="",
               view_review_flags="", quality_status="excluded", supervised_target="",
               ssl_eligible=False, supervised_eligible=False)
    reasons = []
    if row["duplicate_status"] not in {"unique", "kept"}:
        reasons.append("original_" + row["duplicate_status"])
    elif int(row["samples"]) < 5000 or (row["view_policy"] == "strict_10s" and int(row["samples"]) != 5000):
        reasons.append("duration_contract")
    else:
        reasons = verified_view(input_root, row, checksums)
    row["exclusion_reasons"] = ";".join(reasons)
    row["quality_status"] = "passed" if not reasons else "excluded"
    row["ssl_eligible"] = not reasons
    defined = row["primary"] in {"0.0", "1.0", "0", "1"}
    row["supervised_eligible"] = not reasons and int(row["samples"]) == 5000 and defined
    row["supervised_target"] = row["primary"] if row["supervised_eligible"] else ""
    row["label_evidence"] = "human_read_source_snomed_proxy"
    row["crop_label_policy"] = "record_primary retained; longer-record crops are SSL only"
    row["worker_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return row


def supervised_window_duplicates(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve additional strict-window copies with the frozen mapped-label and known-sex rule.

    Parameters
    ----------
    rows : pd.DataFrame
        Executed training view assessments.

    Returns
    -------
    pd.DataFrame
        Rows with conservative strict-window duplicate eligibility and original evidence preserved.
    """
    rows = rows.copy()
    rows["supervised_duplicate_status"] = "not_applicable"
    strict = rows[(rows["quality_status"] == "passed") & (rows["samples"].astype(int) == 5000)].copy()
    if strict.empty:
        return rows
    strict["label_key"] = label_key(strict[list(LABEL_COLUMNS)].apply(pd.to_numeric, errors="coerce"))
    strict["signal_sha256"] = strict["signal_window_sha256"]
    status = duplicate_status(strict.set_index("record")[["signal_sha256", "label_key", "male"]])
    statuses = strict["record"].map(status["duplicate_status"])
    rows.loc[strict.index, "supervised_duplicate_status"] = statuses
    dropped = rows["supervised_duplicate_status"].isin(["dropped_copy", "dropped_conflict"])
    rows.loc[dropped, "supervised_eligible"] = False
    rows.loc[dropped, "supervised_target"] = ""
    return rows


def heldout_references(input_root: Path) -> tuple[set[str], set[str], dict[str, str]]:
    """Require a complete mechanical reference receipt and its exact canonical hash table."""
    receipt_path = REFERENCE_ROOT / "receipt.json"
    receipt = json.loads((input_root / receipt_path).read_text())
    if (not receipt.get("complete") or not receipt.get("coverage_complete") or
            receipt.get("scope") != "mechanical_reference_recovery" or
            receipt.get("original_split_sha256") != SPLIT_SHA256):
        raise ValueError("Complete mechanical held-out window references are required")
    table_path = Path(receipt["table_path"])
    if table_path.is_absolute() or ".." in table_path.parts:
        raise ValueError("Unsafe reference table path")
    if sha256_file(input_root / table_path) != receipt["table_sha256"]:
        raise ValueError("Held-out window reference table changed")
    refs = pd.read_csv(input_root / table_path, dtype=str, keep_default_na=False,
                       usecols=["canonical_sha256", "full_signal_sha256"])
    hashes = {stored(receipt_path): sha256_file(input_root / receipt_path),
              stored(table_path): receipt["table_sha256"]}
    return set(refs["canonical_sha256"]) - {""}, set(refs["full_signal_sha256"]) - {""}, hashes


def exclude_reference_overlaps(rows: pd.DataFrame, windows: set[str], full_signals: set[str]) -> pd.DataFrame:
    """Keep quality evidence but reject any training eligibility matching a known held-out reference."""
    rows = rows.copy()
    full = rows["full_signal_sha256"].isin(full_signals)
    window = rows["signal_window_sha256"].isin(windows)
    rows["heldout_overlap_reasons"] = [
        ";".join(reason for reason, found in (("heldout_full_signal", whole),
                                           ("heldout_canonical_window", crop)) if found)
        for whole, crop in zip(full, window, strict=True)
    ]
    rows.loc[full | window, ["supervised_eligible", "ssl_eligible"]] = False
    rows.loc[full | window, "supervised_target"] = ""
    return rows


def recover(input_root: Path, output: Path, workers: int, limit: int | None) -> dict[str, object]:
    """
    Stage verified training evidence after the exact split gate, without fitting a model.

    Parameters
    ----------
    input_root : Path
        Checkout holding local raw data and the completed split reconstruction.
    output : Path
        New root-relative output directory.
    workers : int
        One to four CPU processes.
    limit : int or None
        First training records per source for a resource profile, or every training record.

    Returns
    -------
    dict[str, object]
        Executed quality, label eligibility, input hashes and resource receipt.
    """
    if not 1 <= workers <= 4 or (limit is not None and limit < 1):
        raise ValueError("Use one to four workers and a positive profile limit")
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("Output must be relative to the input root")
    destination = input_root / output
    if destination.exists():
        raise FileExistsError(destination)
    start = time.monotonic()
    rows, input_hashes = verified_training_rows(input_root)
    window_refs, full_refs, reference_inputs = heldout_references(input_root)
    input_hashes.update(reference_inputs)
    selected = rows if limit is None else rows.groupby("source", sort=False).head(limit)
    checksums = {}
    for release in set(RELEASES.values()):
        path = Path("data/raw") / release / "SHA256SUMS.txt"
        checksums[release] = parse_checksums((input_root / path).read_text())
        input_hashes[stored(path)] = sha256_file(input_root / path)
    source_hashes = {stored(Path(name)): sha256_file(ROOT / name) for name in SOURCE_FILES}
    items = [(str(input_root), row,
              {suffix: checksums[row["release"]][row["path"] + suffix] for suffix in (".hea", ".mat")})
             for row in selected.to_dict("records")]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(assess_training_record, items, chunksize=16), 1):
            results.append(result)
            if index % 2048 == 0:
                print(f"Assessed {index}/{len(selected)} Challenge training records", flush=True)
    for name, digest in input_hashes.items():
        if sha256_file(input_root / name) != digest:
            raise ValueError(f"Recovery input changed during processing: {name}")
    assessed = supervised_window_duplicates(pd.DataFrame(results))
    assessed = exclude_reference_overlaps(assessed, window_refs, full_refs)
    worker_peak = int(assessed.pop("worker_peak_rss_kib").max())
    destination.mkdir(parents=True)
    path = destination / "training_rows.csv"
    assessed.to_csv(path, index=False)
    counts = {}
    for source, part in assessed.groupby("source"):
        labeled = part.loc[part["supervised_eligible"], "supervised_target"]
        counts[source] = {
            "records": len(part), "quality_passed": int((part["quality_status"] == "passed").sum()),
            "ssl_eligible": int(part["ssl_eligible"].sum()),
            "supervised_eligible": int(part["supervised_eligible"].sum()),
            "supervised_labels": labeled.value_counts().to_dict(),
            "heldout_overlap_excluded": int((part["heldout_overlap_reasons"] != "").sum()),
        }
    receipt = {
        "complete": True, "scope": "training_quality" if limit is None else "resource_profile",
        "records": len(assessed), "sources": counts, "original_split_sha256": SPLIT_SHA256,
        "input_sha256": input_hashes, "source_sha256": source_hashes,
        "table_path": stored(output / path.name), "table_sha256": sha256_file(path),
        "workers": workers, "wall_seconds": time.monotonic() - start,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "worker_peak_rss_kib": worker_peak,
        "child_user_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
        "child_system_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        "policy": "Original train split and mapped duplicate rule; historical strict/centered source views; "
                  "frozen quality policy; defined primary labels on original ten-second records only",
        "limitation": "New storage pointers and eligibility evidence; no claim of byte-identical historic "
                      "cohort reconstruction; no ages selected and no model scores",
    }
    write_json_atomic(destination / "receipt.json", receipt, sort_keys=True)
    return receipt
