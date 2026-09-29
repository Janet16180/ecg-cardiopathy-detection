"""Publish nested quality-first cohorts from 25k to 1M ECGs as manifests, without copying any waveform.

Three steps, see ``docs/clean-cohorts-v2.md``:

- ``build`` orders every candidate once and writes each tier as the first N rows of that order.
- ``score-pending`` runs on the machine that downloads the pending MIMIC records: it applies the quality
  policy to every pending record whose files exist and writes the results.
- ``resolve`` drops the failures from the saved order and writes new tier directories, so the tiers stay
  nested and the replacements are the next candidates of the same seeded order.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from ecg_experiment import clean_cohorts_v2 as cohorts
from ecg_experiment.clean_cohorts import UNION, read_signal, table
from ecg_experiment.ecg_quality import assess
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.waveforms import read_record

ROOT = cohorts.ROOT
SIZES = {"25k": 25_000, "50k": 50_000, "100k": 100_000, "150k": 150_000, "200k": 200_000, "500k": 500_000,
         "1m": 1_000_000}
SOURCES = ("ecg_experiment/clean_cohorts_v2.py", "ecg_experiment/clean_cohorts.py",
           "ecg_experiment/challenge_labels.py", "ecg_experiment/ecg_quality.py",
           "scripts/data/build_clean_cohorts_v2.py")
INPUTS = (cohorts.QUALITY_V1, cohorts.NINGBO, cohorts.NINGBO_RECEIPT, cohorts.CODE15, cohorts.MIMIC_RECORDS,
          cohorts.PTBXL_REFERENCE, ROOT / UNION / "heldout_references.csv",
          ROOT / UNION / "labels_fraction1.csv", ROOT / UNION / "labels_fraction0.1.csv")
CHECKED_PER_BACKEND = 20
MIMIC_SELECTION = ROOT / "data/processed/mimic_ssl_200k/metadata.json"


def candidate_order(seed: int) -> pd.DataFrame:
    """
    Assemble, order and deduplicate every candidate.

    Parameters
    ----------
    seed : int
        Selection seed.

    Returns
    -------
    pd.DataFrame
        The global order (``clean_cohorts_v2.COLUMNS`` plus ``block`` and ``order``).
    """
    ordered = cohorts.global_order(cohorts.curated_rows(), cohorts.mimic_local_rows(), cohorts.code15_rows(),
                                   cohorts.mimic_pending_rows(seed), seed)
    return cohorts.remove_overlaps(ordered, *cohorts.heldout_references())


def check_waveforms(rows: pd.DataFrame, seed: int) -> dict[str, int]:
    """
    Read a few rows of every locally readable backend and compare their waveform hash with the manifest.

    Parameters
    ----------
    rows : pd.DataFrame
        Passed rows of a tier.
    seed : int
        Sampling seed.

    Returns
    -------
    dict[str, int]
        Rows checked per backend.

    Raises
    ------
    ValueError
        If a waveform differs from its manifest hash.
    """
    checked = {}
    for backend, group in rows.groupby("backend"):
        if backend == "code15_archive":
            continue
        sample = group.sample(min(CHECKED_PER_BACKEND, len(group)), random_state=seed)
        for row in sample.itertuples(index=False):
            if backend == "challenge_wfdb":
                signal = read_record(ROOT / cohorts.CHALLENGE_ROOT,
                                     str(Path(row.path).relative_to(cohorts.CHALLENGE_ROOT)))
            else:
                signal = read_signal(backend, row.path, row.index)
            if signal_sha256(signal) != row.signal_sha256:
                raise ValueError(f"Waveform differs from the manifest: {row.record_id}")
        checked[backend] = len(sample)
    return checked


def tier_summary(rows: pd.DataFrame) -> dict[str, object]:
    """
    Composition of one tier.

    Parameters
    ----------
    rows : pd.DataFrame
        The tier's rows.

    Returns
    -------
    dict[str, object]
        Counts per source, block and quality status, patients, labels and review flags.
    """
    known = rows["patient_id"] != ""
    return {
        "records": len(rows),
        "source_counts": rows["source"].value_counts().sort_index().to_dict(),
        "block_counts": rows["block"].value_counts().to_dict(),
        "quality_status": rows["quality_status"].value_counts().to_dict(),
        "patients_known": int(rows.loc[known, "patient_id"].nunique()),
        "records_without_patient_id": int((~known).sum()),
        "label_available": int(rows["label_available"].sum()),
        "review_flagged": int((rows["review_flags"] != "").sum()),
    }


def publish_tier(output: Path, rows: pd.DataFrame, labels: dict[str, pd.DataFrame],
                 shared: dict[str, object], seed: int) -> dict[str, object]:
    """
    Write one tier directory: its manifest, the PTB-XL label files and metadata.

    Parameters
    ----------
    output : Path
        New directory; it must not exist.
    rows : pd.DataFrame
        The tier's rows in selection order.
    labels : dict[str, pd.DataFrame]
        PTB-XL label tables keyed by budget.
    shared : dict[str, object]
        Metadata common to all tiers.
    seed : int
        Seed for the waveform spot check.

    Returns
    -------
    dict[str, object]
        The tier metadata.

    Raises
    ------
    ValueError
        If the output exists, a label record is missing from the tier, or a record or waveform repeats.
    """
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    passed = rows[rows["quality_status"] == "passed"]
    if rows["record_id"].duplicated().any() or passed["signal_sha256"].duplicated().any():
        raise ValueError("A tier repeats a record or a waveform")
    if not set(labels["1"]["record_id"]) <= set(rows["record_id"]):
        raise ValueError("A PTB-XL label record is missing from the tier")
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    rows.to_csv(stage / "train_manifest.csv", index=False)
    for budget, frame in labels.items():
        frame.to_csv(stage / f"labels_fraction{budget}.csv", index=False)
    metadata = shared | tier_summary(rows) | {
        "waveforms_checked": check_waveforms(passed, seed),
        "table_sha256": {name: sha256_file(stage / name) for name in
                         ("train_manifest.csv", "labels_fraction1.csv", "labels_fraction0.1.csv")},
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def publish_tiers(order: pd.DataFrame, output_root: Path, suffix: str, shared: dict[str, object],
                  seed: int) -> dict[str, object]:
    """
    Write every feasible tier, and the largest feasible one when 1M is out of reach.

    Parameters
    ----------
    order : pd.DataFrame
        The candidate order.
    output_root : Path
        Parent of the tier directories.
    suffix : str
        Directory suffix, such as ``v2``.
    shared : dict[str, object]
        Metadata common to all tiers.
    seed : int
        Seed for the waveform spot checks.

    Returns
    -------
    dict[str, object]
        Composition and metadata hash per written tier.
    """
    clean_labeled = set(order.loc[order["source"] == "ptbxl", "record_id"])
    labels = {budget: table(UNION / f"labels_fraction{budget}.csv") for budget in ("1", "0.1")}
    labels = {budget: frame[frame["record_id"].isin(clean_labeled)] for budget, frame in labels.items()}
    tiers = {name: size for name, size in SIZES.items() if size <= len(order)}
    if len(tiers) < len(SIZES):
        tiers["max"] = len(order)
    written = {}
    for name, size in tiers.items():
        output = output_root / f"clean_{name}_{suffix}"
        tier = shared | {"tier": name, "size": size}
        metadata = publish_tier(output, order.iloc[:size], labels, tier, seed)
        written[name] = {key: metadata[key] for key in ("records", "source_counts", "block_counts",
                                                        "quality_status", "patients_known",
                                                        "records_without_patient_id", "label_available")}
        written[name]["metadata_sha256"] = sha256_file(output / "metadata.json")
    return written


def pending_projection(order: pd.DataFrame, failure_rate: float) -> dict[str, object]:
    """
    Project the size of each tier once pending rows are scored, at the local MIMIC failure rate.

    Parameters
    ----------
    order : pd.DataFrame
        The candidate order.
    failure_rate : float
        Share of downloaded MIMIC records rejected by the audit or the policy.

    Returns
    -------
    dict[str, object]
        Pending rows per tier and the expected number of candidates after scoring.
    """
    pending = order["quality_status"] == "pending"
    expected_total = int(len(order) - round(pending.sum() * failure_rate))
    return {"failure_rate": failure_rate, "candidates": len(order), "pending_candidates": int(pending.sum()),
            "expected_candidates_after_scoring": expected_total,
            "pending_rows_per_tier": {name: int(pending.iloc[:size].sum()) for name, size in SIZES.items()},
            "tiers_expected_to_fill": [name for name, size in SIZES.items() if size <= expected_total]}


def build(output_root: Path, work_dir: Path, seed: int) -> dict[str, object]:
    """Order every candidate, save the order and publish the tiers."""
    order = candidate_order(seed)
    work_dir.mkdir(parents=True, exist_ok=True)
    order_path = work_dir / "candidate_order.csv.gz"
    order.to_csv(order_path, index=False)
    selected = int(json.loads(MIMIC_SELECTION.read_text())["selected_records"])
    passed_local = int(order["block"].isin(["mimic_top_up", "mimic_local"]).sum())
    failure_rate = (selected - passed_local) / selected
    shared = {
        "schema_version": 2, "complete": True, "seed": seed, "nested_sizes": SIZES,
        "ordering": cohorts.__doc__.strip(),
        "storage_policy": "Manifest only; rows reference union or Chapman shards, raw WFDB files, or CODE-15 "
                          "archive members; no waveform is copied",
        "label_policy": "labels_fraction files hold the clean PTB-XL training proxy labels; every other row "
                        "is SSL only; label_available marks rows whose source label is defined",
        "candidate_order_sha256": sha256_file(order_path),
        "input_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in INPUTS},
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
    }
    receipt = {
        "candidates_per_block": order["block"].value_counts().to_dict(),
        "candidates_per_source": order["source"].value_counts().to_dict(),
        "projection": pending_projection(order, failure_rate),
        "tiers": publish_tiers(order, output_root, "v2", shared, seed),
    } | {key: shared[key] for key in ("seed", "candidate_order_sha256", "input_sha256", "source_sha256")}
    write_json_atomic(work_dir / "receipt.json", receipt, sort_keys=True)
    return receipt


def _score_pending(item: tuple[str, str]) -> dict[str, str]:
    record_id, path = item
    root = ROOT / cohorts.MIMIC_ROOT
    try:
        signal = read_record(root, str(Path(path).relative_to(cohorts.MIMIC_ROOT)))
    except ValueError as error:
        return {"record_id": record_id, "exclusion_reasons": "input_contract", "review_flags": "",
                "signal_sha256": "", "detail": str(error)}
    reasons, flags = assess(signal)
    return {"record_id": record_id, "exclusion_reasons": ";".join(reasons), "review_flags": ";".join(flags),
            "signal_sha256": signal_sha256(signal), "detail": ""}


def score_pending(work_dir: Path, results: Path, workers: int) -> dict[str, int]:
    """
    Apply the input contract and the quality policy to every pending MIMIC record found on disk.

    Parameters
    ----------
    work_dir : Path
        Directory holding ``candidate_order.csv.gz``.
    results : Path
        CSV to write; it must not exist.
    workers : int
        Number of processes.

    Returns
    -------
    dict[str, int]
        Records scored, passed and failed.

    Raises
    ------
    ValueError
        If the results file exists.
    """
    if results.exists():
        raise ValueError(f"Destination exists: {results}")
    order = pd.read_csv(work_dir / "candidate_order.csv.gz", dtype=str, keep_default_na=False)
    pending = order[order["quality_status"] == "pending"]
    present = pending[[(ROOT / f"{path}.hea").exists() for path in pending["path"]]]
    with Pool(workers) as pool:
        scored = pool.map(_score_pending, list(zip(present["record_id"], present["path"], strict=True)),
                          chunksize=64)
    frame = pd.DataFrame(scored, columns=["record_id", "exclusion_reasons", "review_flags", "signal_sha256",
                                          "detail"])
    frame.to_csv(results, index=False)
    failed = int((frame["exclusion_reasons"] != "").sum())
    return {"scored": len(frame), "passed": len(frame) - failed, "failed": failed}


def resolve(work_dir: Path, results: Path, output_root: Path, suffix: str, seed: int) -> dict[str, object]:
    """Drop failed pending records from the saved order and publish new tier directories."""
    order = pd.read_csv(work_dir / "candidate_order.csv.gz", dtype=str, keep_default_na=False)
    order["label_available"] = order["label_available"] == "True"
    scored = pd.read_csv(results, dtype=str, keep_default_na=False)
    resolved = cohorts.resolve_pending(order, scored)
    shared = {"schema_version": 2, "complete": True, "seed": seed, "nested_sizes": SIZES,
              "resolved_from_sha256": sha256_file(work_dir / "candidate_order.csv.gz"),
              "pending_results_sha256": sha256_file(results),
              "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES}}
    return {"tiers": publish_tiers(resolved, output_root, suffix, shared, seed),
            "still_pending": int((resolved["quality_status"] == "pending").sum())}


def main() -> None:
    """Parse arguments and run one step."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["build", "score-pending", "resolve"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/data_quality/clean_cohorts_v2")
    parser.add_argument("--results", type=Path, help="pending MIMIC quality results (score-pending, resolve)")
    parser.add_argument("--suffix", default="v2_resolved", help="tier directory suffix for resolve")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.step == "build":
        result = build(args.output_root, args.work_dir, args.seed)
    elif args.step == "score-pending":
        result = score_pending(args.work_dir, args.results, args.workers)
    else:
        result = resolve(args.work_dir, args.results, args.output_root, args.suffix, args.seed)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
