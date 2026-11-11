"""Run MLC-03 Part A (signal quality) on the train and validation records, and once on the test roles.

Examples
--------
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
    -m scripts.experiments.run_mlc03_signal_quality measure --workers 16
PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc03_signal_quality measure --limit 40 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from ecg_experiment.mlc03_near_duplicates import (
    dense_pairs,
    key_table,
    load_fingerprints,
    run_fingerprints,
    run_long_records,
    shifted_candidates,
    verify_candidates,
)
from ecg_experiment.mlc03_signal_flags import (
    FLAGS,
    compute_flags,
    flag_counts,
    record_summaries,
    summary_quantiles,
)
from ecg_experiment.mlc03_signal_review import annotate, draw_pair_sheet, draw_sheet, review_sample
from ecg_experiment.mlc03_signal_runs import (
    DEVELOPMENT_ROLES,
    OUTPUT,
    WORK,
    assemble,
    code_hashes,
    frame_sha256,
    input_hashes,
    open_block,
    read_checked,
    read_identity,
    read_index,
    run_full_records,
    run_measures,
    write_receipt,
)
from ecg_experiment.paths import to_stored

REVIEW = OUTPUT.parent / "review/signal"
NEAR = OUTPUT / "near_duplicates"
DENSE_FLOOR = 0.8
NEAR_REVIEW = 0.95
NEAR_CONTROL = (0.90, 0.95)
MIN_OVERLAP_S = 4.0
NEAR_CODE = ["ecg_experiment/mlc03_near_duplicates.py", "ecg_experiment/resample.py",
             "scripts/experiments/run_mlc03_signal_quality.py"]
SEED = 20261111
REVIEW_SIZE = 24
FLAG_CODE = ["ecg_experiment/mlc03_signal_flags.py", "ecg_experiment/mlc03_signal_review.py",
             "scripts/experiments/run_mlc03_signal_quality.py"]
MEASURE_CODE = ["ecg_experiment/mlc03_signal_quality.py", "ecg_experiment/mlc03_signal_runs.py",
                "ecg_experiment/measurement_extractor042.py", "ecg_experiment/ecg_quality.py",
                "scripts/experiments/run_mlc03_signal_quality.py"]


def measure(args: argparse.Namespace, command: list[str]) -> None:
    """
    Measure every train and validation window (or a smoke sample) and publish the table.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    command : list[str]
        Command line, for the receipt.
    """
    name = "smoke" if args.smoke else "measures"
    directory = WORK / f"{name}_chunks"
    log = WORK / f"{name}.log"
    WORK.mkdir(parents=True, exist_ok=True)
    summary = run_measures(directory, DEVELOPMENT_ROLES, args.workers, args.limit, log=log)
    index = read_index(DEVELOPMENT_ROLES)
    if args.limit is not None:
        index = index.groupby("block", sort=False).head(args.limit).reset_index(drop=True)
    frame = assemble(directory, index)
    target = (WORK if args.smoke else OUTPUT) / f"{name}_train_validation.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    write_receipt(target.with_suffix(".receipt.json"), {
        "step": "measure", "command": command, "smoke": args.smoke, "limit": args.limit,
        "roles": list(DEVELOPMENT_ROLES), "records": len(frame), "columns": len(frame.columns),
        "summary": summary, "inputs": input_hashes(), "code": code_hashes(MEASURE_CODE),
        "table": {"path": str(target), "csv_sha256": frame_sha256(frame)}})
    print(json.dumps({**summary, "table": str(target)}, indent=1))


def full_records(args: argparse.Namespace, command: list[str]) -> None:
    """
    Measure the whole raw recording of every train and validation record longer than 10 s.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    command : list[str]
        Command line, for the receipt.
    """
    start = time.perf_counter()
    identity = read_identity(DEVELOPMENT_ROLES)
    frame = run_full_records(identity, args.workers)
    target = OUTPUT / "full_records_train_validation.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    write_receipt(target.with_suffix(".receipt.json"), {
        "step": "full-records", "command": command, "roles": list(DEVELOPMENT_ROLES),
        "records": len(frame), "wall_seconds": time.perf_counter() - start, "inputs": input_hashes(),
        "code": code_hashes(MEASURE_CODE),
        "table": {"path": to_stored(target), "csv_sha256": frame_sha256(frame)}})
    print(json.dumps({"records": len(frame), "table": str(target)}))


def development_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Measures, identity and flags of the train and validation records.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Measures (published table), identity (``read_identity``) and ``compute_flags`` output, all in the
        order of the measures table.
    """
    measures = pd.read_parquet(OUTPUT / "measures_train_validation.parquet")
    identity = read_identity(DEVELOPMENT_ROLES).set_index("record_key")
    identity = identity.loc[measures["record_key"]].reset_index()
    return measures, identity, compute_flags(measures)


def flags_step(command: list[str]) -> None:
    """
    Write the flags, their counts, their definitions and the quantiles behind the thresholds.

    Parameters
    ----------
    command : list[str]
        Command line, for the receipt.
    """
    measures, identity, flags = development_tables()
    table = flags.merge(identity[["record_key", "source", "role", "included", "duplicate_status"]],
                        on="record_key", validate="one_to_one")
    table.to_parquet(OUTPUT / "flags_train_validation.parquet", index=False)
    counts = flag_counts(flags, identity)
    counts.to_csv(OUTPUT / "flag_counts_train_validation.csv", index=False)
    every = flag_counts(flags, identity, included_only=False)
    every.to_csv(OUTPUT / "flag_counts_train_validation_all.csv", index=False)
    pd.DataFrame([{"flag": name, **spec} for name, spec in FLAGS.items()]).to_csv(
        OUTPUT / "flag_definitions.csv", index=False)
    summaries = record_summaries(measures)
    summaries.to_parquet(OUTPUT / "record_summaries_train_validation.parquet", index=False)
    quantiles = summary_quantiles(summaries, identity)
    quantiles.to_csv(OUTPUT / "measure_quantiles_train_validation.csv", index=False)
    write_receipt(OUTPUT / "flags_train_validation.receipt.json", {
        "step": "flags", "command": command, "records": len(flags),
        "included": int(identity["included"].sum()),
        "measures_csv_sha256": frame_sha256(measures), "flags_csv_sha256": frame_sha256(flags),
        "code": code_hashes(FLAG_CODE)})
    pivot = counts.groupby(["flag", "source"])["flagged"].sum().unstack().reindex([*FLAGS, "any_flag"])
    print(pivot.assign(total=pivot.sum(axis=1)).to_string())


def review_step(args: argparse.Namespace, command: list[str]) -> None:
    """
    Draw the review sheets of one flag: a fixed random sample of flagged and unflagged included records.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (``flag``, ``seed``, ``tag``).
    command : list[str]
        Command line, for the receipt.
    """
    measures, identity, flags = development_tables()
    included = identity["included"].to_numpy()
    flagged, unflagged = review_sample(flags[included], args.flag, args.seed, REVIEW_SIZE)
    index = read_index(DEVELOPMENT_ROLES).set_index("record_key")
    rows = {key: number for number, key in enumerate(measures["record_key"])}
    directory = REVIEW / (args.flag if args.tag is None else f"{args.flag}__{args.tag}")
    sample = []
    groups = (("flagged", flagged, 6, (2, 3)), ("unflagged", unflagged, 8, (2, 4)))
    for group, keys, per_sheet, layout in groups:
        for sheet_number, start in enumerate(range(0, len(keys), per_sheet), start=1):
            records = []
            for panel, key in enumerate(keys[start : start + per_sheet], start=1):
                place = index.loc[key]
                window = read_checked(open_block(place["block"], False), int(place["row"]),
                                      place["sha256_10s"], key)
                number = rows[key]
                note = annotate(args.flag, window, measures.iloc[number], flags.iloc[number])
                value = note.pop("value")
                source, role = identity.at[number, "source"], identity.at[number, "role"]
                title = f"{panel}. {key} ({source}, {role}) {value}"
                records.append({"window": window, "title": title, **note})
                sample.append({"flag": args.flag, "group": group, "sheet": f"{group}_{sheet_number:02d}.png",
                               "panel": panel, "record_key": key, "source": source, "role": role,
                               "value": value})
            draw_sheet(records, directory / f"{group}_{sheet_number:02d}.png",
                       f"{args.flag}: {group} records, sheet {sheet_number} (seed {args.seed})", *layout)
    frame = pd.DataFrame(sample)
    frame.to_csv(directory / "sample.csv", index=False)
    write_receipt(directory / "receipt.json", {
        "step": "review", "command": command, "flag": args.flag, "seed": args.seed,
        "flagged_pool": int(flags.loc[included, args.flag].sum()),
        "unflagged_pool": int((~flags.loc[included, args.flag]).sum()),
        "flagged_drawn": len(flagged), "unflagged_drawn": len(unflagged),
        "code": code_hashes(FLAG_CODE)})
    print(json.dumps({"flag": args.flag, "flagged": len(flagged), "unflagged": len(unflagged),
                      "directory": str(directory)}))


def pair_table(pairs: pd.DataFrame, first: str, second: str, identity: pd.DataFrame) -> pd.DataFrame:
    """
    Add identity columns of both records of every pair.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pairs with row numbers in columns ``first`` and ``second``.
    first, second : str
        Column names of the two row numbers.
    identity : pd.DataFrame
        Identity in the same row order (``record_key``, ``source``, ``role``, ``included``,
        ``duplicate_status``, ``sha256_10s``).

    Returns
    -------
    pd.DataFrame
        The pairs with ``<column>_a`` and ``<column>_b`` identity columns and ``same_hash``.
    """
    columns = ["record_key", "source", "role", "included", "duplicate_status", "sha256_10s"]
    a = identity.iloc[pairs[first].to_numpy()][columns].reset_index(drop=True).add_suffix("_a")
    b = identity.iloc[pairs[second].to_numpy()][columns].reset_index(drop=True).add_suffix("_b")
    table = pd.concat([pairs.reset_index(drop=True), a, b], axis=1)
    return table.assign(same_hash=table["sha256_10s_a"] == table["sha256_10s_b"])


def near_duplicates_step(args: argparse.Namespace, command: list[str]) -> None:
    """
    Search train and validation for near-duplicate waveforms at zero lag and at RR-anchored lags.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (``workers``).
    command : list[str]
        Command line, for the receipt.
    """
    start = time.perf_counter()
    index = read_index(DEVELOPMENT_ROLES)
    identity = read_identity(DEVELOPMENT_ROLES).set_index("record_key").loc[index["record_key"]].reset_index()
    run_fingerprints(WORK / "fingerprints_train_validation", index, args.workers)
    prints = load_fingerprints(WORK / "fingerprints_train_validation", index)
    long_records = run_long_records(identity, args.workers)
    row_of = {key: number for number, key in enumerate(identity["record_key"])}
    long_keys = {row_of[item["record_key"]]: item["keys"] for item in long_records}
    verify = {number: prints["verify"][number] for number in range(len(identity))}
    verify.update({row_of[item["record_key"]]: item["verify"] for item in long_records})
    pairs, nearest, scores = dense_pairs(prints["dense"], prints["dense"], DENSE_FLOOR, same_set=True)
    NEAR.mkdir(parents=True, exist_ok=True)
    dense = pair_table(pairs, "query", "reference", identity)
    dense.to_parquet(NEAR / "dense_pairs_train_validation.parquet", index=False)
    keys = identity["record_key"].to_numpy()
    neighbours = pd.DataFrame({"record_key": keys, "nearest_key": keys[nearest[:, 0]],
                               "nearest_score": scores[:, 0], "second_score": scores[:, 1]})
    neighbours.to_parquet(NEAR / "nearest_train_validation.parquet", index=False)
    candidates = shifted_candidates(key_table(prints["rr"], long_keys))
    verified = verify_candidates(candidates, verify)
    shifted = pair_table(verified, "first", "second", identity)
    shifted.to_parquet(NEAR / "shifted_pairs_train_validation.parquet", index=False)
    write_receipt(NEAR / "near_duplicates_train_validation.receipt.json", {
        "step": "near-duplicates", "command": command, "records": len(identity),
        "long_records": len(long_records),
        "dense_floor": DENSE_FLOOR, "dense_pairs": len(dense), "shifted_candidates": len(candidates),
        "wall_seconds": time.perf_counter() - start, "inputs": input_hashes(),
        "code": code_hashes(NEAR_CODE)})
    print(json.dumps({"dense_pairs": len(dense), "shifted_candidates": len(candidates),
                      "wall_seconds": time.perf_counter() - start}))


def near_duplicate_candidates() -> pd.DataFrame:
    """
    Non-exact candidate pairs of both searches, one row per pair, with the score and lag that found them.

    Returns
    -------
    pd.DataFrame
        ``record_key_a``, ``record_key_b``, ``search`` (``zero_lag`` or ``shifted``), ``score`` and ``lag``
        (500 Hz samples); pairs with the same first-10-s hash are left out, and shifted pairs need at least
        ``MIN_OVERLAP_S`` seconds of overlap.
    """
    dense = pd.read_parquet(NEAR / "dense_pairs_train_validation.parquet")
    shifted = pd.read_parquet(NEAR / "shifted_pairs_train_validation.parquet")
    dense = dense.loc[~dense["same_hash"], ["record_key_a", "record_key_b", "score"]]
    dense = dense.assign(search="zero_lag", lag=0)
    shifted = shifted[~shifted["same_hash"] & (shifted["overlap_s"] >= MIN_OVERLAP_S)]
    shifted = shifted.assign(search="shifted", score=shifted["correlation"], lag=shifted["fp_lag"] * 10)
    columns = ["record_key_a", "record_key_b", "search", "score", "lag"]
    return pd.concat([dense[columns], shifted[columns]], ignore_index=True)


def near_review_step(args: argparse.Namespace, command: list[str]) -> None:
    """
    Draw every candidate pair at or above ``NEAR_REVIEW`` and a fixed sample of pairs just below it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (``seed``).
    command : list[str]
        Command line, for the receipt.
    """
    import numpy as np

    candidates = near_duplicate_candidates()
    high = candidates[candidates["score"] >= NEAR_REVIEW].sort_values("score", ascending=False)
    band = candidates[(candidates["score"] >= NEAR_CONTROL[0]) & (candidates["score"] < NEAR_CONTROL[1])]
    band = band.sort_values(["record_key_a", "record_key_b", "search"]).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    control = band.iloc[np.sort(rng.choice(len(band), size=min(REVIEW_SIZE, len(band)), replace=False))]
    index = read_index(DEVELOPMENT_ROLES).set_index("record_key")

    def window(key: str) -> np.ndarray:
        place = index.loc[key]
        return read_checked(open_block(place["block"], False), int(place["row"]), place["sha256_10s"], key)

    directory = REVIEW / "near_duplicates"
    sample = []
    for group, frame in (("flagged", high), ("unflagged", control)):
        rows = frame.to_dict("records")
        for sheet_number, start in enumerate(range(0, len(rows), 8), start=1):
            pairs = []
            for panel, row in enumerate(rows[start : start + 8], start=1):
                title = (f"{panel}. {row['record_key_a']} vs {row['record_key_b']}, {row['search']} "
                         f"{row['score']:.3f}, lag {row['lag'] / 500:.2f} s")
                pairs.append({"first": window(row["record_key_a"]), "second": window(row["record_key_b"]),
                              "lag": row["lag"], "title": title})
                sample.append({"flag": "near_duplicate", "group": group,
                               "sheet": f"{group}_{sheet_number:02d}.png", "panel": panel,
                               "record_key": f"{row['record_key_a']}|{row['record_key_b']}",
                               "source": "", "role": "", "value": f"{row['search']} {row['score']:.4f}"})
            draw_pair_sheet(pairs, directory / f"{group}_{sheet_number:02d}.png",
                            f"near duplicates: {group} pairs, sheet {sheet_number} (black first record, "
                            f"red dashed second record shifted by the lag)")
    pd.DataFrame(sample).to_csv(directory / "sample.csv", index=False)
    write_receipt(directory / "receipt.json", {
        "step": "near-review", "command": command, "seed": args.seed, "high": len(high), "band": len(band),
        "control_drawn": len(control), "code": code_hashes(FLAG_CODE)})
    print(json.dumps({"high": len(high), "band": len(band), "control": len(control)}))


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line and run one step.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("measure", "full-records", "flags", "review", "near-duplicates",
                                         "near-review"))
    parser.add_argument("--flag", choices=tuple(FLAGS), help="review: flag to review")
    parser.add_argument("--seed", type=int, default=SEED, help="review: sampling seed")
    parser.add_argument("--tag", help="review: suffix of the review directory (for a re-sample)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="measure: first records of each cache block only")
    parser.add_argument("--smoke", action="store_true", help="measure: write to the scratch directory")
    args = parser.parse_args(argv)
    command = sys.argv if argv is None else ["run_mlc03_signal_quality", *argv]
    if args.step == "measure":
        if args.limit is not None and not args.smoke:
            raise SystemExit("--limit is for smoke tests only; add --smoke")
        measure(args, command)
    elif args.step == "full-records":
        full_records(args, command)
    elif args.step == "flags":
        flags_step(command)
    elif args.step == "review":
        review_step(args, command)
    elif args.step == "near-duplicates":
        near_duplicates_step(args, command)
    elif args.step == "near-review":
        near_review_step(args, command)


if __name__ == "__main__":
    main()
