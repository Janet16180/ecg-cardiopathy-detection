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

from ecg_experiment.mlc03_signal_flags import (
    FLAGS,
    compute_flags,
    flag_counts,
    record_summaries,
    summary_quantiles,
)
from ecg_experiment.mlc03_signal_review import annotate, draw_sheet, review_sample
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


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line and run one step.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("measure", "full-records", "flags", "review"))
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


if __name__ == "__main__":
    main()
