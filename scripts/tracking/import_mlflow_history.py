"""Import completed, aggregate historical results into an MLflow registry."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ecg_experiment.tracking import EXPERIMENT_NAME, discover_historical_runs, import_historical_runs

SQLITE_PREFIX = "sqlite:///"


def import_runs(root: Path, uri: str, experiment_name: str, log_aggregate_artifacts: bool) -> None:
    """
    Import historical runs into MLflow and print how many were added.

    Parameters
    ----------
    root : Path
        Repository root.
    uri : str
        MLflow tracking URI; a SQLite database directory is created if missing.
    experiment_name : str
        MLflow experiment receiving the runs.
    log_aggregate_artifacts : bool
        Also log small aggregate result files as artifacts.
    """
    if uri.startswith(SQLITE_PREFIX):
        Path(uri.removeprefix(SQLITE_PREFIX)).parent.mkdir(parents=True, exist_ok=True)
    imported, skipped = import_historical_runs(
        root, uri, experiment_name=experiment_name, log_aggregate_artifacts=log_aggregate_artifacts,
    )
    print(f"Imported {imported}; already present {skipped}. Tracking URI: {uri}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--log-aggregate-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List historical runs without MLflow")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    List or import historical runs from the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    if args.dry_run:
        for record in discover_historical_runs(args.root):
            print(f"{record.stage:19} {record.source_id} {len(record.metrics)} metrics")
        return
    uri = args.tracking_uri or f"{SQLITE_PREFIX}{args.root.resolve() / 'outputs/mlflow/mlflow.db'}"
    import_runs(args.root, uri, args.experiment_name, args.log_aggregate_artifacts)


if __name__ == "__main__":
    main()
