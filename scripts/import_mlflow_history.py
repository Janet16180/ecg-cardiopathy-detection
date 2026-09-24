"""Import completed, aggregate historical results into an MLflow registry."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ecg_experiment.tracking import EXPERIMENT_NAME, discover_historical_runs, import_historical_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--log-aggregate-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List historical runs without MLflow")
    args = parser.parse_args()
    if args.dry_run:
        for record in discover_historical_runs(args.root):
            print(f"{record.stage:19} {record.source_id} {len(record.metrics)} metrics")
        return
    uri = args.tracking_uri or f"sqlite:///{args.root.resolve() / 'outputs' / 'mlflow' / 'mlflow.db'}"
    if uri.startswith("sqlite:///"):
        database = Path(uri.removeprefix("sqlite:///"))
        database.parent.mkdir(parents=True, exist_ok=True)
    imported, skipped = import_historical_runs(
        args.root, uri, experiment_name=args.experiment_name,
        log_aggregate_artifacts=args.log_aggregate_artifacts,
    )
    print(f"Imported {imported}; already present {skipped}. Tracking URI: {uri}")


if __name__ == "__main__":
    main()
