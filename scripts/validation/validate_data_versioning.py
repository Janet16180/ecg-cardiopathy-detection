"""Check completed sources and frozen PTB splits against their receipts and hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.data_validation import validate_datasets

ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/data-validation.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Validate the configured datasets and write the JSON report.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    # Relative paths are resolved against the repository root, not the working directory.
    report = validate_datasets(ROOT, ROOT / args.config)
    destination = ROOT / args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_files = sum(source["files"] for source in report["sources"])
    print(f"Validated {source_files} source files and PTB patient splits")


if __name__ == "__main__":
    main()
