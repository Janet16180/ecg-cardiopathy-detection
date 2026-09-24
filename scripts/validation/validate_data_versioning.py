"""Check completed sources and frozen PTB splits against their receipts and hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.data_validation import DatasetValidator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/data-validation.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = DatasetValidator(root, root / args.config).validate()
    destination = root / args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated {sum(source['files'] for source in report['sources'])} source files and PTB patient splits")


if __name__ == "__main__":
    main()
