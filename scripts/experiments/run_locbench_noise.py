"""Lead-standardized noise reading (061 amendment 3) for saved LocBench tables of one or more methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.locbench_noise import noise_summary, standardized_noise_table


def main() -> None:
    """Score each tables folder and write noise_std.json and noise_std.csv into it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", nargs="+", type=Path, help="LocBench tables folders")
    args = parser.parse_args()
    for folder in args.tables:
        table = standardized_noise_table(folder)
        summary = noise_summary(table)
        table.to_csv(folder / "noise_std.csv", index=False, float_format="%.17g")
        (folder / "noise_std.json").write_text(json.dumps(summary, indent=1))
        print(folder.name, json.dumps(summary))


if __name__ == "__main__":
    main()
