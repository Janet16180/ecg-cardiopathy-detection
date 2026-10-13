"""QRS-stratified ECGAD reading (061 amendment 2) for saved LocBench tables of one or more methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.locbench import load_tables
from ecg_experiment.locbench_strata import STRATA, strata_paired, strata_summary, strata_table


def main() -> None:
    """Score each tables folder, compare it with the first one, and write strata.json into each folder."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tables", nargs="+", type=Path, help="LocBench tables folders; the first is the reference"
    )
    args = parser.parse_args()
    computed = {}
    for folder in args.tables:
        tables = load_tables(folder, maps=True)
        computed[folder] = strata_table(tables["ecgad_maps"], tables["ecgad"]["ecg_id"].to_numpy())
    reference = args.tables[0]
    for folder, table in computed.items():
        result = {"summary": strata_summary(table), "reference": str(reference.name)}
        if folder != reference:
            result["paired_vs_reference"] = {
                name: strata_paired(table, computed[reference], name) for name in STRATA
            }
        table.to_csv(folder / "strata.csv", index=False, float_format="%.17g")
        (folder / "strata.json").write_text(json.dumps(result, indent=1))
        print(folder.name, json.dumps(result))


if __name__ == "__main__":
    main()
