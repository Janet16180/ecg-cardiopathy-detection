"""Write the frozen LocBench v1 partition of PTB-XL folds 1-8 to docs/locbench-partition-v1.json."""

from __future__ import annotations

from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.locbench_data import PARTITION_PATH, build_partition


def main() -> None:
    """Build the partition from the PTB-XL metadata and print its counts."""
    table = build_partition(load_metadata(), load_statements())
    table.to_json(PARTITION_PATH, orient="split", index=False)
    print(
        table.groupby("partition").agg(
            records=("ecg_id", "size"),
            patients=("patient_id", "nunique"),
            clean_normals=("clean_normal", "sum"),
        )
    )
    noisy = table[table.noisy_leads != ""]
    print("records with lead-specific noise:", noisy.groupby("partition").size().to_dict())


if __name__ == "__main__":
    main()
