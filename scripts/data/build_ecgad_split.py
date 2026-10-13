"""Write the frozen ECGAD development/holdout split to docs/ecgad-split-v1.json."""

from __future__ import annotations

import numpy as np

from ecg_experiment.ecgad import SPLIT_PATH, build_split, load_labels, load_match


def main() -> None:
    """Build the split from the benchmark labels and the PTB-XL patient IDs."""
    match = load_match()
    labeled = np.asarray(load_labels()).any(axis=(1, 2))
    table = build_split(match, labeled)
    table.to_json(SPLIT_PATH, orient="split", index=False)
    print(table.groupby(["partition", "labeled"]).size())


if __name__ == "__main__":
    main()
