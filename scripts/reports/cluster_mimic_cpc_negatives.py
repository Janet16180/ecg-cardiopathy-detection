"""Cluster fixed CPC-negative MIMIC ECGs and inspect machine summaries."""

from __future__ import annotations

import json

from ecg_experiment.mimic_negative_clustering import run


def main() -> None:
    """Run the exploratory negative-only cluster audit."""
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
