"""Cluster one MIMIC ECG per patient using self-supervised CPC features."""

from __future__ import annotations

import json

from ecg_experiment.mimic_cpc_clustering import run


def main() -> None:
    """Run and print the exploratory cluster report."""
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
