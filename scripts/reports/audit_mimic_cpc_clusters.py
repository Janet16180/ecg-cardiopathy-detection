"""Summarize simple signal-quality markers within MIMIC CPC clusters."""

from __future__ import annotations

import json

from ecg_experiment.mimic_cluster_qc import run


def main() -> None:
    """Run the read-only cluster waveform audit."""
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
