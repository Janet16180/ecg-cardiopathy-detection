"""Compare fixed CPC negatives with official MIMIC machine ECG summaries."""

from __future__ import annotations

import json

from ecg_experiment.mimic_machine_disagreement import run


def main() -> None:
    """Write the local machine/CPC disagreement audit."""
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
