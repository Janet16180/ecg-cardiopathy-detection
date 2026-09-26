"""Prepare local review IDs for machine acute-MI alerts missed by CPC."""

from __future__ import annotations

import json

from ecg_experiment.mimic_priority_review import run


def main() -> None:
    """Write the small local-only machine-alert review packet."""
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
