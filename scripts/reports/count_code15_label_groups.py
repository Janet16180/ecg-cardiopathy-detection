"""Count released CODE-15 diagnostic flags and export their exam IDs."""

from __future__ import annotations

import json

from ecg_experiment.code15_label_groups import run


def main() -> None:
    """Write a compact local-only label and ID report."""
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
