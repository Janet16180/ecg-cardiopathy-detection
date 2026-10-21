"""Check a completed independent expert annotation against separately established local bindings."""

import argparse
import json
from pathlib import Path

from ecg_experiment.localization.evidence_import_v2 import import_completed_evidence


def main() -> None:
    """Apply strict import admission without modifying annotations or claiming clinical truth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--signal-binding", type=Path, required=True)
    parser.add_argument("--reviewer-binding", type=Path, required=True)
    parser.add_argument("--beat-binding", type=Path)
    args = parser.parse_args()
    result = import_completed_evidence(
        args.annotation, args.signal_binding, args.reviewer_binding, args.beat_binding
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
