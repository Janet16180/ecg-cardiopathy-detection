"""Print an independent aggregate audit of the corrective development results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.pr74_result_audit import audit


def main() -> None:
    """Read completed local predictions and print only aggregate verification evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.evaluation_dir), indent=2))


if __name__ == "__main__":
    main()
