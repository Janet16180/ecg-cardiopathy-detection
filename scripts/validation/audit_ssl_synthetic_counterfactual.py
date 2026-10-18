"""Run the prospectively frozen existing-map response diagnostic, without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.ssl_synthetic_counterfactual import audit_counterfactual


def main() -> None:
    """Require original independent reproduction and retain a new descriptive audit.

    Returns
    -------
    None
        Original model states, native maps, calibration and failed advancement are read-only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-audit", type=Path, required=True)
    args = parser.parse_args()
    result = audit_counterfactual(args.original_audit)
    print(
        json.dumps(
            {
                "audit_completed": result["audit_completed"],
                "wall_seconds": result["wall_seconds"],
                "progress": result["progress"],
            }
        )
    )


if __name__ == "__main__":
    main()
