"""Execute the forward existing-map diagnostic after explicit original-audit binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.ssl_synthetic_counterfactual_v2 import audit_counterfactual


def main() -> None:
    """Require the independently verified closure and retain a fresh diagnostic output.

    Returns
    -------
    None
        No model inference, threshold fitting or original verdict changes are performed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-audit-closure", type=Path, required=True)
    result = audit_counterfactual(parser.parse_args().original_audit_closure)
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
