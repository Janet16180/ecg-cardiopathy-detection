"""Profile or evaluate frozen R/S introduced-change transfer after coordinator readiness."""

from __future__ import annotations

import argparse
import json

from ecg_experiment.ssl_synthetic_real_transfer import evaluate_transfer, profile_transfer


def main() -> None:
    """Run one admitted immutable stage; never fit a new model or resume a failed stage.

    Returns
    -------
    None
        Only aggregate completion/resource accounting is printed; patient data remain local.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "evaluate"), required=True)
    stage = parser.parse_args().stage
    result = profile_transfer() if stage == "profile" else evaluate_transfer()
    print(
        json.dumps(
            {
                "stage": stage,
                "wall_seconds": result["wall_seconds"],
                "completed": result.get("evaluation_completed", result.get("profile_gate_passed")),
            }
        )
    )
    if stage == "profile" and not result["profile_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
