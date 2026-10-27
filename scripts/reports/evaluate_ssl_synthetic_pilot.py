"""Run the reviewed frozen real-SSL synthetic-localization pilot evaluation."""

from __future__ import annotations

import json

from ecg_experiment.ssl_synthetic_evaluation import evaluate


def main() -> None:
    """Execute one source-bound evaluation after coordinator readiness and scientific review.

    Returns
    -------
    None
        Full local native maps and aggregate result are retained by the evaluation library.
    """
    result = evaluate()
    print(
        json.dumps(
            {
                "evaluation_completed": result["evaluation_completed"],
                "decision": result["decision"],
                "wall_seconds": result["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
