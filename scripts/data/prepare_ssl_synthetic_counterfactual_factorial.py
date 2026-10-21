"""Run the frozen factorial new-validation CPU qualification once."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main() -> None:
    """Keep record identities in local output files and print only aggregate admission counts."""
    import json

    from ecg_experiment.ssl_synthetic_counterfactual_factorial_preparation import prepare

    try:
        result = prepare()
    except Exception:
        raise RuntimeError("Factorial preparation stopped; inspect local failure receipt") from None
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "qualification_gate_passed",
                    "backgrounds",
                    "cases",
                    "failed_cases",
                    "cpu_seconds",
                    "wall_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
