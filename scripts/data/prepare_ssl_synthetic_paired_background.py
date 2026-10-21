"""Execute the frozen matched-control real-background CPU qualification once."""

import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main() -> None:
    """Print aggregate admission counts only; record identities stay in local output files."""
    import json

    from ecg_experiment.ssl_synthetic_paired_background_preparation import prepare

    report = prepare()
    print(
        json.dumps(
            {
                key: report[key]
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
