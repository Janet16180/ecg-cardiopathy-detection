"""Execute KEY-01 with the frozen decimal-membership parsing correction."""

from ecg_experiment.condition_key_study_v2 import run


def main() -> None:
    """Execute the corrected study and print its fixed result."""
    result = run()
    print(result["primary"])
    print(result["decision"])


if __name__ == "__main__":
    main()
