"""Execute the prospective CPU-only condition-key comparison once."""

from ecg_experiment.condition_key_study import run


def main() -> None:
    """Execute and print the frozen primary result and decision."""
    result = run()
    print(result["primary"])
    print(result["decision"])


if __name__ == "__main__":
    main()
