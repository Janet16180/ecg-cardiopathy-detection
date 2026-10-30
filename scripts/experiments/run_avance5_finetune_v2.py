"""Run one fixed BF16 full-backbone pass after a separately recorded resource stop."""

from __future__ import annotations

import argparse

from ecg_experiment import avance5_finetune_v2 as study


def main() -> None:
    """Run one explicit stage of the bounded successor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("profile", "prepare", "train"))
    args = parser.parse_args()
    result = getattr(study, args.stage)()
    print({key: value for key, value in result.items() if key in
           ("passed", "forecast_seconds", "records", "complete", "total_seconds")})


if __name__ == "__main__":
    main()
