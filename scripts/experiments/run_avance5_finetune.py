"""Run the separately authorized post-delivery full-backbone GPU study."""

from __future__ import annotations

import argparse

from ecg_experiment import avance5_finetune as study


def main() -> None:
    """Run exactly one reviewable stage at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("profile", "prepare", "train"))
    args = parser.parse_args()
    result = getattr(study, args.stage)()
    print({key: value for key, value in result.items() if key in
           ("passed", "chosen", "records", "seconds", "complete", "trainable_parameters")})


if __name__ == "__main__":
    main()
