"""Profile or count CPC diagnostic-proxy flags on cached MIMIC ECGs."""

from __future__ import annotations

import argparse
import json

from ecg_experiment.mimic_cpc_flags import run


def main() -> None:
    """Run the read-only inference audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(profile=args.profile), indent=2))


if __name__ == "__main__":
    main()
