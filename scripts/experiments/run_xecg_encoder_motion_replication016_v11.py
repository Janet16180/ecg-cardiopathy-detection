"""Versioned preflight for the frozen xECG seed-47 replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.xecg_encoder_motion_v11 import preflight


def main() -> None:
    """Validate exact v11 identity and the installed device API first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight",), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    receipt = preflight(args.manifest, args.manifest_sha256, args.stage, args.device)
    print(json.dumps({"stage": args.stage, "status": receipt["status"]}), flush=True)


if __name__ == "__main__":
    main()
