#!/usr/bin/env python3
"""Experiment 017 version 2: CPU checkpoint roundtrip for CUDA profile runs.

All scientific arms, inputs, and training steps stay in the v1 module. This
version adds its source and protocol addendum to the experiment fingerprint,
uses a separate output directory, and restores profile states on CPU to compare
CPU checkpoint tensors on the same device.
"""

from pathlib import Path

from scripts.experiments import run_cpc_morphology017 as original

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment017_morphology_templates_v2"
EXTRA_CODE = ("scripts/experiments/run_cpc_morphology017_v2.py", "docs/experiment-017-morphology-v2.md")
ROUNDTRIP_DEVICE = "cpu"


def main() -> None:
    """Run the v1 entry point with the versioned output, fingerprint and CPU roundtrip."""
    original.main(output=OUTPUT, extra_code=EXTRA_CODE, roundtrip_device=ROUNDTRIP_DEVICE)


if __name__ == "__main__":
    main()
