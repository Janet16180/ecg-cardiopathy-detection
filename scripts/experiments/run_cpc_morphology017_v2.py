#!/usr/bin/env python3
"""Experiment 017 version 2: CPU checkpoint roundtrip for CUDA profile runs.

All scientific arms, inputs, and training steps stay in the frozen v1 module.
This version adds its source and protocol addendum to the experiment fingerprint,
uses a separate output directory, and restores profile states on CPU to compare
CPU checkpoint tensors on the same device.
"""

import argparse
from pathlib import Path

from ecg_experiment.files import sha256_file
from scripts.experiments import run_cpc_morphology017 as original


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/experiment017_morphology_templates_v2'
ORIGINAL_LOAD_INPUTS = original.load_inputs
ORIGINAL_VERIFY_ROUNDTRIP = original.verify_roundtrip


def load_inputs(args):
    data = ORIGINAL_LOAD_INPUTS(args)
    data['provenance']['code'].update({
        name: sha256_file(ROOT / name) for name in (
            'scripts/experiments/run_cpc_morphology017_v2.py',
            'docs/experiment-017-morphology-v2.md',
        )
    })
    return data


def verify_roundtrip(args, data, kind, directory):
    """Restore model, AdamW slots and RNG on CPU after a real GPU full pass."""
    cpu_args = argparse.Namespace(**vars(args))
    cpu_args.device = 'cpu'
    return ORIGINAL_VERIFY_ROUNDTRIP(cpu_args, data, kind, directory)


def main():
    # Patch only this process. The v1 source and its active immutable manifest
    # are untouched; all inherited calls use these versioned functions.
    original.load_inputs = load_inputs
    original.verify_roundtrip = verify_roundtrip
    original.OUTPUT = OUTPUT
    original.main()


if __name__ == '__main__':
    main()
