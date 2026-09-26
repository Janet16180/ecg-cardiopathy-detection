"""Verify the derived 250 Hz CPC cache against its immutable source cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecg_experiment.cpc_input_audit import historical_resample
from ecg_experiment.cpc_pool import Pool
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.sampled_training_dataset import SampledTrainingECGDataset

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data/processed/sampled_100k_plus_labels_v1"
CACHE = ROOT / "data/processed/sampled_100k_cpc_cache_v1"
OLD = ROOT / "data/processed/cpc_pool_40k"
RECEIPT = ROOT / "outputs/experiment018_cpc_data_scaling_v2/cache_verification.json"


def replay_rows(source: SampledTrainingECGDataset, cached: np.ndarray) -> tuple[int, int]:
    """Replay sampled source rows and compare overlapping PTB cache bytes."""
    selected = np.linspace(0, len(source) - 1, 20, dtype=np.int64).tolist()
    selected += [index for index, row in enumerate(source.rows) if row["source"] == "ptbxl"][:5]
    known_constant = {"mimic:46065905", "mimic:43144026"}
    selected += [index for index, row in enumerate(source.rows)
                 if row["record_id"] in known_constant]
    old = Pool(OLD)
    old_matches = 0
    for index in selected:
        row = source[index]
        expected = historical_resample(row["signal"])
        if not np.array_equal(cached[index], expected):
            raise ValueError(f"CPC cache transform mismatch at manifest row {index}")
        if row["source"] == "ptbxl":
            ecg_id = row["record_id"].split(":", 1)[1]
            if ecg_id in old.index:
                if not np.array_equal(cached[index], old.signals[old.index[ecg_id]]):
                    raise ValueError(f"Historical PTB transform mismatch at row {index}")
                old_matches += 1
    if old_matches < 1:
        raise ValueError("No historical PTB transform overlap checked")
    return len(selected), old_matches


def verify() -> dict[str, object]:
    """Check full cache bytes, sampled source replay, and historical PTB bits."""
    complete = json.loads((CACHE / "complete.json").read_text())
    source = SampledTrainingECGDataset(COHORT, purpose="ssl")
    signal_path = CACHE / "signals.npy"
    digest = sha256_file(signal_path)
    if digest != complete["signals_sha256"]:
        raise ValueError("CPC cache byte hash mismatch")
    cached = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    if cached.shape != (len(source), 12, 2500) or cached.dtype != np.float32:
        raise ValueError("CPC cache shape or dtype mismatch")
    for name, expected in (("train_manifest.csv", "source_manifest_sha256"),
                           ("metadata.json", "source_metadata_sha256")):
        if complete["identity"][expected] != sha256_file(COHORT / name):
            raise ValueError(f"CPC source {name} changed")
    source_replays, old_matches = replay_rows(source, cached)
    return {
        "status": "passed", "cache_sha256": digest,
        "record_count": len(source), "source_replays": source_replays,
        "historical_ptb_bitwise_matches": old_matches,
        "constant_lead_mimic_count": complete["constant_lead_mimic"],
    }


def main() -> None:
    """Write a local aggregate verification receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    write_json_atomic(args.receipt, verify())
    print(json.dumps(json.loads(args.receipt.read_text())), flush=True)


if __name__ == "__main__":
    main()
