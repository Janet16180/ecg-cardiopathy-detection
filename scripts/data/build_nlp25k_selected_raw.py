"""Build the local raw 25k cache after the live profile has finished.

This command is opt-in and does not profile or train a model. It writes only
``data/processed/nlp25k_selected_raw_v1`` after verifying the frozen source.
"""

from __future__ import annotations

import json

from ecg_experiment import ROOT
from ecg_experiment.compact_selected_raw import build_compact_cache
from ecg_experiment.cpc_subset25 import select_indices
from ecg_experiment.files import read_csv, sha256_file, sha256_json

MANIFEST = ROOT / "data/processed/sampled_100k_plus_labels_v1/train_manifest.csv"
SOURCE = ROOT / "data/processed/sampled_100k_cpc_cache_v1/signals.npy"
SOURCE_RECEIPT = SOURCE.with_name("complete.json")
SEAL = ROOT / "outputs/cache_sessions/nlp25k_v1/seal.json"
DESTINATION = ROOT / "data/processed/nlp25k_selected_raw_v1"
MANIFEST_SHA256 = "82a316a360a95dac3f24545c964afc1ad6b9654fa830541f6871b49371353288"
SOURCE_RECEIPT_SHA256 = "8ab358ee040cd876948e406067524e30dc20b4d10885c65d734f95889ab8e37b"
SOURCE_SHA256 = "df4a270e46a2870ac66b40bf73e8ad3c3dbf702aa08e98ac67de1393d772fef4"
SEAL_SHA256 = "2d26ae805bfe83e1238ecc0434108ad8ca5f2e951310e957c0b4a14d26ac3a63"
SELECTED_SHA256 = "43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791"


def main() -> None:
    """Verify source selection, then publish a byte-exact local compact cache."""
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("Frozen 115k source manifest changed")
    if sha256_file(SOURCE_RECEIPT) != SOURCE_RECEIPT_SHA256:
        raise ValueError("Frozen source cache receipt changed")
    source_receipt = json.loads(SOURCE_RECEIPT.read_text())
    if (source_receipt["identity"]["source_manifest_sha256"] != MANIFEST_SHA256
            or source_receipt["signals_sha256"] != SOURCE_SHA256):
        raise ValueError("Source cache receipt and manifest disagree")
    rows = read_csv(MANIFEST, required=("record_id", "source", "split"))
    if (len(rows) != 115_359 or any(row["split"] != "train" for row in rows)
            or len({row["record_id"] for row in rows}) != len(rows)):
        raise ValueError("Frozen training manifest contract changed")
    indices = select_indices([row["source"] for row in rows], 25_000, 18046)
    if sha256_json(indices.tolist()) != SELECTED_SHA256:
        raise ValueError("Experiment 019 selection changed")
    receipt = build_compact_cache(
        SOURCE, DESTINATION, SEAL, indices,
        seal_sha256=SEAL_SHA256, source_receipt_sha256=SOURCE_RECEIPT_SHA256,
        source_sha256=SOURCE_SHA256, selected_sha256=SELECTED_SHA256,
        source_manifest_sha256=MANIFEST_SHA256,
        builder_path=ROOT / "ecg_experiment/compact_selected_raw.py",
        entrypoint_path=ROOT / "scripts/data/build_nlp25k_selected_raw.py",
    )
    print(json.dumps({"output": str(DESTINATION),
                      "receipt_sha256": sha256_file(DESTINATION / "receipt.json"),
                      "signals_sha256": receipt["signals"]["sha256"],
                      "selected_indices_sha256": receipt["selected_indices_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
