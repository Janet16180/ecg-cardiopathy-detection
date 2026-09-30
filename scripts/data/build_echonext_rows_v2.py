"""Write EchoNext rows v2 over the v1 arrays, check the clean data and decide if it may enter the cohorts.

See ``docs/clean-echonext-v2.md``. The arrays are not copied: ``rows.csv`` refers to the v1 ``train.npy`` and
``val.npy``, whose SHA-256 are checked and bound. Only the ``ecg_key``, ``split`` and ``most_recent_ecg``
columns of the release metadata are read; test waveforms and labels are never loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import echonext_v2 as v2
from ecg_experiment.eda.echonext import OPEN_SPLITS, member_sha256
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored
from ecg_experiment.public_sources import signal_sha256

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data/processed/echonext_250hz_v1"
SOURCES = ("ecg_experiment/echonext_v2.py", "ecg_experiment/eda/echonext.py", "ecg_experiment/eda/signals.py",
           "scripts/data/build_echonext_rows_v2.py")
SEED = 20260930
GATE = {"usable_train_share": 0.95, "within_10_bpm": 0.95, "within_10_bpm_shuffled_max": 0.5,
        "limb_r2": 0.9}


def verify_v1() -> dict[str, str]:
    """Check the v1 arrays, rows and release metadata against the v1 cache metadata."""
    cache = json.loads((V1 / "metadata.json").read_text())
    for name, expected in cache["arrays_sha256"].items():
        if sha256_file(V1 / name) != expected:
            raise ValueError(f"EchoNext v1 {name} differs from its metadata")
    if sha256_file(V1 / "rows.csv") != cache["rows_sha256"]:
        raise ValueError("EchoNext v1 rows.csv differs from its metadata")
    name = "echonext_metadata_100k.csv"
    if member_sha256(name) != cache["release_sha256"][name]:
        raise ValueError("The EchoNext metadata differs from the verified release")
    return {to_stored(V1 / item): sha256_file(V1 / item) for item in ("rows.csv", "metadata.json")} | {
        to_stored(V1 / item): digest for item, digest in cache["arrays_sha256"].items()}


def duplicate_waveforms(rows: pd.DataFrame, arrays: dict[str, np.ndarray]) -> int:
    """Count the train and val ECGs whose waveform repeats another one."""
    pairs = zip(rows["split"], rows["row"], strict=True)
    hashes = pd.Series([signal_sha256(arrays[split][int(row)]) for split, row in pairs])
    return int(hashes.duplicated(keep=False).sum())


def gate(checks: dict[str, object]) -> dict[str, bool]:
    """Apply the cohort entry criteria of ``docs/clean-echonext-v2.md`` to the checks."""
    alignment = checks["alignment"]
    order = checks["lead_order"]["limb_relations"]
    return {
        "usable_train_share": checks["usable_train_share"] >= GATE["usable_train_share"],
        "rows_match_metadata": all(part["within_10_bpm"] >= GATE["within_10_bpm"]
                                   and part["within_10_bpm_shuffled"] <= GATE["within_10_bpm_shuffled_max"]
                                   for part in alignment.values()),
        "standard_lead_order": all(item["signs_match"] and item["r2"] >= GATE["limb_r2"]
                                   for item in order.values()),
        "no_duplicate_waveforms": checks["duplicate_waveforms"] == 0,
        "patients_disjoint": checks["train_patients_in_other_splits"] == 0,
    }


def build(output: Path) -> dict[str, object]:
    """Verify the inputs, write rows v2 and the checks, and return the metadata."""
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    start = time.monotonic()
    inputs = verify_v1()
    rows = pd.read_csv(V1 / "rows.csv", dtype=str, keep_default_na=False)
    rows["use"] = rows["use"] == "True"
    release = v2.read_release_columns(["patient_key", "most_recent_ecg"])
    table = v2.rows_v2(rows, release[["ecg_key", "split", "most_recent_ecg"]])
    train_patients = set(release.loc[release["split"] == "train", "patient_key"])
    other_patients = set(release.loc[release["split"] != "train", "patient_key"])
    arrays = {split: np.load(V1 / f"{split}.npy", mmap_mode="r") for split in OPEN_SPLITS}
    usable = {split: table[(table["split"] == split) & table["use_training"]] for split in OPEN_SPLITS}
    sample = np.sort(usable["train"].sample(2000, random_state=SEED)["row"].astype(int).to_numpy())
    checks = {
        "usable_train_share": float(table.loc[table["split"] == "train", "use_training"].mean()),
        "alignment": {split: v2.check_alignment(usable[split], arrays[split], 3000, SEED)
                      for split in OPEN_SPLITS},
        "lead_order": v2.check_lead_order(np.stack([arrays["train"][row] for row in sample])),
        "reason_profile": v2.reason_profile(table),
        "bound_shares": {split: v2.bound_shares(usable[split], arrays[split], 5000, SEED)
                         for split in OPEN_SPLITS},
        "one_per_patient": v2.one_per_patient(table),
        "duplicate_waveforms": duplicate_waveforms(table, arrays),
        "train_patients_in_other_splits": len(train_patients & other_patients),
    }
    passed = gate(checks)
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    table.to_csv(stage / "rows.csv", index=False)
    metadata = {
        "complete": True, "seed": SEED,
        "arrays": {split: to_stored(V1 / f"{split}.npy") for split in OPEN_SPLITS},
        "layout": "rows.csv row r of split s is ECG r of the v1 s.npy: float32 (12, 2500), 250 Hz, "
                  "standardized",
        "records": table["split"].value_counts().to_dict(),
        "use_training": table[table["use_training"]]["split"].value_counts().to_dict(),
        "use_evaluation": table[table["use_evaluation"]]["split"].value_counts().to_dict(),
        "checks": checks, "gate_thresholds": GATE, "gate": passed, "cohort_entry": all(passed.values()),
        "rows_sha256": sha256_file(stage / "rows.csv"), "input_sha256": inputs,
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
        "build_seconds": time.monotonic() - start,
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def main() -> None:
    """Parse the output directory and build rows v2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/echonext_250hz_v2")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
