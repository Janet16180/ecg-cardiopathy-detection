#!/usr/bin/env python3
"""Inspect a fixed sample from the checksum-verified CODE-15% part 0 archive."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/raw/code-15pct/zenodo-4916206/exams_part0.zip"
OUTPUT = ROOT / "outputs/data_quality/code15_part0.json"
SEED = 42
SAMPLE_SIZE = 64


def main() -> None:
    receipt = json.loads((ROOT / "data/acquisition/code_15pct.json").read_text())
    if not any(item["name"] == ARCHIVE.name for item in receipt["verified_files"]):
        raise ValueError("CODE part 0 lacks an official checksum receipt")
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != "exams_part0.hdf5":
            raise ValueError("Unexpected CODE archive contents")
        with tempfile.TemporaryDirectory(prefix="code15_audit_") as directory:
            temporary = Path(directory) / members[0].filename
            with archive.open(members[0]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            with h5py.File(temporary, "r") as handle:
                keys = list(handle.keys())
                signal_key = "tracings" if "tracings" in handle else "signal"
                signal = handle[signal_key]
                selected = random.Random(SEED).sample(range(signal.shape[0]),
                                                        min(SAMPLE_SIZE, signal.shape[0]))
                counts = Counter()
                pads = Counter()
                for index in selected:
                    ecg = np.asarray(signal[index])
                    counts["read"] += 1
                    counts["shape_4096x12"] += ecg.shape == (4096, 12)
                    finite = bool(np.isfinite(ecg).all())
                    counts["finite"] += finite
                    if not finite or ecg.shape != (4096, 12):
                        continue
                    active = np.flatnonzero(np.any(ecg != 0, axis=1))
                    if len(active):
                        left, right = int(active[0]), int(ecg.shape[0] - 1 - active[-1])
                        pads[(left, right)] += 1
                    else:
                        counts["all_zero"] += 1
                    counts["any_near_flat_lead"] += bool(np.any(ecg.std(axis=0) < 0.01))
                result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                          "source_archive": str(ARCHIVE.relative_to(ROOT)),
                          "hdf5_keys": keys, "signal_key": signal_key,
                          "signal_shape": list(signal.shape), "signal_dtype": str(signal.dtype),
                          "sample_size": len(selected), "seed": SEED,
                          "counts": dict(counts),
                          "zero_padding_counts": {f"{left},{right}": count
                                                  for (left, right), count in sorted(pads.items())},
                          "near_flat_rule": "screening flag only; std < 0.01 in native stored units"}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
