"""Prepare the frozen real TRAIN subset cache without model training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_cache import (
    join_eligibility,
    load_audited_inputs,
    prepare_cache,
    select_records,
    selection_identity,
)

OUTPUT = ROOT / "outputs/ssl_synthetic_localization/real_cache_v1"
SOURCE_FILES = (
    "ecg_experiment/ssl_synthetic_cache.py",
    "scripts/data/prepare_ssl_synthetic_real_cache.py",
    "docs/ssl-synthetic-localization-real-cache.md",
    "docs/ssl-synthetic-localization-pilot.md",
    "configs/ssl_synthetic_pilot_v1.json",
    "ecg_experiment/resample.py",
    "ecg_experiment/ssl_synthetic_inputs.py",
    "ecg_experiment/waveforms.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/ssl_synthetic_checksum_audit.py",
    "ecg_experiment/ssl_synthetic_partition.py",
    "ecg_experiment/ssl_synthetic_readiness.py",
    "pyproject.toml",
    "uv.lock",
)


def stream_output_bytes(path: Path) -> int:
    """
    Count local output file sizes without opening payloads.

    Parameters
    ----------
    path : Path
        Research stream output directory.

    Returns
    -------
    int
        Total regular file size observed at this instant.
    """
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    """
    Prepare and hash the fixed subset cache with aggregate admission evidence.

    Returns
    -------
    int
        Zero for a passing cache gate; two for an integrity, sufficiency or budget rejection.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()
    required = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in required) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set numerical thread limits to 1 and CUDA_VISIBLE_DEVICES to empty")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Real-cache output directory must be new")
    started = datetime.now(UTC).isoformat()
    wall, cpu = time.perf_counter(), time.process_time()
    config = json.loads((ROOT / "configs/ssl_synthetic_pilot_v1.json").read_bytes())
    manifest, audited, checksums, binding = load_audited_inputs(ROOT)
    selected = select_records(
        manifest["records"], config["real_subset_namespace"], config["real_subset_patients"]
    )
    identity = selection_identity(selected)
    selection = join_eligibility(selected, audited)
    for row in selection:
        key = f"{config['real_subset_namespace']}:{row['patient_id']}"
        row["selection_digest"] = hashlib.sha256(key.encode()).hexdigest()
    bytes_before = stream_output_bytes(OUTPUT.parent)
    projected = bytes_before + len(selection) * 12 * 2500 * 4 + 10_000_000
    if projected > config["cache_bytes_ceiling"]:
        raise ValueError("Prospective total stream output size exceeds cache ceiling")
    report = prepare_cache(
        selection, ROOT / "data/raw/ptb-xl/1.0.3", checksums, OUTPUT, config["minimum_usable_real_patients"]
    )
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    selection_report = {
        "schema_version": 1,
        "selection_sha256": identity,
        "subset_policy_origin": "530cbb9",
        "namespace": config["real_subset_namespace"],
        "source_binding": binding,
        "source_sha256": sources,
        "records": selection,
        "training_admitted": False,
    }
    with (OUTPUT / "selection.json").open("x") as stream:
        stream.write(json.dumps(selection_report, indent=2) + "\n")
    hashes = {name: sha256_file(OUTPUT / name) for name in ("signals.npy", "selection.json")}
    bytes_after = stream_output_bytes(OUTPUT.parent)
    report.update(binding)
    report.update(
        schema_version=1,
        cache_id="ssl_synthetic_real_cache_v1",
        source_sha256=sources,
        selection_sha256=identity,
        subset_policy_origin="530cbb9",
        pilot_policy_revision="4f6f268",
        output_sha256=hashes,
        cache_bytes=(OUTPUT / "signals.npy").stat().st_size,
        selection_bytes=(OUTPUT / "selection.json").stat().st_size,
        stream_output_bytes_before=bytes_before,
        stream_output_bytes_before_summary=bytes_after,
        cache_bytes_ceiling=config["cache_bytes_ceiling"],
        started_at_utc=started,
        finished_at_utc=datetime.now(UTC).isoformat(),
        wall_seconds=time.perf_counter() - wall,
        cpu_seconds=time.process_time() - cpu,
        numerical_threads=1,
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "wfdb")},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        outputs={
            name: to_stored(OUTPUT / name) for name in ("signals.npy", "selection.json", "summary.json")
        },
    )
    report["cache_budget_gate_passed"] = bytes_after + 100_000 <= config["cache_bytes_ceiling"]
    report["cache_gate_passed"] &= report["cache_budget_gate_passed"]
    with (OUTPUT / "summary.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "summary": to_stored(OUTPUT / "summary.json"),
                "cache_gate_passed": report["cache_gate_passed"],
                "usable_patients": report["usable_patients"],
            }
        )
    )
    return 0 if report["cache_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
