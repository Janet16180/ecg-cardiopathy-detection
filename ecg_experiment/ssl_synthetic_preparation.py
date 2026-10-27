"""Bind successful input/generator gates and prepare label-budget-limited local banks."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_data import build_bank, latent_seed, save_bank
from ecg_experiment.ssl_synthetic_readiness import checked_relative

BASE = ROOT / "outputs/ssl_synthetic_localization"
PREPARATION = BASE / "pilot_v1/preparation"


def verify_hashes(hashes: dict[str, str], root: Path = ROOT) -> None:
    """
    Verify repository-relative source/input hashes without resolving data symlinks.

    Parameters
    ----------
    hashes : dict
        Repository-relative file path to expected SHA-256.
    root : Path
        Repository root, replaceable with a fixture for tests.
    """
    for name, expected in hashes.items():
        if sha256_file(root / checked_relative(name)) != expected:
            raise ValueError(f"Frozen input/source hash changed: {name}")


def verify_receipt(path: Path) -> dict[str, Any]:
    """
    Read a receipt and verify its declared sources, inputs and adjacent output files.

    Parameters
    ----------
    path : Path
        Local aggregate receipt.

    Returns
    -------
    dict
        Parsed receipt after every declared file hash matches.
    """
    receipt = json.loads(path.read_bytes())
    for key in ("source_sha256", "input_sha256", "input_audit_source_sha256"):
        verify_hashes(receipt.get(key, {}))
    for name, expected in receipt.get("output_sha256", {}).items():
        target = path.parent / checked_relative(name)
        if sha256_file(target) != expected:
            raise ValueError("Receipt output hash changed")
    return receipt


def write_new_json(path: Path, content: dict[str, Any]) -> None:
    """
    Write a new local scientific receipt, refusing overwrite and nonfinite JSON numbers.

    Parameters
    ----------
    path : Path
        New destination.
    content : dict
        Aggregate or local-record content.
    """
    with path.open("x") as stream:
        stream.write(json.dumps(content, indent=2, allow_nan=False) + "\n")


def stream_bytes(directory: Path = BASE) -> int:
    """
    Count all current stream artifacts for the conservative new-cache ceiling.

    Parameters
    ----------
    directory : Path
        Stream-local output root.

    Returns
    -------
    int
        Total bytes of existing files, including earlier qualification artifacts.
    """
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def prepare() -> dict[str, Any]:
    """
    Prepare A-only training/validation banks after all fixed-domain and real-cache gates pass.

    Returns
    -------
    dict
        Source-bound preparation receipt; no model has trained or scored any case.
    """
    from ecg_experiment.ssl_synthetic_generator_qualification import gate_failures
    from ecg_experiment.ssl_synthetic_generator_v2 import generate_pair

    if PREPARATION.exists() or PREPARATION.is_symlink():
        raise FileExistsError("Pilot preparation directory must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    config_path = ROOT / "configs/ssl_synthetic_pilot_v1.json"
    config = json.loads(config_path.read_text())
    cache_path = BASE / "real_cache_v1/summary.json"
    generator_path = BASE / "generator_pilot_v2/qualification.json"
    failed_path = BASE / "generator_pilot_v1/qualification.json"
    cache, generator = verify_receipt(cache_path), verify_receipt(generator_path)
    previous = verify_receipt(failed_path)
    if previous["all_passed"] or previous["case_admission"]["failed_cases"] != 28:
        raise ValueError("Original failed generator admission no longer matches its receipt")
    if not cache["cache_gate_passed"] or cache["usable_patients"] < config["minimum_usable_real_patients"]:
        raise ValueError("Real pretraining subset is not admitted")
    if not generator["all_passed"] or generator["case_admission"]["failed_cases"]:
        raise ValueError("Bounded generator failed fixed-domain qualification")
    if generator["case_admission"]["count"] != 3060 or not generator["role_integrity"]["all_disjoint"]:
        raise ValueError("Generator qualification does not cover the exact planned domain")
    gate_path = ROOT / "configs/ssl_synthetic_generator_v1.json"
    gates = json.loads(gate_path.read_text())
    counts = {
        "train": config["training_subjects"],
        "validation": config["validation_subjects"],
        "evaluation": config["evaluation_subjects"],
    }
    seeds = [{latent_seed(role, index) for index in range(count)} for role, count in counts.items()]
    if len(set.union(*seeds)) != sum(map(len, seeds)):
        raise ValueError("Synthetic latent identities overlap across roles")
    PREPARATION.mkdir(parents=True, exist_ok=False)
    banks = {}
    for role in ("train", "validation"):
        bank = build_bank(role, counts[role], "A", generate_pair, lambda pair: gate_failures(pair, gates))
        bank_dir = PREPARATION / f"{role}_bank"
        hashes = save_bank(bank, bank_dir)
        banks[role] = {
            "path": to_stored(bank_dir),
            "subjects": counts[role],
            "cases": len(bank["signals"]),
            "positive_cases": int((bank["kinds"] == "positive").sum()),
            "output_sha256": {to_stored(bank_dir / name): digest for name, digest in hashes.items()},
        }
    observed = stream_bytes()
    if observed > config["cache_bytes_ceiling"]:
        raise ValueError("Preparation exceeds the conservative entire-stream cache ceiling")
    source_names = [
        "ecg_experiment/ssl_synthetic_preparation.py",
        "ecg_experiment/ssl_synthetic_data.py",
        "scripts/data/prepare_ssl_synthetic_pilot.py",
        "configs/ssl_synthetic_pilot_v1.json",
        "configs/ssl_synthetic_generator_v1.json",
        "docs/ssl-synthetic-localization-pilot.md",
        "docs/ssl-synthetic-localization-pilot-generator-v2.md",
        "ecg_experiment/files.py",
        "ecg_experiment/paths.py",
        "ecg_experiment/ssl_synthetic_readiness.py",
        "pyproject.toml",
        "uv.lock",
    ]
    sources = {name: sha256_file(ROOT / name) for name in source_names}
    sources.update(generator["source_sha256"])
    report = {
        "schema_version": 1,
        "pilot_id": config["pilot_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "preparation_gate_passed": True,
        "banks": banks,
        "real_cache": to_stored(BASE / "real_cache_v1/signals.npy"),
        "real_patients": cache["usable_patients"],
        "input_sha256": {
            to_stored(path): sha256_file(path) for path in (cache_path, generator_path, failed_path)
        },
        "source_sha256": sources,
        "bank_sha256": {
            name: digest for bank in banks.values() for name, digest in bank["output_sha256"].items()
        },
        "stream_bytes": observed,
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
        "training_started": False,
        "performance_scored": False,
        "training_admitted": False,
    }
    write_new_json(PREPARATION / "summary.json", report)
    return report
