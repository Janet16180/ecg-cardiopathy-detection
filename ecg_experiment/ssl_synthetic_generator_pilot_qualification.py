"""Exact independently seeded pilot-domain admission without localization scores."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_generator import generate_pair
from ecg_experiment.ssl_synthetic_generator_qualification import (
    SOURCE_PATHS,
    _quantiles,
    _same_pair,
    gate_failures,
    pair_metrics,
)

PILOT_SOURCES = (
    "docs/ssl-synthetic-localization-generator-pilot-policy.md",
    "ecg_experiment/ssl_synthetic_generator_pilot_qualification.py",
    "scripts/validation/qualify_ssl_synthetic_generator_pilot.py",
    "tests/test_ssl_synthetic_generator_pilot.py",
)


def pilot_seed(role: str, index: int) -> int:
    """Derive the prospectively fixed independent latent identity.

    Parameters
    ----------
    role : str
        Exactly train, validation or evaluation.
    index : int
        Nonnegative within-role identity index.

    Returns
    -------
    int
        First eight SHA-256 bytes as an unsigned big-endian integer.
    """
    if role not in ("train", "validation", "evaluation") or index < 0:
        raise ValueError("Expected a declared role and nonnegative index")
    value = f"ssl_pilot_v1:synthetic:{role}:{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def pilot_cases() -> list[dict[str, Any]]:
    """Enumerate all planned cases without drawing or inspecting a waveform.

    Returns
    -------
    list[dict]
        Exact 3,060 source-independent case descriptors, with no dropped/retried identities.
    """
    rows = []
    for index in range(500):
        seed = pilot_seed("train", index)
        positive = index < 375
        mode = "focal" if not positive or index % 2 == 0 else "persistent"
        rows.append(
            {
                "cohort": "train/A",
                "index": index,
                "role": "train",
                "seed": seed,
                "profile": "A",
                "mode": mode,
                "target": positive,
                "nuisance": bool(seed & 1) if positive else bool(index % 2),
            }
        )
    for role, count, profiles in (("validation", 128, ("A",)), ("evaluation", 256, ("A", "B"))):
        for profile in profiles:
            for index in range(count):
                seed = pilot_seed(role, index)
                conditions = (
                    ("focal", True, bool(seed & 1)),
                    ("persistent", True, bool(seed & 1)),
                    ("focal", False, False),
                    ("focal", False, True),
                )
                for mode, target, nuisance in conditions:
                    rows.append(
                        {
                            "cohort": f"{role}/{profile}",
                            "index": index,
                            "role": role,
                            "seed": seed,
                            "profile": profile,
                            "mode": mode,
                            "target": target,
                            "nuisance": nuisance,
                        }
                    )
    return rows


def _role_integrity(cases: list[dict[str, Any]]) -> dict[str, Any]:
    role_seeds = {
        role: {row["seed"] for row in cases if row["role"] == role}
        for role in ("train", "validation", "evaluation")
    }
    count = sum(len(values) for values in role_seeds.values())
    total = len(set.union(*role_seeds.values()))
    return {
        "counts": {role: len(values) for role, values in role_seeds.items()},
        "unique_latent_seeds": total,
        "all_disjoint": count == total == 884,
    }


def _audit_cases(
    cases: list[dict[str, Any]], generator_version: str, config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generator = generate_pair
    if generator_version == "v2":
        from ecg_experiment.ssl_synthetic_generator_v2 import generate_pair as generate_pair_v2

        generator = generate_pair_v2
    failures: Counter[str] = Counter()
    failed_cases = replay_passed = 0
    rows = defaultdict(list)
    audit_rows = []
    for case in cases:
        args = {key: case[key] for key in ("seed", "profile", "mode", "target", "nuisance")}
        pair = generator(sample_rate=250, **args)
        replay_passed += _same_pair(pair, generator(sample_rate=250, **args))
        failed = gate_failures(pair, config)
        failures.update(failed)
        failed_cases += bool(failed)
        metrics = pair_metrics(pair)
        condition = case["mode"] if case["target"] else ("nuisance_only" if case["nuisance"] else "sham")
        stratum = f"{case['cohort']}/{condition}"
        rows[stratum].append(metrics)
        audit_rows.append(
            {
                **case,
                "gate_failures": failed,
                "metrics": metrics,
                "metadata": pair.metadata,
                "baseline_sha256": hashlib.sha256(pair.baseline.tobytes()).hexdigest(),
                "edited_sha256": hashlib.sha256(pair.edited.tobytes()).hexdigest(),
                "support_sha256": hashlib.sha256(pair.support.tobytes()).hexdigest(),
            }
        )
    return {
        "count": len(cases),
        "failed_cases": failed_cases,
        "gate_failure_counts": dict(failures),
        "deterministic_replays_passed": replay_passed,
        "all_passed": failed_cases == 0 and replay_passed == len(cases),
        "strata": {
            name: {"count": len(values), "metrics": _quantiles(values)} for name, values in rows.items()
        },
    }, audit_rows


def qualify_pilot_domain(output_directory: Path, generator_version: str = "v1") -> dict[str, Any]:
    """Audit every planned pilot pair and retain all failures before any model training.

    Parameters
    ----------
    output_directory : Path
        New repository-local directory, never overwritten.
    generator_version : str
        v1, or prospective narrowed-domain forward v2 after retaining a failed v1 receipt.

    Returns
    -------
    dict
        Source-bound aggregate admission; per-case parameters and hashes stay local.
    """
    started = time.monotonic()
    if generator_version not in ("v1", "v2"):
        raise ValueError("Expected generator v1 or v2")
    cases = pilot_cases()
    integrity = _role_integrity(cases)
    sources = (*SOURCE_PATHS, *PILOT_SOURCES)
    if generator_version == "v2":
        sources = (*sources, "ecg_experiment/ssl_synthetic_generator_v2.py")
    source_hashes = {path: sha256_file(ROOT / path) for path in sources}
    config = json.loads((ROOT / "configs/ssl_synthetic_generator_v1.json").read_text())
    output_directory.mkdir(parents=True, exist_ok=False)
    result, audit_rows = _audit_cases(cases, generator_version, config)
    audit_path = output_directory / "local_case_audit.json"
    audit_path.write_text(json.dumps(audit_rows, sort_keys=True) + "\n")
    thread_names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    threads = {name: os.environ.get(name) for name in thread_names}
    single_thread = all(value == "1" for value in threads.values())
    receipt = {
        "version": f"ssl_synthetic_pilot_domain_{generator_version}",
        "generator_version": generator_version,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes,
        "command": "PYTHONPATH=. .venv/bin/python -m scripts.validation.qualify_ssl_synthetic_generator_pilot"
        f" --generator {generator_version}",
        "seed_rule": "first8 SHA256 bytes, unsigned big endian: ssl_pilot_v1:synthetic:<role>:<index>",
        "role_integrity": integrity,
        "case_admission": result,
        "local_case_audit": {
            "path": to_stored(audit_path),
            "sha256": sha256_file(audit_path),
            "bytes": audit_path.stat().st_size,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threads": threads,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "all_passed": result["all_passed"] and integrity["all_disjoint"] and single_thread,
        "wall_seconds": time.monotonic() - started,
        "interpretation": "Exact declared generator-design domain only; no model or clinical score.",
    }
    (output_directory / "qualification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
