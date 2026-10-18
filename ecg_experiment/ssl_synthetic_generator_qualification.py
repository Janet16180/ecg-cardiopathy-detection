"""Aggregate qualification of paired engineering truth, without model scoring."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_generator import (
    RAW_RATE,
    SUPPORT_FLOOR_MV,
    InterventionPair,
    couple_limb_leads,
    edit_profile,
    generate_pair,
)

SOURCE_PATHS = (
    "docs/ssl-synthetic-localization-generator-policy.md",
    "configs/ssl_synthetic_generator_v1.json",
    "ecg_experiment/ssl_synthetic_generator.py",
    "ecg_experiment/ssl_synthetic_generator_qualification.py",
    "scripts/validation/qualify_ssl_synthetic_generator.py",
    "tests/test_ssl_synthetic_generator.py",
    "ecg_experiment/resample.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/waveforms.py",
    "ecg_experiment/files.py",
    "ecg_experiment/__init__.py",
    "pyproject.toml",
    "uv.lock",
)


def _limb_error(signal: np.ndarray) -> float:
    expected = couple_limb_leads(signal[[0, 1, 6, 7, 8, 9, 10, 11]].astype(np.float64))
    return float(np.max(np.abs(signal - expected)))


def _slope_metrics(delta: np.ndarray, sample_rate: int) -> tuple[float, float]:
    peaks = np.max(np.abs(delta), axis=1)
    active = peaks > 1e-6
    if not active.any():
        return 0.0, 0.0
    values = delta[active]
    peak = peaks[active, None]
    derivative = np.diff(values, axis=1, prepend=0) * sample_rate / peak
    present = np.abs(values) > np.maximum(1e-7, peak * 1e-6)
    starts = present & ~np.pad(present[:, :-1], ((0, 0), (1, 0)))
    ends = present & ~np.pad(present[:, 1:], ((0, 0), (0, 1)))
    backward = np.diff(values, axis=1, append=0) * sample_rate / peak
    edge = max(np.max(np.abs(derivative[starts]), initial=0), np.max(np.abs(backward[ends]), initial=0))
    return float(np.max(np.abs(derivative))), float(edge)


def pair_metrics(pair: InterventionPair) -> dict[str, float]:
    """Measure actual support, algebra, occupancy, finite differences and tails.

    Parameters
    ----------
    pair : InterventionPair
        The paired final observations and label-only raw arrays/domains.

    Returns
    -------
    dict[str, float]
        Diagnostics in mV, fractions and amplitude-normalized slopes per second.
    """
    delta = pair.difference
    energy = delta**2
    total = float(energy.sum())
    raw_delta = pair.raw_edited.astype(np.float64) - pair.raw_baseline.astype(np.float64)
    slope, _ = _slope_metrics(delta, pair.metadata["sample_rate"])
    raw_slope, raw_edge = _slope_metrics(raw_delta, RAW_RATE)
    expected = np.abs(delta) > SUPPORT_FLOOR_MV
    return {
        "occupancy": float(pair.support.mean()),
        "candidate_fraction": float(pair.candidate.mean()),
        "conditional_candidate_occupancy": float(pair.support.sum() / max(pair.candidate.sum(), 1)),
        "energy_mv2_samples": total,
        "subfloor_energy_fraction": float(energy[~pair.support].sum() / total) if total else 0.0,
        "subfloor_changed_cell_fraction": float(((delta != 0) & ~pair.support).mean()),
        "tail_energy_fraction": float(energy[~pair.nominal].sum() / total) if total else 0.0,
        "tail_support_fraction": float((pair.support & ~pair.nominal).sum() / max(pair.support.sum(), 1)),
        "outside_candidate_max_mv": float(np.max(np.abs(delta[~pair.candidate]), initial=0)),
        "outside_support_max_mv": float(np.max(np.abs(delta[~pair.support]), initial=0)),
        "support_mismatches": float(np.count_nonzero(expected != pair.support)),
        "support_outside_candidate": float(np.count_nonzero(pair.support & ~pair.candidate)),
        "limb_error_mv": max(
            _limb_error(array) for array in (pair.baseline, pair.edited, pair.raw_baseline, pair.raw_edited)
        ),
        "maximum_voltage_mv": max(float(np.max(np.abs(pair.baseline))), float(np.max(np.abs(pair.edited)))),
        "maximum_edit_mv": float(np.max(np.abs(delta))),
        "maximum_slope_per_second": max(slope, raw_slope),
        "maximum_raw_edge_slope_per_second": raw_edge,
    }


def gate_failures(pair: InterventionPair, config: dict[str, Any]) -> list[str]:
    """Apply prospectively fixed gates to one pair, returning all failures.

    Parameters
    ----------
    pair : InterventionPair
        Paired observations whose declared support is checked against actual differences.
    config : dict
        Committed v1 grid and thresholds.

    Returns
    -------
    list[str]
        Failed gates, empty only for an admitted engineering example.
    """
    metrics = pair_metrics(pair)
    limit = config[f"maximum_{pair.metadata['intervention']['mode']}_occupancy"]
    positive = pair.metadata["target"]
    checks = {
        "finite_arrays": all(
            np.isfinite(value).all()
            for value in (pair.baseline, pair.edited, pair.raw_baseline, pair.raw_edited)
        ),
        "shape": pair.edited.shape == pair.baseline.shape == (12, pair.metadata["sample_rate"] * 10)
        and pair.raw_edited.shape == pair.raw_baseline.shape == (12, 5000),
        "exact_support": metrics["support_mismatches"] == 0,
        "support_outside_candidate": metrics["support_outside_candidate"] == 0,
        "outside_support_floor": metrics["outside_support_max_mv"] <= SUPPORT_FLOOR_MV,
        "occupancy": 0 < metrics["occupancy"] <= limit if positive else metrics["occupancy"] == 0,
        "negative_pair_identical": positive or np.array_equal(pair.baseline, pair.edited),
        "limb_algebra": metrics["limb_error_mv"] <= 1e-6,
        "voltage_scale": metrics["maximum_voltage_mv"] < 10 and metrics["maximum_edit_mv"] <= 0.50,
        "slope": metrics["maximum_slope_per_second"] <= config["maximum_slope_per_second"],
        "raw_edge_slope": metrics["maximum_raw_edge_slope_per_second"]
        <= config["maximum_raw_edge_slope_per_second"],
        "subfloor_energy": metrics["subfloor_energy_fraction"] <= config["maximum_subfloor_energy_fraction"],
        "finite_candidate_tails": metrics["outside_candidate_max_mv"] <= 1e-6,
    }
    return [name for name, passed in checks.items() if not passed]


def _same_pair(first: InterventionPair, second: InterventionPair) -> bool:
    attributes = ("baseline", "edited", "support", "candidate", "raw_baseline", "raw_edited", "nominal")
    return first.metadata == second.metadata and all(
        np.array_equal(getattr(first, name), getattr(second, name)) for name in attributes
    )


def _independent_streams(seed: int, pair: InterventionPair) -> bool:
    arguments = {
        "profile": pair.metadata["profile"],
        "mode": pair.metadata["intervention"]["mode"],
        "sample_rate": pair.metadata["sample_rate"],
    }
    nuisance_changed = generate_pair(seed, nuisance_seed=seed + 10000, **arguments)
    edit_changed = generate_pair(seed, edit_seed=seed + 20000, **arguments)
    no_target = generate_pair(seed, target=False, **arguments)
    return (
        nuisance_changed.metadata["latent"] == pair.metadata["latent"]
        and nuisance_changed.metadata["intervention"] == pair.metadata["intervention"]
        and edit_changed.metadata["latent"] == pair.metadata["latent"]
        and edit_changed.metadata["nuisance"] == pair.metadata["nuisance"]
        and np.array_equal(edit_changed.baseline, pair.baseline)
        and no_target.metadata["nuisance"] == pair.metadata["nuisance"]
        and np.array_equal(no_target.baseline, pair.baseline)
    )


def _invalid_controls(config: dict[str, Any]) -> dict[str, bool]:
    pair = generate_pair(17, sample_rate=500)
    broad = replace(pair, support=np.ones_like(pair.support))
    empty = replace(pair, support=np.zeros_like(pair.support))
    wrong_limb = pair.edited.copy()
    wrong_limb[2] += 0.02
    hard = pair.baseline.copy()
    hard[6, 1000:1100] += 0.1
    hard_pair = replace(pair, edited=hard, raw_edited=hard)
    nuisance = generate_pair(17, nuisance_seed=170001)
    unpaired = replace(pair, baseline=generate_pair(17, nuisance_seed=170001, sample_rate=500).baseline)
    return {
        "broad_mask_rejected": {"occupancy", "exact_support"}.issubset(gate_failures(broad, config)),
        "empty_mask_rejected": "exact_support" in gate_failures(empty, config),
        "hard_seam_rejected": "slope" in gate_failures(hard_pair, config),
        "corrupt_limb_rejected": "limb_algebra" in gate_failures(replace(pair, edited=wrong_limb), config),
        "unpaired_nuisance_rejected": "exact_support" in gate_failures(unpaired, config),
        "distinct_nuisance_is_observable": not np.array_equal(nuisance.baseline, generate_pair(17).baseline),
    }


def _save_pair(directory: Path, name: str, pair: InterventionPair) -> dict[str, str | int]:
    destination = directory / f"{name}.npz"
    np.savez_compressed(
        destination,
        baseline=pair.baseline,
        edited=pair.edited,
        support=pair.support,
        candidate=pair.candidate,
        difference=pair.difference,
        raw_baseline=pair.raw_baseline,
        raw_edited=pair.raw_edited,
        nominal=pair.nominal,
        metadata=np.array(json.dumps(pair.metadata, sort_keys=True)),
    )
    return {
        "path": to_stored(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def _quantiles(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: dict(
            zip(
                ("minimum", "p05", "median", "p95", "maximum"),
                np.quantile([row[key] for row in rows], [0, 0.05, 0.5, 0.95, 1]).tolist(),
                strict=True,
            )
        )
        for key in rows[0]
    }


def _positive_grid(config: dict[str, Any], directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = defaultdict(list)
    failures: Counter[str] = Counter()
    index = []
    replay_count = independence_count = 0
    grid = product(config["profiles"], config["modes"], config["sample_rates"], config["seed_ids"])
    for profile, mode, rate, seed in grid:
        pair = generate_pair(seed, profile=profile, mode=mode, sample_rate=rate)
        replay_count += _same_pair(pair, generate_pair(seed, profile=profile, mode=mode, sample_rate=rate))
        independence_count += _independent_streams(seed, pair)
        failures.update(gate_failures(pair, config))
        rows[f"{profile}/{mode}/{rate}"].append(pair_metrics(pair))
        index.append(_save_pair(directory, f"positive_{profile}_{mode}_{rate}_{seed:03d}", pair))
    count = len(index)
    report = {
        "count": count,
        "deterministic_replays_passed": replay_count,
        "independent_substream_checks_passed": independence_count,
        "gate_failure_counts": dict(failures),
        "strata": {
            name: {"count": len(values), "metrics": _quantiles(values)} for name, values in rows.items()
        },
        "all_passed": not failures and replay_count == count and independence_count == count,
    }
    return report, index


def _negative_grid(config: dict[str, Any], directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: Counter[str] = Counter()
    index = []
    rows = defaultdict(list)
    replay_count = 0
    for kind, rate, seed in product(config["negative_controls"], config["sample_rates"], config["seed_ids"]):
        pair = generate_pair(seed, sample_rate=rate, target=False, nuisance=kind == "nuisance_only")
        replay_count += _same_pair(
            pair, generate_pair(seed, sample_rate=rate, target=False, nuisance=kind == "nuisance_only")
        )
        failures.update(gate_failures(pair, config))
        rows[f"{kind}/{rate}"].append(pair_metrics(pair))
        index.append(_save_pair(directory, f"{kind}_{rate}_{seed:03d}", pair))
    return {
        "count": len(index),
        "gate_failure_counts": dict(failures),
        "all_passed": not failures and replay_count == len(index),
        "deterministic_replays_passed": replay_count,
        "strata": {
            name: {"count": len(values), "metrics": _quantiles(values)} for name, values in rows.items()
        },
    }, index


def qualify_generator(output_directory: Path) -> dict[str, Any]:
    """Execute the committed complete grid and write a source-bound aggregate receipt.

    Parameters
    ----------
    output_directory : Path
        New repository-local directory; existing directories are never overwritten.

    Returns
    -------
    dict
        Qualification receipt, including failures if any. Per-example arrays/parameters stay local.
    """
    started = time.monotonic()
    config_path = ROOT / "configs/ssl_synthetic_generator_v1.json"
    config = json.loads(config_path.read_text())
    if config["support_floor_mv"] != SUPPORT_FLOOR_MV:
        raise ValueError("Configured support floor disagrees with implementation")
    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}
    output_directory.mkdir(parents=True, exist_ok=False)
    examples = output_directory / "examples"
    examples.mkdir()
    positives, index = _positive_grid(config, examples)
    negatives, negative_index = _negative_grid(config, examples)
    index.extend(negative_index)
    index_path = output_directory / "local_examples_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    invalid = _invalid_controls(config)
    endpoint = {
        profile: bool(
            np.array_equal(edit_profile(np.array([-0.1, 0, 1, 1.1]), profile), np.zeros(4))
            and np.max(edit_profile(np.array([1e-9, 1 - 1e-9]), profile)) / 1e-9 < 1e-5
        )
        for profile in config["profiles"]
    }
    thread_names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    thread_environment = {name: os.environ.get(name) for name in thread_names}
    single_thread = all(value == "1" for value in thread_environment.values())
    receipt = {
        "version": config["version"],
        "purpose": "pure simulated intervention truth; no model scores",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes,
        "configuration": config,
        "command": "PYTHONPATH=. .venv/bin/python -m scripts.validation.qualify_ssl_synthetic_generator",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": thread_environment,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "positive_grid": positives,
        "negative_grid": negatives,
        "expected_invalid_control_rejections": invalid,
        "zero_endpoint_and_derivative_checks": endpoint,
        "single_numerical_thread_environment": single_thread,
        "local_examples": {
            "count": len(index),
            "total_bytes": sum(row["bytes"] for row in index),
            "index_path": to_stored(index_path),
            "index_sha256": sha256_file(index_path),
        },
        "wall_seconds": time.monotonic() - started,
        "all_passed": positives["all_passed"]
        and negatives["all_passed"]
        and all(invalid.values())
        and all(endpoint.values())
        and single_thread,
        "interpretation": "Engineering truth only; no clinical validity, model scores or GPU admission.",
        "predecessor_integrity": "Independent generator; missing Experiment 058 artifacts; not reproduced.",
    }
    (output_directory / "qualification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
