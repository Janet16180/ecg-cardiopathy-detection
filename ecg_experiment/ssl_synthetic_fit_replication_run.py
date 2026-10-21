"""Run source-frozen independent-fit replication on old TRAIN and new generated backgrounds."""

from __future__ import annotations

import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_control_diagnosis as diagnosis
from ecg_experiment import ssl_synthetic_fit_replication as replication
from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_generated_quality_run as generator_runtime
from ecg_experiment import ssl_synthetic_pretrained_generator as public
from ecg_experiment import ssl_synthetic_randomized_localization as learning
from ecg_experiment import ssl_synthetic_randomized_localization_run as fit_runtime
from ecg_experiment.paths import from_stored, to_stored
from ecg_experiment.ssl_synthetic_fit_replication_figures import make_figures
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = "configs/ssl_synthetic_fit_replication_v1.json"
SOURCES = (
    CONFIG,
    "docs/ssl-synthetic-localization-fit-replication-v1.md",
    "ecg_experiment/ssl_synthetic_fit_replication.py",
    "ecg_experiment/ssl_synthetic_fit_replication_run.py",
    "ecg_experiment/ssl_synthetic_fit_replication_figures.py",
    "scripts/experiments/run_ssl_synthetic_fit_replication.py",
    "tests/test_ssl_synthetic_fit_replication.py",
    "ecg_experiment/ssl_synthetic_randomized_localization.py",
    "ecg_experiment/ssl_synthetic_randomized_localization_run.py",
    "ecg_experiment/ssl_synthetic_control_diagnosis.py",
    "ecg_experiment/ssl_synthetic_generated_quality.py",
    "ecg_experiment/ssl_synthetic_generated_quality_run.py",
    "ecg_experiment/ssl_synthetic_pretrained_generator.py",
    "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
    "ecg_experiment/synthetic_audit_io.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/gpu.py",
)
ARMS = ("supervised", "permuted")


def snapshot(destination: Path, guard: Any) -> dict[str, str]:
    """Save every authored executable dependency before opening admitted scientific payloads."""
    sources = {}
    for name in SOURCES:
        guard()
        path = destination / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(from_stored(name).read_bytes())
        sources[to_stored(from_stored(name))] = file_sha256(path, guard)
    return sources


def admit_parents(
    output_root: Path, config: dict[str, Any], inputs: dict[str, dict[str, str]], guard: Any
) -> dict[str, Path]:
    """Verify each explicit multi-study allowlist without globbing or recovering missing artifacts."""
    roots = {}
    for label, parent in config["parents"].items():
        root = relative_file(output_root, parent["root"])
        roots[label] = root
        inputs[label] = {}
        for name, expected in parent["sha256"].items():
            if file_sha256(relative_file(root, name), guard) != expected:
                raise ValueError(f"Frozen {label} parent identity mismatch: {name}")
            inputs[label][name] = expected
        audit = read_json(root / "independent_audit_v1/summary.json")
        if not audit["all_passed"] or audit["mismatch_count"]:
            raise ValueError(f"Admitted {label} independent audit did not pass")
    ledger = read_json(roots["diagnosis"] / config["prior_accounting"]["ledger"])
    if (
        ledger["current_machine_cumulative_cpu_seconds"]
        != config["prior_accounting"]["cumulative_cpu_seconds"]
        or ledger["current_machine_cumulative_gpu_wall_seconds"]
        != config["prior_accounting"]["cumulative_gpu_wall_seconds"]
    ):
        raise ValueError("Literal preceding cumulative accounting differs")
    return roots


def verify_bindings(
    destination: Path,
    roots: dict[str, Path],
    sources: dict[str, str],
    inputs: dict[str, dict[str, str]],
    generated: dict[str, str],
    guard: Any,
) -> None:
    """Rehash current/snapshot sources, admitted parents and newly immutable numerical inputs."""
    for name, expected in sources.items():
        if (
            file_sha256(from_stored(name), guard) != expected
            or file_sha256(destination / "source_snapshot" / name, guard) != expected
        ):
            raise ValueError("Replication source or source snapshot changed")
    for label, catalog in inputs.items():
        for name, expected in catalog.items():
            if file_sha256(relative_file(roots[label], name), guard) != expected:
                raise ValueError("Admitted parent bytes changed")
    for name, expected in generated.items():
        if file_sha256(relative_file(destination, name), guard) != expected:
            raise ValueError("New immutable replication inputs changed")


def predecessor_closures(roots: dict[str, Path], destination: Path, guard: Any) -> dict[str, Any]:
    """Reproduce old learning endpoints and the immediately preceding diagnostic primary exactly."""
    parent = roots["learning"]
    main = read_json(parent / "summary.json")
    if not main["all_passed"]:
        raise ValueError("Old randomized learning predecessor did not pass")
    observed = np.load(parent / "observed.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(parent / "support.npy", mmap_mode="r", allow_pickle=False)
    prior = np.load(parent / "spatial_prior.npy", allow_pickle=False)
    maps = {
        arm: np.load(parent / arm / "evaluation_probabilities.npy", mmap_mode="r", allow_pickle=False)
        for arm in ARMS
    }
    arrays = {"main": main, "observed": observed[80:], "support": support[80:], "prior": prior, "maps": maps}
    learning_closure = diagnosis.predecessor_closure(parent, arrays, destination, guard)
    for arm in ARMS:
        if not replication.verify_legacy_scoring(maps[arm], observed[80:], support[80:], prior):
            raise ValueError("Generic panel scoring differs from the frozen old 48-background helper")
    learning_closure["generic_panel_scoring_exact"] = True
    ranked_prior, _ = diagnosis.ranked_mask(prior)
    rows = []
    for offset, index in enumerate(range(80, 128)):
        guard()
        rows.append(
            diagnosis.diagnose_case(
                {arm: value[offset] for arm, value in maps.items()},
                observed[index],
                support[index],
                ranked_prior,
                index,
            )
        )
    primary = diagnosis.summarize(rows)["primary"]
    compare_values(
        primary,
        read_json(roots["diagnosis"] / "summary.json")["primary"],
        "immediate diagnostic primary closure",
        tolerance=0,
    )
    public.write_new_json(
        destination / "diagnostic_primary_closure.json", {"all_exact": True, "primary": primary}
    )
    return {
        "observed": observed,
        "support": support,
        "prior": prior,
        "old_main": main,
        "learning_closure": learning_closure,
        "diagnostic_primary": primary,
    }


def load_public_generator(
    root: Path, destination: Path, counts: dict[str, int], guard: Any
) -> tuple[Any, str]:
    """Strictly load the existing public model and reproduce all four original CPU generations."""
    model = public.load_generator(
        root / "pytorch_model.bin",
        public.architecture_class(
            (root / "mit_pulse2pulse.py").read_text(), (root / "hf_modeling_deepfake.py").read_text()
        ),
    )
    initial = public.tensor_state_sha256(model)
    if initial != read_json(root / "summary.json")["model_state_sha256"]:
        raise ValueError("Public model named tensor identity differs")
    noise = np.load(root / "noise.npy", allow_pickle=False)
    cases = []
    for index in range(4):
        guard()
        counts["cpu_generator_attempts"] += 1
        actual = public.generate(model, noise[index])
        counts["cpu_generator_forwards"] += 1
        checks = {
            name: bool(np.array_equal(value, np.load(root / f"case_{index}_{name}.npy", allow_pickle=False)))
            for name, value in zip(["raw", "author_export", "canonical12"], actual, strict=True)
        }
        cases.append(checks)
    closure = {
        "all_exact": all(all(row.values()) for row in cases),
        "cases": cases,
        "forwards": 4,
        "model_state_sha256": initial,
    }
    public.write_new_json(destination / "public_generator_closure.json", closure)
    if not closure["all_exact"]:
        raise ValueError("Original public generator CPU closure failed")
    return model, initial


def save_generation_case(directory: Path, prefix: str, raw: np.ndarray, index: int) -> tuple[Any, ...]:
    """Retain generated raw/derived/mask arrays and metadata before qualifying a profile case."""
    quality.save_array(directory / f"{prefix}_raw.npy", raw)
    observed, strict, support, row = learning.make_case(raw, index)
    for name, value in [("observed", observed), ("strict_support", strict), ("support", support)]:
        quality.save_array(directory / f"{prefix}_{name}.npy", value)
    public.write_new_json(directory / f"{prefix}_case.json", row)
    qualify_case(raw, observed, strict, support, row)
    return raw, observed, strict, support, row


def qualify_case(
    raw: np.ndarray, observed: np.ndarray, strict: np.ndarray, support: np.ndarray, row: dict[str, Any]
) -> None:
    """Enforce the unchanged numerical support, finite, sham and limb closure requirements."""
    if raw.dtype != np.float32 or raw.shape != (8, 5000) or not np.all(np.ptp(raw, axis=1) > 0):
        raise ValueError("Fresh native geometry or temporal variation failed")
    if not np.array_equal(observed[0], quality.derive_leads(quality.standardize(raw)[0])):
        raise ValueError("Fresh sham closure failed")
    if not np.all(np.ptp(observed[0], axis=1) > 0):
        raise ValueError("Fresh standardized canonical sham contains a flat lead")
    if not np.isfinite(observed).all() or support[:2].any():
        raise ValueError("Fresh observed/control geometry failed")
    if not all(
        np.allclose(case, quality.derive_leads(case[[0, 1, 6, 7, 8, 9, 10, 11]]), atol=2e-6, rtol=1e-6)
        for case in observed
    ):
        raise ValueError("Fresh canonical limb equations failed")
    for check in row["checks"].values():
        if not (
            check["strict_mask_exact"]
            and check["threshold_mask_exact"]
            and check["nonempty"]
            and check["threshold_subset_strict"]
            and check["outside_hull_cells"] == 0
            and check["retained_energy_fraction"] >= 0.99
        ):
            raise ValueError("Fresh intervention support integrity or energy retention failed")
    if strict.shape != (2, 12, 5000) or support.dtype != np.bool_:
        raise ValueError("Fresh strict/target support geometry failed")


def profile_generation(
    model: torch.nn.Module,
    noise: np.ndarray,
    indices: np.ndarray,
    destination: Path,
    counts: dict[str, int],
    guard: Any,
) -> dict[str, Any]:
    """Profile three complete fresh generation paths and save one exact first-input replay."""
    directory = destination / "generation_profile"
    directory.mkdir()
    model.to("cuda:0")
    initial_state = public.tensor_state_sha256(model)
    timings = []
    for offset in range(3):
        guard()
        started = {"cpu": time.process_time(), "wall": time.perf_counter()}
        counts["gpu_generator_profile_attempts"] += 1
        raw = generator_runtime.gpu_forward(model, noise[offset])
        counts["gpu_generator_profile_forwards"] += 1
        save_generation_case(directory, f"original_{offset}", raw, int(indices[offset]))
        timings.append(quality.elapsed(started))
    guard()
    started = {"cpu": time.process_time(), "wall": time.perf_counter()}
    counts["gpu_generator_profile_attempts"] += 1
    raw = generator_runtime.gpu_forward(model, noise[0])
    counts["gpu_generator_profile_forwards"] += 1
    replay = save_generation_case(directory, "replay_0", raw, int(indices[0]))
    replay_timing = quality.elapsed(started)
    exact = all(
        np.array_equal(value, np.load(directory / f"original_0_{name}.npy", allow_pickle=False))
        for name, value in zip(["raw", "observed", "strict_support", "support"], replay[:4], strict=True)
    )
    final_state = public.tensor_state_sha256(model)
    result = {
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "model_state_unchanged": initial_state == final_state,
        "original_timings": timings,
        "replay_timing": replay_timing,
        "first_input_exact_replay": exact,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
    }
    public.write_new_json(destination / "generation_profile.json", result)
    if not exact or initial_state != final_state:
        raise ValueError("Fresh generator profile replay or state identity differs")
    return result


def prepare_fit_plans(config: dict[str, Any], parent: Path, destination: Path) -> list[dict[str, Any]]:
    """Recreate the old initialization/schedule and save all prospectively fixed independent plans."""
    plans = []
    for specification in [config["training"]["old_fit"], *config["training"]["new_fits"]]:
        directory = destination / f"fit_{specification['fit_id']}"
        directory.mkdir()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(specification["initialization_seed"])
            model = learning.build_model()
        initial = fit_runtime.cpu_tree(model.state_dict())
        schedule = replication.training_schedule(specification["schedule_seed"])
        if specification["fit_id"] == 0:
            expected = torch.load(parent / "initial_state.pt", map_location="cpu", weights_only=True)
            if not fit_runtime.trees_equal(initial, expected) or not np.array_equal(
                schedule, np.load(parent / "training_schedule.npy", allow_pickle=False)
            ):
                raise ValueError("Recreated old initialization or schedule differs")
        fit_runtime.save_state(directory / "initial_state.pt", initial)
        quality.save_array(directory / "training_schedule.npy", schedule)
        quality.save_array(directory / "background_schedule.npy", schedule[:, [0, 4]] // 4)
        public.write_new_json(directory / "fit_design.json", specification)
        plans.append(
            {
                **specification,
                "directory": directory,
                "initial": initial,
                "schedule": schedule,
                "initial_state_sha256": public.tensor_state_sha256(model),
            }
        )
    return plans


def combined_profile(
    generation: dict[str, Any],
    training: dict[str, Any],
    destination: Path,
    clock: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Admit all remaining generation, replay, fits and panel inference with one conservative forecast."""
    spent = quality.elapsed(clock)
    lock_spent = time.perf_counter() - clock["gpu_lock_start"]
    result = {
        "generation": generation,
        "training": training,
        "spent": spent,
        "spent_gpu_lock_wall_seconds": lock_spent,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
    }
    for kind in ["cpu", "wall"]:
        g = max(row[f"{kind}_seconds"] for row in generation["original_timings"][1:])
        r = max(row["timings"][index][f"{kind}_seconds"] for row in training.values() for index in [1, 2])
        e = max(row["inference"][f"{kind}_seconds"] for row in training.values())
        result[f"recurrent_{kind}_seconds"] = {"generation": g, "update": r, "inference": e}
        future = 2 * 64 * g + 2 * 3200 * r + 2 * 400 * e + 90
        result[f"forecast_{kind}_seconds"] = spent[f"{kind}_seconds"] + future
        if kind == "wall":
            result["forecast_gpu_lock_wall_seconds"] = lock_spent + future
    map_bytes = (8 * 64 + 6 * 48) * 4 * 12 * 5000 * 4
    fresh_bytes = 64 * (8 * 5000 * 4 + 4 * 12 * 5000 * 4 + 2 * 12 * 5000 + 4 * 12 * 5000)
    result["forecast_saved_bytes"] = (
        public.directory_bytes(destination)
        + map_bytes
        + fresh_bytes
        + 16 * 1024**2
        + config["resources"]["output_forecast_reserve_bytes"]
    )
    public.write_new_json(destination / "combined_profile.json", result)
    if (
        result["forecast_cpu_seconds"] >= 1800
        or result["forecast_wall_seconds"] >= 1800
        or result["forecast_gpu_lock_wall_seconds"] >= 1200
        or result["peak_allocated_bytes"] >= 2 * 1024**3
        or result["forecast_saved_bytes"] >= config["resources"]["output_bytes"]
    ):
        raise ValueError("Combined generation/training/evaluation forecast fails")
    return result


def generate_fresh_bank(
    model: torch.nn.Module,
    noise: np.ndarray,
    config: dict[str, Any],
    destination: Path,
    old_shams: np.ndarray,
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    guard: Any,
) -> dict[str, np.ndarray]:
    """Generate all 64 predeclared cases, retaining failures and exact profile/bank closure."""
    banks = {}
    for name, shape, dtype in [
        ("raw", (64, 8, 5000), np.float32),
        ("observed", (64, 4, 12, 5000), np.float32),
        ("support", (64, 4, 12, 5000), np.bool_),
        ("strict_support", (64, 2, 12, 5000), np.bool_),
    ]:
        path = destination / f"fresh_{name}.npy"
        if path.exists():
            raise FileExistsError(path)
        banks[name] = np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
    old_hashes = {public.array_sha256(value) for value in old_shams}
    hashes, raw_hashes = set(), set()
    for offset, index in enumerate(config["fresh_panel"]["global_indices"]):
        guard()
        counts["gpu_generator_bank_attempts"] += 1
        raw = generator_runtime.gpu_forward(model, noise[offset])
        counts["gpu_generator_bank_forwards"] += 1
        banks["raw"][offset] = raw
        banks["raw"].flush()
        observed, strict, support, row = learning.make_case(raw, index)
        for name, value in [("observed", observed), ("support", support), ("strict_support", strict)]:
            banks[name][offset] = value
            banks[name].flush()
        rows.append(row)
        qualify_case(raw, observed, strict, support, row)
        sham_hash, raw_hash = public.array_sha256(observed[0]), public.array_sha256(raw)
        if sham_hash in hashes or sham_hash in old_hashes or raw_hash in raw_hashes:
            raise ValueError("Fresh raw/sham exact duplicate guard failed; no replacement allowed")
        hashes.add(sham_hash)
        raw_hashes.add(raw_hash)
        if offset < 3:
            for name, value in [
                ("raw", raw),
                ("observed", observed),
                ("support", support),
                ("strict_support", strict),
            ]:
                if not np.array_equal(
                    value,
                    np.load(
                        destination / "generation_profile" / f"original_{offset}_{name}.npy",
                        allow_pickle=False,
                    ),
                ):
                    raise ValueError("Fresh bank first-three profile closure failed")
    quality.write_rows(destination / "fresh_cases.jsonl", rows)
    public.write_new_json(
        destination / "fresh_uniqueness.json",
        {
            "fresh_sham_sha256": sorted(hashes),
            "fresh_raw_sha256": sorted(raw_hashes),
            "old_sham_count": len(old_hashes),
            "fresh_count": 64,
            "disjoint": True,
            "first_three_profile_exact": True,
        },
    )
    return banks


def exact_old_fit_replay(
    parent: Path, plan: dict[str, Any], models: dict[str, torch.nn.Module]
) -> dict[str, Any]:
    """Require all old 400 losses and complete final model/Adam values to reproduce exactly."""
    checks = {}
    for arm in ARMS:
        directory = plan["directory"] / arm
        losses = np.load(directory / "losses.npy", allow_pickle=False)
        previous = np.load(parent / arm / "losses.npy", allow_pickle=False)
        model_state = torch.load(parent / arm / "final_state.pt", map_location="cpu", weights_only=True)
        optimizer = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
        previous_optimizer = torch.load(parent / arm / "optimizer.pt", map_location="cpu", weights_only=True)
        checks[arm] = {
            "all_400_losses_exact": bool(np.array_equal(losses, previous)),
            "final_model_exact": fit_runtime.trees_equal(
                fit_runtime.cpu_tree(models[arm].state_dict()), model_state
            ),
            "complete_adam_exact": fit_runtime.trees_equal(optimizer, previous_optimizer),
        }
    result = {"all_exact": all(all(row.values()) for row in checks.values()), "arms": checks}
    public.write_new_json(plan["directory"] / "old_training_closure.json", result)
    if not result["all_exact"]:
        raise ValueError("Exact old 400-update model/Adam/loss replay failed")
    return result


def fit_counts() -> dict[str, int]:
    """Initialize exact attempted/completed per-arm updates and both evaluation-panel counts."""
    result = {}
    for arm in ARMS:
        for suffix in [
            "update_attempts",
            "updates",
            "fresh_inference_attempts",
            "fresh_inferences",
            "spent_inference_attempts",
            "spent_inferences",
        ]:
            result[f"{arm}_{suffix}"] = 0
    return result


def evaluate_panel(
    model: torch.nn.Module,
    observed: np.ndarray,
    path: Path,
    panel: str,
    arm: str,
    counts: dict[str, int],
    guard: Any,
) -> None:
    """Persist every model map for a complete prospective panel in batch-eight order."""
    count = observed.shape[0]
    if count not in {48, 64} or observed.shape[1:] != (4, 12, 5000):
        raise ValueError("Panel inference geometry differs")
    if path.exists():
        raise FileExistsError(path)
    probability = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(count, 4, 12, 5000))
    flattened = probability.reshape(count * 4, 12, 5000)
    values = observed.reshape(count * 4, 12, 5000)
    for start in range(0, count * 4, 8):
        guard()
        counts[f"{arm}_{panel}_inference_attempts"] += 1
        predicted = fit_runtime.infer(model, values[start : start + 8], "cuda:0")
        counts[f"{arm}_{panel}_inferences"] += 1
        flattened[start : start + 8] = predicted
        probability.flush()
        if not np.isfinite(predicted).all() or not ((predicted >= 0) & (predicted <= 1)).all():
            raise ValueError("Panel probability geometry or finite range failed")


def train_and_evaluate(
    plans: list[dict[str, Any]],
    old: dict[str, Any],
    fresh: dict[str, np.ndarray],
    roots: dict[str, Path],
    targets: dict[str, np.ndarray],
    weight: float,
    counts: dict[str, Any],
    guard: Any,
    verify: Any,
) -> dict[str, Any]:
    """Exactly replay fit0, train all new fits, then infer every predeclared panel without selection."""
    models, hashes = {}, {}
    train_observed = old["observed"][:80].reshape(320, 12, 5000)
    old_closure = {}
    for plan in plans:
        identifier = str(plan["fit_id"])
        verify()
        models[identifier] = fit_runtime.train_pair(
            plan["initial"],
            train_observed,
            targets,
            plan["schedule"],
            weight,
            plan["directory"],
            counts["fits"][identifier],
            guard,
        )
        verify()
        if plan["fit_id"] == 0:
            old_closure = exact_old_fit_replay(roots["learning"], plan, models[identifier])
    for plan in plans:
        identifier = str(plan["fit_id"])
        hashes[identifier] = {}
        for arm, model in models[identifier].items():
            evaluate_panel(
                model,
                fresh["observed"],
                plan["directory"] / arm / "fresh_probabilities.npy",
                "fresh",
                arm,
                counts["fits"][identifier],
                guard,
            )
            if plan["fit_id"] != 0:
                evaluate_panel(
                    model,
                    old["observed"][80:],
                    plan["directory"] / arm / "spent_probabilities.npy",
                    "spent",
                    arm,
                    counts["fits"][identifier],
                    guard,
                )
            hashes[identifier][arm] = public.tensor_state_sha256(model)
            model.cpu()
    return {"old_training_replay": old_closure, "final_state_sha256": hashes}


def score_outputs(
    plans: list[dict[str, Any]],
    old: dict[str, Any],
    fresh: dict[str, np.ndarray],
    config: dict[str, Any],
    guard: Any,
) -> dict[str, Any]:
    """Score all saved panels, keeping fit0 and the spent panel outside the fresh replication primary."""
    fresh_fits, conditional, spent_conditional, absolute = [], {}, {}, {}
    for plan in plans:
        fit = {"fit_id": plan["fit_id"]}
        spent = {}
        identifier = str(plan["fit_id"])
        absolute[identifier] = {}
        for panel in ["fresh", "spent"]:
            if panel == "spent" and plan["fit_id"] == 0:
                continue
            observed = fresh["observed"] if panel == "fresh" else old["observed"][80:]
            support = fresh["support"] if panel == "fresh" else old["support"][80:]
            indices = (
                np.array(config["fresh_panel"]["global_indices"]) if panel == "fresh" else np.arange(80, 128)
            )
            seeds = (
                np.array(config["fresh_panel"]["generator_seeds"])
                if panel == "fresh"
                else np.arange(2026103080, 2026103128)
            )
            absolute[identifier][panel] = {}
            for arm in ARMS:
                guard()
                probability = np.load(
                    plan["directory"] / arm / f"{panel}_probabilities.npy", mmap_mode="r", allow_pickle=False
                )
                rows = replication.score_panel(probability, observed, support, indices, seeds, old["prior"])
                metrics = {
                    "cases": rows,
                    "controls": replication.control_scores(probability),
                    "hotspots": replication.hotspot_scores(probability, support, indices, seeds),
                }
                public.write_new_json(plan["directory"] / arm / f"{panel}_metrics.json", metrics)
                absolute[identifier][panel][arm] = {
                    mode: {key: float(np.mean([row[mode][key] for row in rows])) for key in rows[0][mode]}
                    for mode in ["focal", "repeated"]
                }
                if panel == "fresh":
                    fit[arm] = rows
                else:
                    spent[arm] = rows
            if panel == "spent":
                spent_conditional[identifier] = learning.summarize_metrics(
                    spent["supervised"], spent["permuted"]
                )["contrasts"]
        conditional[identifier] = replication.conditional_fresh_summary(fit)
        if plan["fit_id"] != 0:
            fresh_fits.append(fit)
    return {
        "aggregate": replication.aggregate_replication(fresh_fits),
        "conditional_fresh_contrasts": conditional,
        "conditional_spent_contrasts": spent_conditional,
        "absolute_mode_means": absolute,
        "fit0_spent_reference_contrasts": old["old_main"]["contrasts"],
        "fit0_spent_maps_reused_without_inference": True,
    }


def finish_ledger(
    destination: Path,
    roots: dict[str, Path],
    config: dict[str, Any],
    receipt: str,
    clock: dict[str, float],
    gpu_cost: float,
) -> None:
    """Record terminal own cost once after the literal latest audited cumulative ledger."""
    digest = file_sha256(destination / receipt)
    prior_spec = config["prior_accounting"]
    expected = config["parents"][prior_spec["parent"]]["sha256"][prior_spec["ledger"]]
    parent_root = roots.get(prior_spec["parent"])
    path = parent_root / prior_spec["ledger"] if parent_root is not None else None
    prior_digest = file_sha256(path) if path is not None and path.is_file() else None
    prior = read_json(path) if prior_digest == expected else {}
    spent = quality.elapsed(clock)
    cpu, gpu = (
        prior.get("current_machine_cumulative_cpu_seconds"),
        prior.get("current_machine_cumulative_gpu_wall_seconds"),
    )
    public.write_new_json(
        destination / "prospective_ledger.json",
        {
            "stage": "generated_fit_replication_v1",
            "receipt": receipt,
            "receipt_sha256": digest,
            **spent,
            "gpu_wall_seconds": gpu_cost,
            "prior_ledger_sha256": prior_digest,
            "prior_accounting_verified": prior_digest == expected,
            "prior_current_machine_cpu_seconds": cpu,
            "prior_current_machine_gpu_wall_seconds": gpu,
            "current_machine_cumulative_cpu_seconds": cpu + spent["cpu_seconds"] if cpu is not None else None,
            "current_machine_cumulative_gpu_wall_seconds": gpu + gpu_cost if gpu is not None else None,
            "cpu_budget_passed": spent["cpu_seconds"] < 1800,
            "wall_budget_passed": spent["wall_seconds"] < 1800,
            "gpu_budget_passed": gpu_cost < 1200,
            "output_bytes_before_terminal_ledger": public.directory_bytes(destination),
            "historical_costs_verified": False,
            "measurement_endpoint": "after receipt write/hash, before terminal ledger serialization",
            "reserved_independent_audit": {"cpu_seconds": 300, "wall_seconds": 300, "bytes": 32 * 1024**2},
        },
    )


def initial_counts() -> dict[str, Any]:
    """Initialize every planned generator, profile, training and panel inference counter."""
    result = dict.fromkeys(
        [
            "cpu_generator_attempts",
            "cpu_generator_forwards",
            "gpu_generator_profile_attempts",
            "gpu_generator_profile_forwards",
            "gpu_generator_bank_attempts",
            "gpu_generator_bank_forwards",
        ],
        0,
    )
    result["training_profile"] = dict.fromkeys(
        ["profile_update_attempts", "profile_updates", "profile_inference_attempts", "profile_inferences"], 0
    )
    result["fits"] = {str(index): fit_counts() for index in range(4)}
    return result


def operation_totals(counts: dict[str, Any]) -> dict[str, int]:
    """Summarize actual completed operations without counting nested cumulative values again."""
    fits = counts["fits"]
    return {
        "cpu_generator_forwards": counts["cpu_generator_forwards"],
        "gpu_generator_forwards": counts["gpu_generator_profile_forwards"]
        + counts["gpu_generator_bank_forwards"],
        "profile_optimizer_updates": counts["training_profile"]["profile_updates"],
        "old_replay_optimizer_updates": sum(fits["0"][arm + "_updates"] for arm in ARMS),
        "new_fit_optimizer_updates": sum(
            fits[str(index)][arm + "_updates"] for index in [1, 2, 3] for arm in ARMS
        ),
        "profile_localizer_inference_batches": counts["training_profile"]["profile_inferences"],
        "fresh_localizer_inference_batches": sum(
            row[arm + "_fresh_inferences"] for row in fits.values() for arm in ARMS
        ),
        "spent_localizer_inference_batches": sum(
            row[arm + "_spent_inferences"] for row in fits.values() for arm in ARMS
        ),
        "model_selection_updates": 0,
    }


def execute_gpu(
    public_model: torch.nn.Module,
    public_initial: str,
    plans: list[dict[str, Any]],
    old: dict[str, Any],
    roots: dict[str, Path],
    config: dict[str, Any],
    destination: Path,
    clock: dict[str, float],
    counts: dict[str, Any],
    rows: list[dict[str, Any]],
    generated: dict[str, str],
    guard: Any,
    verify: Any,
) -> dict[str, Any]:
    """Profile the full path, generate the fixed bank, exactly replay fit0 and execute all new fits."""
    noise = np.stack(
        [
            torch.empty((1, 8, 5000), dtype=torch.float32)
            .uniform_(-1, 1, generator=torch.Generator(device="cpu").manual_seed(seed))
            .numpy()
            for seed in config["fresh_panel"]["generator_seeds"]
        ]
    )
    quality.save_array(destination / "fresh_noise.npy", noise)
    generated["fresh_noise.npy"] = file_sha256(destination / "fresh_noise.npy", guard)
    verify()
    train_observed = old["observed"][:80].reshape(320, 12, 5000)
    targets = {arm: learning.training_targets(old["support"][:80], arm) for arm in ARMS}
    weight = learning.positive_weight(old["support"][:80])
    donor = (np.arange(80) + 1) % 80
    if not np.array_equal(donor, np.load(roots["learning"] / "donor_map.npy", allow_pickle=False)):
        raise ValueError("Unchanged donor mapping differs")
    design = read_json(roots["learning"] / "training_design.json")
    positives = int(np.count_nonzero(old["support"][:80]))
    if (
        weight != design["positive_weight"]
        or positives != design["positive_cells"]
        or np.count_nonzero(targets["supervised"]) != np.count_nonzero(targets["permuted"])
    ):
        raise ValueError("Unchanged training target counts or class weight differs")
    public.write_new_json(destination / "training_design.json", design)
    quality.save_array(destination / "donor_map.npy", donor)
    with generator_runtime.measured_gpu_lock(clock):
        clock["gpu_lock_start"] = time.perf_counter()
        free, total = torch.cuda.mem_get_info(0)
        if free < config["resources"]["minimum_free_gpu_bytes"]:
            raise ValueError("Replication GPU admission has insufficient free memory")
        torch.cuda.reset_peak_memory_stats(0)
        generation_profile = profile_generation(
            public_model, noise, np.array(config["fresh_panel"]["global_indices"]), destination, counts, guard
        )
        training_profile = {
            arm: fit_runtime.profile_arm(
                arm,
                plans[0]["initial"],
                train_observed,
                targets[arm],
                plans[0]["schedule"],
                weight,
                plans[0]["directory"],
                counts["training_profile"],
                guard,
            )
            for arm in ARMS
        }
        profile = combined_profile(generation_profile, training_profile, destination, clock, config)
        verify()
        fresh = generate_fresh_bank(
            public_model, noise, config, destination, old["observed"][:, 0], rows, counts, guard
        )
        public_model.cpu()
        public_final = public.tensor_state_sha256(public_model)
        if public_final != public_initial or generation_profile["initial_state_sha256"] != public_initial:
            raise ValueError("Public generator state changed across CPU/profile/bank execution")
        public.write_new_json(
            destination / "generator_state_identity.json",
            {"initial_state_sha256": public_initial, "final_state_sha256": public_final, "unchanged": True},
        )
        for name in [
            "fresh_raw.npy",
            "fresh_observed.npy",
            "fresh_support.npy",
            "fresh_strict_support.npy",
            "fresh_cases.jsonl",
            "donor_map.npy",
            "training_design.json",
        ]:
            generated[name] = file_sha256(destination / name, guard)
        verify()
        trained = train_and_evaluate(plans, old, fresh, roots, targets, weight, counts, guard, verify)
        metadata = {
            "name": torch.cuda.get_device_name(0),
            "driver": generator_runtime.subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).splitlines()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "free_bytes_at_admission": free,
            "total_bytes": total,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        }
    return {
        "fresh": fresh,
        "profile": profile,
        "trained": trained,
        "gpu_metadata": metadata,
        "positive_weight": weight,
    }


def plan_hashes(plans: list[dict[str, Any]], destination: Path, guard: Any) -> dict[str, str]:
    """Bind every serialized initialization and schedule before any scientific profile or fit."""
    hashes = {}
    for plan in plans:
        for filename in [
            "initial_state.pt",
            "training_schedule.npy",
            "background_schedule.npy",
            "fit_design.json",
        ]:
            path = plan["directory"] / filename
            hashes[path.relative_to(destination).as_posix()] = file_sha256(path, guard)
    return hashes


def run_study(output_root: Path) -> dict[str, Any]:
    """Run the committed replication once, preserving all numerical failures and exact operation costs.

    Parameters
    ----------
    output_root : Path
        Local repository root hosting all explicitly admitted old and new outputs.

    Returns
    -------
    dict
        Executed crossed replication verdict, integrity checks, all panel statistics and receipts.
    """
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    config = read_json(from_stored(CONFIG))
    destination = output_root / config["output_root"]
    destination.mkdir(exist_ok=False)
    counts, rows, sources, inputs, generated, roots = initial_counts(), [], {}, {}, {}, {}
    handlers = {}

    def guard() -> None:
        spent = quality.elapsed(clock)
        if spent["cpu_seconds"] + 2 >= 1800 or spent["wall_seconds"] + 2 >= 1800:
            raise TimeoutError("Replication whole-stage CPU/wall deadline reached")
        if (
            "gpu_lock_start" in clock
            and "gpu_cost" not in clock
            and time.perf_counter() - clock["gpu_lock_start"] + 2 >= 1200
        ):
            raise TimeoutError("Replication shared GPU-lock wall deadline reached")
        if (
            "gpu_lock_start" in clock
            and "gpu_cost" not in clock
            and torch.cuda.max_memory_allocated(0) >= config["resources"]["maximum_gpu_allocated_bytes"]
        ):
            raise ValueError("Replication live GPU allocation ceiling reached")
        if public.directory_bytes(destination) + 1024**2 >= config["resources"]["output_bytes"]:
            raise ValueError("Replication terminal receipt reserve exhausted")

    try:
        generator_runtime.set_runtime()
        generator_runtime.arm_timers(config["resources"], handlers)
        sources = snapshot(destination, guard)
        roots = admit_parents(output_root, config, inputs, guard)
        verify_bindings(destination, roots, sources, inputs, generated, guard)
        old = predecessor_closures(roots, destination, guard)
        public_model, public_initial = load_public_generator(roots["generator"], destination, counts, guard)
        plans = prepare_fit_plans(config, roots["learning"], destination)
        generated.update(plan_hashes(plans, destination, guard))

        def verify() -> None:
            verify_bindings(destination, roots, sources, inputs, generated, guard)

        executed = execute_gpu(
            public_model,
            public_initial,
            plans,
            old,
            roots,
            config,
            destination,
            clock,
            counts,
            rows,
            generated,
            guard,
            verify,
        )
        fresh = executed.pop("fresh")
        verify()
        statistics = score_outputs(plans, old, fresh, config, guard)
        figure_maps = {
            plan["fit_id"]: {
                arm: np.load(
                    plan["directory"] / arm / "fresh_probabilities.npy", mmap_mode="r", allow_pickle=False
                )
                for arm in ARMS
            }
            for plan in plans
        }
        figure_metrics = {
            plan["fit_id"]: {arm: read_json(plan["directory"] / arm / "fresh_metrics.json") for arm in ARMS}
            for plan in plans
        }
        figures = make_figures(
            destination,
            fresh["observed"],
            fresh["support"],
            figure_maps,
            figure_metrics,
            statistics["aggregate"],
        )
        verify()
        totals = operation_totals(counts)
        integrity = {
            "old_learning_metrics_exact": old["learning_closure"]["all_exact"],
            "immediate_diagnosis_primary_exact": True,
            "old_public_cpu_generation_exact": counts["cpu_generator_forwards"] == 4,
            "generator_profile_replay_exact": executed["profile"]["generation"]["first_input_exact_replay"],
            "generator_state_unchanged": True,
            "fresh_64_qualified_unique_and_disjoint": len(rows) == 64,
            "bank_first_three_profile_exact": True,
            "training_profile_recovery_exact": all(
                row["exact_replay"] for row in executed["profile"]["training"].values()
            ),
            "old_400_step_pair_exact": executed["trained"]["old_training_replay"]["all_exact"],
            "all_operations_exact": totals == config["allowed_operations"],
            "resources": clock["gpu_cost"] < 1200
            and executed["gpu_metadata"]["peak_allocated_bytes"] < 2 * 1024**3,
        }
        guard()
        result = {
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "output_relative": to_stored(ROOT / config["output_root"]),
            **statistics,
            "integrity_gates": integrity,
            "all_passed": statistics["aggregate"]["scientific_passed"] and all(integrity.values()),
            "source_sha256": sources,
            "input_sha256": inputs,
            "generated_input_sha256": generated,
            "parent_roots": {label: row["root"] for label, row in config["parents"].items()},
            "output_sha256": {
                path.relative_to(destination).as_posix(): file_sha256(path, guard)
                for path in destination.rglob("*")
                if path.is_file()
            },
            "counts": counts,
            "operation_totals": totals,
            "fit_designs": [
                {
                    key: plan[key]
                    for key in ["fit_id", "initialization_seed", "schedule_seed", "initial_state_sha256"]
                }
                for plan in plans
            ],
            **executed,
            "figures": figures,
            "completed_fresh_indices": [row["index"] for row in rows],
            "gpu_lock_wall_seconds": clock["gpu_cost"],
            **quality.elapsed(clock),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "output_bytes_before_receipt": public.directory_bytes(destination),
            "runtime_environment": {
                key: os.environ.get(key)
                for key in [
                    "CUDA_VISIBLE_DEVICES",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]
            },
            "reserved_successor_backgrounds_generated": 0,
            "model_selection_updates": 0,
            "real_waveform_reads": 0,
            "external_api_calls": 0,
            "clinical_claim": False,
            "physical_units_verified": False,
        }
        guard()
        public.write_new_json(destination / "summary.json", result)
        guard()
        finish_ledger(destination, roots, config, "summary.json", clock, clock["gpu_cost"])
        return result
    except Exception as error:
        if rows and not (destination / "partial_fresh_cases.jsonl").exists():
            quality.write_rows(destination / "partial_fresh_cases.jsonl", rows)
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "counts": counts,
            "operation_totals": operation_totals(counts),
            "completed_or_offending_fresh_case_rows": len(rows),
            "source_sha256": sources,
            "input_sha256": inputs,
            "generated_input_sha256": generated,
            "gpu_lock_wall_seconds": clock.get("gpu_cost", 0.0),
            **quality.elapsed(clock),
            "historical_costs_verified": False,
            "reserved_successor_backgrounds_generated": 0,
        }
        public.write_new_json(destination / "failure.json", failure)
        finish_ledger(destination, roots, config, "failure.json", clock, clock.get("gpu_cost", 0.0))
        raise
    finally:
        fit_runtime.restore_timers(handlers)
