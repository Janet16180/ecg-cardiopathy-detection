"""Execute a bounded randomized-support learning study on saved generated backgrounds."""

from __future__ import annotations

import copy
import json
import resource
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as functional

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_generated_quality_run as runtime
from ecg_experiment import ssl_synthetic_pretrained_generator as shared
from ecg_experiment import ssl_synthetic_randomized_localization as learning

CONFIG = "configs/ssl_synthetic_randomized_localization_v1.json"
SOURCES = (
    CONFIG,
    "docs/ssl-synthetic-localization-randomized-learning-v1.md",
    "ecg_experiment/ssl_synthetic_randomized_localization.py",
    "ecg_experiment/ssl_synthetic_randomized_localization_run.py",
    "scripts/experiments/run_ssl_synthetic_randomized_localization.py",
    "tests/test_ssl_synthetic_randomized_localization.py",
    "ecg_experiment/ssl_synthetic_generated_quality.py",
    "ecg_experiment/ssl_synthetic_generated_quality_run.py",
    "ecg_experiment/ssl_synthetic_pretrained_generator.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/paths.py",
)


def cpu_tree(value: Any) -> Any:
    """Copy a model/optimizer state to independent CPU tensors for immutable snapshots."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def trees_equal(first: Any, second: Any) -> bool:
    """Compare replayed model and optimizer states including all scalar/tensor members."""
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and torch.equal(first, second.cpu())
        )
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            trees_equal(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            trees_equal(a, b) for a, b in zip(first, second, strict=True)
        )
    return first == second


def save_state(path: Path, value: Any) -> None:
    """Save a local tensor state while refusing any overwrite."""
    with path.open("xb") as stream:
        torch.save(cpu_tree(value), stream)


def optimizer_for(model: torch.nn.Module) -> torch.optim.Adam:
    """Use the complete frozen Adam specification for either matched arm."""
    return torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )


def update(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    observed: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    weight: float,
    device: str,
) -> float:
    """Run the complete observed-only minibatch transfer, loss and optimizer path once.

    Parameters
    ----------
    model : torch.nn.Module
        Observed-only network.
    optimizer : torch.optim.Adam
        Arm-specific optimizer without shared moments.
    observed, targets : np.ndarray
        Training waveform bank and separate supervision bank.
    indices : np.ndarray
        Eight scheduled case indices.
    weight : float
        Training-only positive weight.
    device : str
        Admitted execution device.

    Returns
    -------
    float
        Full-grid weighted BCE before this update.
    """
    model.train()
    inputs = torch.from_numpy(np.array(observed[indices], dtype=np.float32)).to(device)
    truth = torch.from_numpy(np.array(targets[indices], dtype=np.float32)).to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = functional.binary_cross_entropy_with_logits(
        logits, truth, pos_weight=torch.tensor(weight, device=device), reduction="mean"
    )
    loss.backward()
    optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize(0)
    return float(loss.detach().cpu())


def infer(model: torch.nn.Module, observed: np.ndarray, device: str) -> np.ndarray:
    """Return float32 sigmoid maps with only observed ECG values as model inputs."""
    model.eval()
    with torch.inference_mode():
        result = model(torch.from_numpy(np.array(observed, dtype=np.float32)).to(device)).sigmoid()
    if device.startswith("cuda"):
        torch.cuda.synchronize(0)
    return result.cpu().numpy().copy()


def profile_arm(
    arm: str,
    initial: dict[str, Any],
    observed: np.ndarray,
    targets: np.ndarray,
    schedule: np.ndarray,
    weight: float,
    destination: Path,
    counts: dict[str, int],
    guard: Any,
) -> dict[str, Any]:
    """Measure three updates and exactly replay updates two/three from the saved first state.

    Parameters
    ----------
    arm : str
        Fixed arm name.
    initial : dict
        Immutable common initialization.
    observed, targets, schedule : np.ndarray
        Training-only waveform, target and schedule arrays.
    weight : float
        Shared positive weight.
    destination : Path
        New study directory.
    counts : dict
        Mutable profile operation accounting.
    guard : callable
        Scientific resource admission check.

    Returns
    -------
    dict
        Cold/recurrent/replay/inference timings and exact recovery evidence.
    """
    directory = destination / "profile" / arm
    directory.mkdir(parents=True)
    model = learning.build_model().to("cuda:0")
    model.load_state_dict(initial)
    optimizer = optimizer_for(model)
    losses, timings, trajectory = [], [], {}
    for step in range(3):
        guard()
        started = {"cpu": time.process_time(), "wall": time.perf_counter()}
        counts["profile_update_attempts"] += 1
        loss = update(model, optimizer, observed, targets, schedule[step], weight, "cuda:0")
        counts["profile_updates"] += 1
        timings.append(quality.elapsed(started))
        losses.append(loss)
        if step == 0:
            save_state(directory / "step1_model.pt", model.state_dict())
            save_state(directory / "step1_optimizer.pt", optimizer.state_dict())
        if step in [1, 2]:
            trajectory[step] = (cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict()))
            save_state(directory / f"original_step{step + 1}_model.pt", trajectory[step][0])
            save_state(directory / f"original_step{step + 1}_optimizer.pt", trajectory[step][1])
    model.load_state_dict(torch.load(directory / "step1_model.pt", map_location="cpu", weights_only=True))
    optimizer.load_state_dict(
        torch.load(directory / "step1_optimizer.pt", map_location="cpu", weights_only=True)
    )
    replay, replay_timings, state_exact = [], [], []
    for step in [1, 2]:
        guard()
        counts["profile_update_attempts"] += 1
        started = {"cpu": time.process_time(), "wall": time.perf_counter()}
        replay.append(update(model, optimizer, observed, targets, schedule[step], weight, "cuda:0"))
        replay_timings.append(quality.elapsed(started))
        counts["profile_updates"] += 1
        replay_model, replay_optimizer = cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict())
        save_state(directory / f"replay_step{step + 1}_model.pt", replay_model)
        save_state(directory / f"replay_step{step + 1}_optimizer.pt", replay_optimizer)
        state_exact.append(
            trees_equal(trajectory[step][0], replay_model)
            and trees_equal(trajectory[step][1], replay_optimizer)
        )
    exact = replay == losses[1:] and all(state_exact)
    if not exact:
        raise ValueError("Profile model/Adam/loss replay differs")
    started = {"cpu": time.process_time(), "wall": time.perf_counter()}
    counts["profile_inference_attempts"] += 1
    infer(model, observed[schedule[0]], "cuda:0")
    counts["profile_inferences"] += 1
    evaluation = quality.elapsed(started)
    row = {
        "losses": losses,
        "replay_losses": replay,
        "timings": timings,
        "replay_timings": replay_timings,
        "trajectory_state_exact": state_exact,
        "inference": evaluation,
        "exact_replay": exact,
    }
    shared.write_new_json(directory / "summary.json", row)
    del model, optimizer
    return row


def profile_models(
    initial: dict[str, Any],
    observed: np.ndarray,
    targets: dict[str, np.ndarray],
    schedule: np.ndarray,
    weight: float,
    destination: Path,
    clock: dict[str, float],
    counts: dict[str, int],
    guard: Any,
) -> dict[str, Any]:
    """Admit main fits using cold-path charged time plus the frozen recurrent/evaluation forecast."""
    profiles = {
        arm: profile_arm(arm, initial, observed, target, schedule, weight, destination, counts, guard)
        for arm, target in targets.items()
    }
    spent = quality.elapsed(clock)
    result = {"arms": profiles, "spent": spent, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)}
    for kind in ["cpu", "wall"]:
        recurrent = max(
            row["timings"][step][f"{kind}_seconds"] for row in profiles.values() for step in [1, 2]
        )
        evaluation = max(row["inference"][f"{kind}_seconds"] for row in profiles.values())
        result[f"forecast_{kind}_seconds"] = (
            spent[f"{kind}_seconds"] + 2 * 800 * recurrent + 2 * 48 * evaluation + 60
        )
    result["spent_gpu_lock_wall_seconds"] = time.perf_counter() - clock["gpu_lock_start"]
    recurrent_wall = max(row["timings"][step]["wall_seconds"] for row in profiles.values() for step in [1, 2])
    inference_wall = max(row["inference"]["wall_seconds"] for row in profiles.values())
    result["forecast_gpu_lock_wall_seconds"] = (
        result["spent_gpu_lock_wall_seconds"] + 2 * 800 * recurrent_wall + 2 * 48 * inference_wall + 60
    )
    result["forecast_saved_bytes"] = (
        shared.directory_bytes(destination) + 2 * 48 * 4 * 12 * 5000 * 4 + 32 * 1024**2
    )
    shared.write_new_json(destination / "profile.json", result)
    if (
        result["forecast_cpu_seconds"] >= 1800
        or result["forecast_wall_seconds"] >= 1800
        or result["forecast_gpu_lock_wall_seconds"] >= 1200
        or result["forecast_saved_bytes"] >= 512 * 1024**2
        or result["peak_allocated_bytes"] >= 2 * 1024**3
    ):
        raise ValueError("Actual-path learning profile fails the prospective resource forecast")
    return result


def snapshot(destination: Path) -> dict[str, str]:
    """Preserve exact authored sources required by this learning executable."""
    sources = {}
    for name in SOURCES:
        path = destination / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        sources[name] = shared.file_sha256(path)
    return sources


def admit_parent(parent: Path, config: dict[str, Any], destination: Path, guard: Any) -> tuple[Any, ...]:
    """Reproduce all immediate predecessor coherence numbers without a generator forward.

    Parameters
    ----------
    parent : Path
        Available audited generated bank.
    config : dict
        Frozen predecessor byte bindings.
    destination : Path
        New learning directory.
    guard : callable
        Resource check used before each permitted read.

    Returns
    -------
    tuple
        Memory-mapped raw bank, predecessor summary, verified hashes and exact closure.
    """
    inputs = {}
    for name, expected in config["predecessor_sha256"].items():
        guard()
        if shared.file_sha256(parent / name, guard) != expected:
            raise ValueError("Frozen quality predecessor identity mismatch")
        inputs[name] = expected
    main = json.loads((parent / "summary.json").read_text())
    audit = json.loads((parent / "independent_audit_v1/summary.json").read_text())
    if not main["operational_support_qualified"] or not audit["all_passed"] or audit["mismatch_count"]:
        raise ValueError("Immediate quality predecessor did not pass independently")
    raw = np.load(parent / "raw.npy", mmap_mode="r", allow_pickle=False)
    if raw.dtype != np.float32 or raw.shape != (128, 8, 5000):
        raise ValueError("Predecessor raw bank geometry differs")
    saved = [json.loads(line) for line in (parent / "cases.jsonl").read_text().splitlines()]
    if len(saved) != 128:
        raise ValueError("Predecessor case rows incomplete")
    rows = []
    for index, case in enumerate(raw):
        guard()
        values, parameters = quality.standardize(case)
        coherent = quality.coherence(values, index)
        if parameters != saved[index]["standardization"] or coherent != saved[index]["coherence"]:
            raise ValueError("Exact predecessor per-seed standardization/coherence closure failed")
        rows.append({"coherence": coherent})
    primary = quality.primary_interval(rows)
    if primary != main["primary"]:
        raise ValueError("Exact predecessor primary/interval closure failed")
    closure = {"all_exact": True, "cases": 128, "primary": primary, "new_generator_forwards": 0}
    shared.write_new_json(destination / "predecessor_closure.json", closure)
    return raw, main, inputs, closure


def prepare_bank(
    raw: np.ndarray, destination: Path, rows: list[dict[str, Any]], guard: Any
) -> dict[str, np.ndarray]:
    """Retain new randomized observed fixtures and masks case by case in fresh memory maps.

    Parameters
    ----------
    raw : np.ndarray
        Existing 128 native generated outputs.
    destination : Path
        Fresh study directory.
    rows : list
        Mutable metadata prefix including any offending case.
    guard : callable
        Resource admission check.

    Returns
    -------
    dict
        Observed, target and strict-support memory maps after qualification.
    """
    banks = {}
    for name, shape, dtype in [
        ("observed", (128, 4, 12, 5000), np.float32),
        ("support", (128, 4, 12, 5000), bool),
        ("strict_support", (128, 2, 12, 5000), bool),
    ]:
        path = destination / f"{name}.npy"
        if path.exists():
            raise FileExistsError(path)
        banks[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    for index, native in enumerate(raw):
        guard()
        observed, strict, support, row = learning.make_case(native, index)
        for name, values in [("observed", observed), ("support", support), ("strict_support", strict)]:
            banks[name][index] = values
            banks[name].flush()
        row["sham_exact"] = bool(
            np.array_equal(observed[0], quality.derive_leads(quality.standardize(native)[0]))
        )
        row["limb_equations"] = bool(
            all(
                np.allclose(
                    case, quality.derive_leads(case[[0, 1, 6, 7, 8, 9, 10, 11]]), atol=2e-6, rtol=1e-6
                )
                for case in observed
            )
        )
        rows.append(row)
        if not all(
            check["strict_mask_exact"]
            and check["threshold_mask_exact"]
            and check["nonempty"]
            and check["threshold_subset_strict"]
            and check["outside_hull_cells"] == 0
            and check["retained_energy_fraction"] >= 0.99
            for check in row["checks"].values()
        ):
            raise ValueError("Randomized intervention integrity/energy gate failed")
        if (
            not np.isfinite(observed).all()
            or support[:2].any()
            or not row["sham_exact"]
            or not row["limb_equations"]
        ):
            raise ValueError("Observed geometry/control/sham/limb checks failed")
    quality.write_rows(destination / "cases.jsonl", rows)
    return banks


def verify_inputs(
    destination: Path,
    parent: Path,
    sources: dict[str, str],
    inputs: dict[str, str],
    bank_hashes: dict[str, str],
    guard: Any,
) -> None:
    """Rehash sources, raw predecessor and all immutable fitting/evaluation inputs."""
    for name, expected in sources.items():
        if (
            shared.file_sha256(ROOT / name, guard) != expected
            or shared.file_sha256(destination / "source_snapshot" / name, guard) != expected
        ):
            raise ValueError("Learning source or retained source snapshot changed")
    for root, bindings in [(parent, inputs), (destination, bank_hashes)]:
        for name, expected in bindings.items():
            if shared.file_sha256(root / name, guard) != expected:
                raise ValueError("Learning input bytes changed")


def train_pair(
    initial: dict[str, Any],
    observed: np.ndarray,
    targets: dict[str, np.ndarray],
    schedule: np.ndarray,
    weight: float,
    destination: Path,
    counts: dict[str, int],
    guard: Any,
) -> dict[str, torch.nn.Module]:
    """Fit supervised then permuted on each of the 400 identical minibatches.

    Parameters
    ----------
    initial : dict
        Common immutable initialization.
    observed : np.ndarray
        Only the 320 observed training variants, without auxiliary inputs.
    targets : dict
        Matched true and misplaced training masks.
    schedule : np.ndarray
        Fixed 400 by 8 training case indices.
    weight : float
        Training-only capped positive weight shared by both arms.
    destination : Path
        Fresh study output directory.
    counts : dict
        Attempted and completed scientific update counts.
    guard : callable
        Scientific resource guard.

    Returns
    -------
    dict
        Both fitted models after exactly 400 steps each.
    """
    models, optimizers, losses = {}, {}, {}
    for arm in ["supervised", "permuted"]:
        directory = destination / arm
        directory.mkdir()
        model = learning.build_model().to("cuda:0")
        model.load_state_dict(initial)
        if not trees_equal(initial, cpu_tree(model.state_dict())):
            raise ValueError("Main arm initialization differs from common source state")
        save_state(directory / "initial_state.pt", model.state_dict())
        models[arm], optimizers[arm] = model, optimizer_for(model)
        if optimizers[arm].state:
            raise ValueError("Main optimizer must start without inherited moments")
        losses[arm] = np.lib.format.open_memmap(
            directory / "losses.npy", mode="w+", dtype=np.float64, shape=(400,)
        )
    for index, batch in enumerate(schedule):
        for arm in ["supervised", "permuted"]:
            guard()
            counts[arm + "_update_attempts"] += 1
            losses[arm][index] = update(
                models[arm], optimizers[arm], observed, targets[arm], batch, weight, "cuda:0"
            )
            counts[arm + "_updates"] += 1
            losses[arm].flush()
            if not np.isfinite(losses[arm][index]):
                raise ValueError("Training loss became nonfinite")
    for arm, model in models.items():
        if any(float(state["step"]) != 400 for state in optimizers[arm].state.values()):
            raise ValueError("Main Adam steps differ from fixed update count")
        save_state(destination / arm / "final_state.pt", model.state_dict())
        save_state(destination / arm / "optimizer.pt", optimizers[arm].state_dict())
    return models


def evaluate_arm(
    arm: str,
    model: torch.nn.Module,
    observed: np.ndarray,
    support: np.ndarray,
    destination: Path,
    counts: dict[str, int],
    guard: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Evaluate 48 generator-seed backgrounds held out from this fit in fixed batches of eight cases.

    Parameters
    ----------
    arm : str
        Fitted arm name.
    model : torch.nn.Module
        Frozen final fitted network.
    observed, support : np.ndarray
        Held-out observed variants and separate scoring targets.
    destination : Path
        Current study directory.
    counts : dict
        Inference operation counts.
    guard : callable
        Resource check.

    Returns
    -------
    tuple
        Saved float32 probability memory map and per-seed scoring rows.
    """
    path = destination / arm / "evaluation_probabilities.npy"
    probabilities = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(48, 4, 12, 5000))
    flattened = probabilities.reshape(192, 12, 5000)
    values = observed.reshape(192, 12, 5000)
    for start in range(0, 192, 8):
        guard()
        counts[arm + "_inference_attempts"] += 1
        flattened[start : start + 8] = infer(model, values[start : start + 8], "cuda:0")
        counts[arm + "_inferences"] += 1
        probabilities.flush()
    if not np.isfinite(probabilities).all():
        raise ValueError("Evaluation maps are nonfinite")
    rows = learning.score_maps(probabilities, observed, support)
    shared.write_new_json(
        destination / arm / "metrics.json",
        {"cases": rows, "controls": learning.control_scores(probabilities)},
    )
    return probabilities, rows


def finish_ledger(
    destination: Path, receipt: dict[str, Any], name: str, clock: dict[str, float], parent: Path
) -> None:
    """Charge own terminal learning cost once after the previously audited cumulative chain."""
    digest = shared.file_sha256(destination / name)
    prior_path = parent / "independent_audit_v1/prospective_ledger.json"
    expected = json.loads((ROOT / CONFIG).read_text())["predecessor_sha256"][
        "independent_audit_v1/prospective_ledger.json"
    ]
    prior_digest = shared.file_sha256(prior_path) if prior_path.is_file() else None
    prior = json.loads(prior_path.read_text()) if prior_digest == expected else {}
    elapsed = quality.elapsed(clock)
    shared.write_new_json(
        destination / "prospective_ledger.json",
        {
            "stage": "randomized_localization_v1",
            "receipt": name,
            "receipt_sha256": digest,
            **elapsed,
            "gpu_wall_seconds": receipt.get("gpu_lock_wall_seconds", 0.0),
            "prior_quality_audit_ledger_sha256": prior_digest,
            "prior_accounting_verified": prior_digest == expected,
            "prior_current_machine_cpu_seconds": prior.get("current_machine_cumulative_cpu_seconds"),
            "prior_current_machine_gpu_wall_seconds": prior.get(
                "current_machine_cumulative_gpu_wall_seconds"
            ),
            "current_machine_cumulative_cpu_seconds": (
                prior["current_machine_cumulative_cpu_seconds"] + elapsed["cpu_seconds"]
            )
            if prior
            else None,
            "current_machine_cumulative_gpu_wall_seconds": (
                prior["current_machine_cumulative_gpu_wall_seconds"]
                + receipt.get("gpu_lock_wall_seconds", 0.0)
            )
            if prior
            else None,
            "historical_costs_verified": False,
            "measurement_endpoint": "after receipt write/hash, before terminal ledger serialization",
            "reserved_independent_audit": {"cpu_seconds": 180, "wall_seconds": 180, "bytes": 16 * 1024**2},
        },
    )


def execute_gpu(
    banks: dict[str, np.ndarray],
    initial: dict[str, Any],
    schedule: np.ndarray,
    weight: float,
    destination: Path,
    clock: dict[str, float],
    counts: dict[str, int],
    guard: Any,
    verify: Any,
) -> dict[str, Any]:
    """Profile, interleave both fits, then evaluate under one measured shared GPU lock."""
    train_observed = banks["observed"][:80].reshape(320, 12, 5000)
    targets = {
        arm: learning.training_targets(banks["support"][:80], arm) for arm in ["supervised", "permuted"]
    }
    if np.count_nonzero(targets["supervised"]) != np.count_nonzero(targets["permuted"]):
        raise ValueError("Negative control changed the global target collection")
    with runtime.measured_gpu_lock(clock):
        clock["gpu_lock_start"] = time.perf_counter()
        free, total = torch.cuda.mem_get_info(0)
        if free < 2 * 1024**3:
            raise ValueError("Learning GPU admission has insufficient free memory")
        torch.cuda.reset_peak_memory_stats(0)
        profile = profile_models(
            initial, train_observed, targets, schedule, weight, destination, clock, counts, guard
        )
        verify()
        models = train_pair(initial, train_observed, targets, schedule, weight, destination, counts, guard)
        verify()
        maps, case_metrics, model_hashes = {}, {}, {}
        for arm, model in models.items():
            maps[arm], case_metrics[arm] = evaluate_arm(
                arm, model, banks["observed"][80:], banks["support"][80:], destination, counts, guard
            )
            model_hashes[arm] = shared.tensor_state_sha256(model)
            model.cpu()
        metadata = {
            "name": torch.cuda.get_device_name(0),
            "driver": runtime.subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).splitlines()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
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
        "maps": maps,
        "cases": case_metrics,
        "profile": profile,
        "gpu_metadata": metadata,
        "final_state_sha256": model_hashes,
    }


def count_template() -> dict[str, int]:
    """Create every attempted/completed profile and main operation counter before admission."""
    counts = dict.fromkeys(
        ["profile_update_attempts", "profile_updates", "profile_inference_attempts", "profile_inferences"], 0
    )
    for arm in ["supervised", "permuted"]:
        counts.update(
            {
                arm + suffix: 0
                for suffix in ["_update_attempts", "_updates", "_inference_attempts", "_inferences"]
            }
        )
    return counts


def run_study(output_root: Path) -> dict[str, Any]:
    """Execute the committed learning protocol once and preserve any failed attempt.

    Parameters
    ----------
    output_root : Path
        Local repository root hosting the audited predecessor and fresh outputs.

    Returns
    -------
    dict
        Executed scientific verdict, integrity gates, provenance and measured costs.
    """
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    config = json.loads((ROOT / CONFIG).read_text())
    destination = output_root / config["output_root"]
    destination.mkdir(exist_ok=False)
    parent = output_root / config["predecessor_root"]
    counts, rows, sources, inputs, bank_hashes = count_template(), [], {}, {}, {}
    old_handlers = {}

    def guard() -> None:
        spent = quality.elapsed(clock)
        if max(spent.values()) + 2 >= 1800:
            raise TimeoutError("Learning whole-stage CPU or wall deadline reached")
        if (
            "gpu_lock_start" in clock
            and "gpu_cost" not in clock
            and time.perf_counter() - clock["gpu_lock_start"] + 2 >= 1200
        ):
            raise TimeoutError("Learning GPU-lock wall deadline reached")
        if shared.directory_bytes(destination) + 1024**2 >= 512 * 1024**2:
            raise ValueError("Learning output receipt reserve exhausted")

    try:
        runtime.set_runtime()
        runtime.arm_timers(config["resources"], old_handlers)
        sources = snapshot(destination)
        raw, _, inputs, closure = admit_parent(parent, config, destination, guard)
        banks = prepare_bank(raw, destination, rows, guard)
        schedule = learning.training_schedule()
        quality.save_array(destination / "training_schedule.npy", schedule)
        quality.save_array(destination / "background_schedule.npy", schedule[:, [0, 4]] // 4)
        quality.save_array(destination / "donor_map.npy", (np.arange(80) + 1) % 80)
        prior = banks["support"][:80, 2:].mean(axis=(0, 1))
        quality.save_array(destination / "spatial_prior.npy", prior)
        weight = learning.positive_weight(banks["support"][:80])
        positives = int(np.count_nonzero(banks["support"][:80]))
        shared.write_new_json(
            destination / "training_design.json",
            {
                "positive_weight": weight,
                "positive_cells": positives,
                "negative_cells": int(banks["support"][:80].size - positives),
                "train_backgrounds": 80,
                "evaluation_backgrounds": 48,
            },
        )
        torch.manual_seed(2026103501)
        initial_model = learning.build_model()
        initial = cpu_tree(initial_model.state_dict())
        initial_hash = shared.tensor_state_sha256(initial_model)
        save_state(destination / "initial_state.pt", initial)
        bank_hashes = {
            name: shared.file_sha256(destination / name, guard)
            for name in [
                "observed.npy",
                "support.npy",
                "strict_support.npy",
                "cases.jsonl",
                "training_schedule.npy",
                "background_schedule.npy",
                "donor_map.npy",
                "spatial_prior.npy",
                "initial_state.pt",
                "training_design.json",
            ]
        }

        def verify() -> None:
            verify_inputs(destination, parent, sources, inputs, bank_hashes, guard)

        verify()
        executed = execute_gpu(banks, initial, schedule, weight, destination, clock, counts, guard, verify)
        maps, case_metrics = executed.pop("maps"), executed.pop("cases")
        for arm, values in case_metrics.items():
            learning.add_spatial_prior(values, banks["support"][80:], prior)
            shared.write_new_json(
                destination / arm / "evaluation_metrics.json",
                {"cases": values, "controls": learning.control_scores(maps[arm])},
            )
        metrics = learning.summarize_metrics(case_metrics["supervised"], case_metrics["permuted"])
        figures = learning.make_figures(
            destination, banks["observed"][80:], banks["support"][80:], maps, rows, metrics
        )
        verify()
        integrity = {
            "parent_exact": closure["all_exact"],
            "all_128_supports_qualified": len(rows) == 128,
            "matched_initial_states": all(
                trees_equal(
                    initial,
                    torch.load(destination / arm / "initial_state.pt", map_location="cpu", weights_only=True),
                )
                for arm in ["supervised", "permuted"]
            ),
            "profile_exact": all(value["exact_replay"] for value in executed["profile"]["arms"].values()),
            "profile_updates_10": counts["profile_updates"] == counts["profile_update_attempts"] == 10,
            "profile_inferences_2": counts["profile_inferences"] == counts["profile_inference_attempts"] == 2,
            "main_updates_400_each": all(
                counts[arm + "_updates"] == counts[arm + "_update_attempts"] == 400
                for arm in ["supervised", "permuted"]
            ),
            "evaluation_batches_24_each": all(
                counts[arm + "_inferences"] == counts[arm + "_inference_attempts"] == 24
                for arm in ["supervised", "permuted"]
            ),
            "resources": clock["gpu_cost"] < 1200
            and executed["gpu_metadata"]["peak_allocated_bytes"] < 2 * 1024**3,
        }
        guard()
        result = {
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            **metrics,
            "integrity_gates": integrity,
            "all_passed": metrics["scientific_passed"] and all(integrity.values()),
            "source_sha256": sources,
            "input_sha256": inputs,
            "bank_sha256": bank_hashes,
            "output_sha256": {
                path.relative_to(destination).as_posix(): shared.file_sha256(path, guard)
                for path in destination.rglob("*")
                if path.is_file()
            },
            "initial_state_sha256": initial_hash,
            **executed,
            "counts": counts,
            "positive_weight": weight,
            "figures": figures,
            "gpu_lock_wall_seconds": clock["gpu_cost"],
            **quality.elapsed(clock),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "output_bytes_before_receipt": shared.directory_bytes(destination),
            "new_generator_forwards": 0,
            "real_waveform_reads": 0,
            "external_api_calls": 0,
            "clinical_claim": False,
            "physical_units_verified": False,
            "support_kind": "introduced numerical smooth offsets on generated backgrounds",
        }
        guard()
        shared.write_new_json(destination / "summary.json", result)
        finish_ledger(destination, result, "summary.json", clock, parent)
        return result
    except Exception as error:
        if rows and not (destination / "partial_cases.jsonl").exists():
            quality.write_rows(destination / "partial_cases.jsonl", rows)
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "counts": counts,
            "completed_or_offending_case_rows": len(rows),
            "source_sha256": sources,
            "input_sha256": inputs,
            "bank_sha256": bank_hashes,
            "gpu_lock_wall_seconds": clock.get("gpu_cost", 0.0),
            **quality.elapsed(clock),
            "historical_costs_verified": False,
        }
        shared.write_new_json(destination / "failure.json", failure)
        finish_ledger(destination, failure, "failure.json", clock, parent)
        raise
    finally:
        restore_timers(old_handlers)


def restore_timers(old_handlers: dict[int, Any]) -> None:
    """Restore only signal handlers and timers installed by this learning attempt."""
    for timer, sig in [(signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)]:
        if sig in old_handlers:
            signal.setitimer(timer, 0)
            signal.signal(sig, old_handlers[sig])
