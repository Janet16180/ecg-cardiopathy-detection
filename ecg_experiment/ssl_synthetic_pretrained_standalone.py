"""Forward-only Pulse2Pulse feasibility when historical local receipts are absent."""

from __future__ import annotations

import json
import resource
import signal
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_pretrained_generator as shared

CONFIG_NAME = "configs/ssl_synthetic_pretrained_generator_standalone_v1.json"
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_pretrained_standalone.py",
    "scripts/validation/probe_ssl_synthetic_pretrained_standalone.py",
    "tests/test_ssl_synthetic_pretrained_standalone.py",
    "docs/ssl-synthetic-localization-pretrained-generator-standalone-v1.md",
    CONFIG_NAME,
    *shared.SOURCE_NAMES,
)


def snapshot_sources(study: Path, guard: Callable[[], None]) -> dict[str, str]:
    """Preserve exact authored inputs before contacting any public asset host."""
    sources = {}
    for name in dict.fromkeys(SOURCE_NAMES):
        guard()
        path = ROOT / name
        data = path.read_bytes()
        target = study / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        sources[name] = shared.file_sha256(target, guard)
        if shared.file_sha256(path, guard) != sources[name]:
            raise ValueError("Source changed while its snapshot was written")
    return sources


def save_array(path: Path, value: np.ndarray) -> None:
    """Retain a completed numerical artifact without replacing another attempt."""
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def save_case(study: Path, prefix: str, values: tuple[np.ndarray, ...]) -> dict[str, str]:
    """Save every completed forward before evaluating its technical gates."""
    files = {}
    for name, value in zip(["raw", "author_export", "canonical12"], values, strict=True):
        filename = f"{prefix}_{name}.npy"
        save_array(study / filename, value)
        files[name] = filename
    return files


def deadline_handler(signum: int, frame: Any) -> None:
    """Interrupt an in-flight forward or transfer before the receipt reserve."""
    raise TimeoutError(f"Standalone technical probe timer expired ({signum})")


def verify_config(config: dict[str, Any]) -> None:
    """Reject changes to the frozen geometry, seed, budget and public-asset contract."""
    if config["seeds"] != [2026102901, 2026102902, 2026102903, 2026102904]:
        raise ValueError("Frozen four-seed contract changed")
    if (
        config["cpu_wall_limit_seconds"] != 180
        or config["output_bytes_ceiling"] != 96 * 1024**2
        or config["batch_size"] != 1
        or config["samples"] != 5000
        or config["independent_leads"] != 8
        or config["accounting_scope"] != "prospective_standalone_only"
    ):
        raise ValueError("Frozen standalone geometry or prospective budget changed")
    names = [asset["filename"] for asset in config["public_assets"]]
    if set(names) != {"pytorch_model.bin", "hf_modeling_deepfake.py", "mit_pulse2pulse.py", "MIT_LICENSE"}:
        raise ValueError("Unexpected standalone public-asset allowlist")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate public asset")


def profile_forward(
    model: torch.nn.Module, noise: np.ndarray, clock: dict[str, float], config: dict[str, Any]
) -> dict[str, float]:
    """Measure the one disposable forward and retain conservative CPU/wall forecasts."""
    wall, cpu = time.perf_counter(), time.process_time()
    shared.generate(model, noise)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    spent_wall, spent_cpu = time.perf_counter() - clock["wall"], time.process_time() - clock["cpu"]
    return {
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "spent_wall_seconds": spent_wall,
        "spent_cpu_seconds": spent_cpu,
        "forecast_wall_seconds": spent_wall
        + config["profile_forward_multiplier"] * wall
        + config["report_reserve_seconds"],
        "forecast_cpu_seconds": spent_cpu
        + config["profile_forward_multiplier"] * cpu
        + config["report_reserve_seconds"],
    }


def fetch_assets(
    config: dict[str, Any], study: Path, inputs: dict[str, str], guard: Callable[[], None]
) -> dict[str, Path]:
    """Fetch exactly the declared public files while retaining verified input bindings."""
    assets = {}
    for asset in config["public_assets"]:
        path = shared.download_asset(asset, study, guard)
        assets[asset["filename"]] = path
        inputs[path.name] = asset["sha256"]
    return assets


def verify_artifacts(
    study: Path, sources: dict[str, str], inputs: dict[str, str], guard: Callable[[], None]
) -> dict[str, str]:
    """Recheck source, snapshot and asset bytes and bind every retained output file."""
    for name, digest in sources.items():
        current = shared.file_sha256(ROOT / name, guard)
        snapshot = shared.file_sha256(study / "source_snapshot" / name, guard)
        if current != digest or snapshot != digest:
            raise ValueError("Source or source snapshot changed during execution")
    for name, digest in inputs.items():
        if shared.file_sha256(study / name, guard) != digest:
            raise ValueError("Public input changed during execution")
    return {
        p.relative_to(study).as_posix(): shared.file_sha256(p, guard) for p in study.rglob("*") if p.is_file()
    }


def technical_gates(
    checks: list[dict[str, bool]],
    completed: list[int],
    first: tuple[np.ndarray, ...],
    replay: tuple[np.ndarray, ...],
    raw_bank: list[np.ndarray],
    architecture_proof: dict[str, bool],
    initial_state: str,
    final_state: str,
) -> dict[str, bool]:
    """Return every prospective technical gate without clinical interpretation."""
    return {
        "all_case_checks": all(all(row.values()) for row in checks),
        "four_fixed_inputs": len(completed) == 4,
        "exact_replay": all(np.array_equal(a, b) for a, b in zip(first, replay, strict=True)),
        "distinct_fixed_seed_outputs": all(
            not np.array_equal(raw_bank[i], raw_bank[j]) for i in range(4) for j in range(i)
        ),
        "strict_finite_tensor_loading": True,
        "architecture_equivalence": all(architecture_proof.values()),
        "model_state_unchanged": initial_state == final_state,
    }


def arm_deadlines(config: dict[str, Any], old_handlers: dict[int, Any]) -> None:
    """Install CPU and wall timers without replacing another active process timer."""
    for timer in [signal.ITIMER_REAL, signal.ITIMER_PROF]:
        if signal.getitimer(timer) != (0.0, 0.0):
            raise ValueError("Existing process timer prevents exclusive deadline admission")
    for timer, sig in [(signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)]:
        old_handlers[sig] = signal.signal(sig, deadline_handler)
        signal.setitimer(timer, config["cpu_wall_limit_seconds"] - config["failure_reserve_seconds"])


def disarm_deadlines(old_handlers: dict[int, Any]) -> None:
    """Restore only timers and signal handlers installed by this attempt."""
    for timer, sig in [(signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)]:
        if sig in old_handlers:
            signal.setitimer(timer, 0)
            signal.signal(sig, old_handlers[sig])


def run_probe(output_root: Path, config_path: Path = ROOT / CONFIG_NAME) -> dict[str, Any]:
    """Run one standalone attempt, with a fresh zero-based accounting scope and no old data."""
    clock = {"wall": time.perf_counter(), "cpu": time.process_time()}
    study = output_root / "outputs/ssl_synthetic_localization/pretrained_generator_standalone_v1"
    study.parent.mkdir(parents=True, exist_ok=True)
    study.mkdir(exist_ok=False)
    config: dict[str, Any] = {}
    sources: dict[str, str] = {}
    inputs: dict[str, str] = {}
    completed: list[int] = []
    files: list[dict[str, str]] = []
    counts = {"forward_attempts": 0, "completed_forwards": 0}
    old_handlers = {}

    def guard() -> None:
        shared.check_limits(
            config,
            clock,
            0.0,
            study_bytes=shared.directory_bytes(study),
            stream_bytes=shared.directory_bytes(study.parent),
            reserve=config["failure_reserve_seconds"],
        )
        if shared.directory_bytes(study) + 1024**2 >= config["output_bytes_ceiling"]:
            raise ValueError("One MiB immutable receipt reserve exhausted")

    try:
        config = json.loads(config_path.read_text())
        verify_config(config)
        shared.configure_cpu(config)
        arm_deadlines(config, old_handlers)
        guard()
        sources = snapshot_sources(study, guard)
        if shared.file_sha256(config_path, guard) != sources[CONFIG_NAME]:
            raise ValueError("Execution config differs from committed standalone config snapshot")
        if (
            shared.directory_bytes(study.parent)
            + config["output_bytes_ceiling"]
            + config["audit_output_bytes_ceiling"]
            >= config["cache_bytes_ceiling"]
        ):
            raise ValueError("Prospective stream storage reservation fails")
        shared.write_new_json(
            study / "admission.json",
            {
                "source_sha256": sources,
                "config": config,
                "output_relative": config["output_relative"],
                "accounting_scope": config["accounting_scope"],
                "verified_prior_cpu_seconds": 0.0,
                "verified_prior_gpu_wall_seconds": 0.0,
                "historical_receipts_available": False,
                "historical_context_unverified": config["historical_context_unverified"],
                "historical_reproduction": "not possible: historical local payloads are absent",
                "inference_admission": "coordinator requires committed source and independent source review",
            },
        )
        assets = fetch_assets(config, study, inputs, guard)
        mit = assets["mit_pulse2pulse.py"].read_text()
        hf = assets["hf_modeling_deepfake.py"].read_text()
        architecture_proof = shared.verify_architecture_equivalence(mit, hf)
        model = shared.load_generator(assets["pytorch_model.bin"], shared.architecture_class(mit, hf))
        initial_state = shared.tensor_state_sha256(model)
        noises = np.stack(
            [
                torch.empty((1, 8, 5000), dtype=torch.float32)
                .uniform_(-1, 1, generator=torch.Generator(device="cpu").manual_seed(seed))
                .numpy()
                for seed in config["seeds"]
            ]
        )
        save_array(study / "noise.npy", noises)
        guard()
        counts["forward_attempts"] += 1
        profile = profile_forward(model, noises[0], clock, config)
        counts["completed_forwards"] += 1
        shared.write_new_json(study / "profile.json", profile)
        guard()
        if (
            max(profile["forecast_cpu_seconds"], profile["forecast_wall_seconds"])
            >= config["cpu_wall_limit_seconds"]
        ):
            raise TimeoutError("Actual-forward conservative technical generation forecast fails")
        checks, raw_bank, first = [], [], None
        for index, (seed, noise) in enumerate(zip(config["seeds"], noises, strict=True)):
            guard()
            counts["forward_attempts"] += 1
            values = shared.generate(model, noise)
            counts["completed_forwards"] += 1
            files.append(save_case(study, f"case_{index}", values))
            completed.append(seed)
            checks.append(shared.output_checks(*values))
            raw_bank.append(values[0])
            if index == 0:
                first = values
            guard()
        counts["forward_attempts"] += 1
        replay = shared.generate(model, noises[0])
        counts["completed_forwards"] += 1
        replay_files = save_case(study, "replay", replay)
        guard()
        final_state = shared.tensor_state_sha256(model)
        gates = technical_gates(
            checks, completed, first, replay, raw_bank, architecture_proof, initial_state, final_state
        )
        outputs = verify_artifacts(study, sources, inputs, guard)
        guard()
        result = {
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "technical_generation_feasible": all(gates.values()),
            "gates": gates,
            "case_checks": checks,
            "completed_seeds": completed,
            "case_files": files,
            "replay_files": replay_files,
            "architecture_proof": architecture_proof,
            **counts,
            "forward_allocation": {"profile": 1, "fixed_seed_bank": 4, "replay": 1},
            "model_buffers": len(list(model.named_buffers())),
            "model_state_sha256": initial_state,
            "initial_state_sha256": initial_state,
            "final_state_sha256": final_state,
            "model_parameters": sum(value.numel() for value in model.parameters()),
            "source_sha256": sources,
            "input_sha256": inputs,
            "output_sha256": outputs,
            "profile": profile,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "study_bytes_before_receipt": shared.directory_bytes(study),
            "stream_bytes_before_receipt": shared.directory_bytes(study.parent),
            "raw_units": "dimensionless tanh output",
            "author_export_units": "unverified numerical integer units",
            "canonical12_units": "same unverified author-export numerical units",
            "physical_units_verified": False,
            "clinical_claim": False,
            "regional_mask_control": False,
            "checkpoint_cohort_membership_verified": False,
            "new_training_updates": 0,
            "real_waveform_reads": 0,
            "external_inference_api_calls": 0,
            "gpu_used": False,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "input_array_sha256": shared.array_sha256(noises),
            "raw_array_sha256": shared.array_sha256(np.stack(raw_bank)),
        }
        result.update(accounting_fields(clock, config))
        guard()
        shared.write_new_json(study / "summary.json", result)
        write_ledger(study, result, "summary.json", clock)
        return result
    except Exception as error:
        failure = {
            "source_sha256": sources,
            "input_sha256": inputs,
            "completed_seeds": completed,
            "case_files": files,
            "failure_type": type(error).__name__,
            "failure_detail": str(error),
            **counts,
            **accounting_fields(clock, config),
            "gpu_used": False,
            "clinical_performance_scored": False,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        shared.write_new_json(study / "failure.json", failure)
        write_ledger(study, failure, "failure.json", clock)
        raise RuntimeError(
            "Standalone generator stopped; retained local failure and accounting receipts"
        ) from None
    finally:
        disarm_deadlines(old_handlers)


def accounting_fields(clock: dict[str, float], config: dict[str, Any]) -> dict[str, Any]:
    """Keep missing historical costs distinct from this verifiable prospective attempt."""
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    return {
        "accounting_scope": "prospective_standalone_only",
        "historical_receipts_available": False,
        "historical_context_unverified": config.get("historical_context_unverified", {}),
        "prior_cpu_seconds": 0.0,
        "prior_gpu_wall_seconds": 0.0,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "cumulative_cpu_seconds": cpu,
        "global_gpu_wall_seconds": 0.0,
    }


def write_ledger(study: Path, receipt: dict[str, Any], receipt_name: str, clock: dict[str, float]) -> None:
    """Bind this success or failure once, reserving a separate future independent audit."""
    receipt_sha256 = shared.file_sha256(study / receipt_name)
    terminal_cpu = time.process_time() - clock["cpu"]
    terminal_wall = time.perf_counter() - clock["wall"]
    shared.write_new_json(
        study / "prospective_ledger.json",
        {
            "accounting_scope": "prospective_standalone_only",
            "historical_receipts_available": False,
            "historical_context_unverified": receipt["historical_context_unverified"],
            "entries": [
                {
                    "stage": "generator_feasibility",
                    "receipt": receipt_name,
                    "sha256": receipt_sha256,
                    "cpu_seconds": terminal_cpu,
                    "wall_seconds": terminal_wall,
                    "receipt_cpu_seconds": receipt["cpu_seconds"],
                    "receipt_wall_seconds": receipt["wall_seconds"],
                    "gpu_wall_seconds": 0.0,
                }
            ],
            "cpu_seconds": terminal_cpu,
            "gpu_wall_seconds": 0.0,
            "measurement_endpoint": "after main receipt write/hash, before terminal ledger serialization",
            "reserved_independent_audit": {"cpu_seconds": 120, "wall_seconds": 120, "bytes": 16 * 1024**2},
        },
    )
