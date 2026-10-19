"""Bounded local technical probe of a pinned, public Pulse2Pulse generator."""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import os
import resource
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from ecg_experiment import ROOT
from ecg_experiment.paths import to_stored

BASE = ROOT / "outputs/ssl_synthetic_localization"
STUDY = BASE / "pretrained_generator_feasibility_v1"
CONFIG = ROOT / "configs/ssl_synthetic_pretrained_generator_feasibility_v1.json"
PARENT = BASE / "counterfactual_factorial_v1/independent_audit_v1/summary.json"
PARENT_SHA256 = "74b02f60ac0d77bf071bc99e3ae6fcf460fd8ef0dfba21b9626f5d67646d6c04"
CLASS_NAMES = ("Transpose1dLayer", "Transpose1dLayer_multi_input", "WaveGANGenerator")
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_pretrained_generator.py",
    "scripts/validation/probe_ssl_synthetic_pretrained_generator.py",
    "tests/test_ssl_synthetic_pretrained_generator.py",
    "docs/ssl-synthetic-localization-pretrained-generator-feasibility.md",
    "docs/ssl-synthetic-localization-pretrained-generator-assessment.md",
    "configs/ssl_synthetic_pretrained_generator_feasibility_v1.json",
    "ecg_experiment/paths.py",
)


def write_new_json(path: Path, value: Any) -> None:
    """Write JSON without replacing any existing artifact."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_sha256(path: Path, guard: Callable[[], None] = lambda: None) -> str:
    """Hash in bounded chunks, checking the live deadline between reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            guard()
            chunk = stream.read(1024**2)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash canonical contiguous array bytes."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_state_sha256(model: nn.Module) -> str:
    """Bind all named CPU tensor values before and after inference."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    """Count actual local file bytes beneath a study or stream."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def check_limits(
    config: dict[str, Any],
    clock: dict[str, float],
    prior_cpu: float,
    *,
    study_bytes: int,
    stream_bytes: int,
    reserve: float = 2.0,
) -> None:
    """Enforce CPU, monotonic wall and actual study/stream byte ceilings."""
    cpu = time.process_time() - clock["cpu"]
    wall = time.perf_counter() - clock["wall"]
    if max(cpu, wall) + reserve >= config["cpu_wall_limit_seconds"]:
        raise TimeoutError("Technical probe CPU or monotonic wall deadline reached")
    if prior_cpu + cpu + reserve >= config["cpu_seconds_ceiling"]:
        raise TimeoutError("Cumulative CPU deadline reached")
    if study_bytes >= config["output_bytes_ceiling"] or stream_bytes >= config["cache_bytes_ceiling"]:
        raise ValueError("Actual study or complete stream byte ceiling reached")


def download_asset(asset: dict[str, Any], destination: Path, guard: Callable[[], None]) -> Path:
    """Fetch one explicitly frozen public file, preserving partial transfers on failure."""
    name = asset["filename"]
    if Path(name).name != name or not asset["url"].startswith("https://"):
        raise ValueError("Public asset must have a flat filename and HTTPS URL")
    path = destination / name
    partial = destination / (name + ".partial")
    guard()
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "ECG-local-technical-probe"})
    count = 0
    with partial.open("xb") as stream, urllib.request.urlopen(request, timeout=10) as response:
        while True:
            guard()
            chunk = response.read(min(1024**2, asset["bytes"] + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > asset["bytes"]:
                raise ValueError("Public transfer exceeds its frozen byte count")
            stream.write(chunk)
    if count != asset["bytes"] or file_sha256(partial, guard) != asset["sha256"]:
        raise ValueError("Public transfer size or SHA-256 differs from the frozen declaration")
    if path.exists():
        raise FileExistsError(path)
    partial.rename(path)
    return path


def _classes(source: str) -> dict[str, ast.ClassDef]:
    """Parse declarations without importing or executing the remote module."""
    tree = ast.parse(source)
    allowed_imports = {"torch", "torch.nn", "torch.nn.functional", "torch.utils.data"}
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name in classes:
                raise ValueError("Duplicate architecture class")
            classes[node.name] = node
        elif isinstance(node, ast.Import):
            if any(item.name not in allowed_imports for item in node.names):
                raise ValueError("Unexpected architecture import")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in {"transformers", "configurations_deepfake"}:
                raise ValueError("Unexpected architecture from-import")
        elif not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            raise ValueError("Unexpected executable architecture top-level structure")
    return classes


def verify_architecture_equivalence(mit_source: str, hf_source: str) -> dict[str, bool]:
    """Verify reviewed MIT classes match the pinned Hub computation and compatibility delta."""
    mit, hf = _classes(mit_source), _classes(hf_source)
    if not all(name in mit for name in CLASS_NAMES) or "Pulse2pulseGenerator" not in hf:
        raise ValueError("Expected reviewed architecture class is absent")
    for name in CLASS_NAMES[:2]:
        if ast.dump(mit[name]) != ast.dump(hf[name]):
            raise ValueError("MIT and Hub transpose-layer definitions differ")
    normalized = copy.deepcopy(hf["Pulse2pulseGenerator"])
    normalized.name = "WaveGANGenerator"
    for node in ast.walk(normalized):
        if isinstance(node, ast.Name) and node.id == "Pulse2pulseGenerator":
            node.id = "WaveGANGenerator"
    init = next(
        node for node in normalized.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    position = next(i for i, arg in enumerate(init.args.args) if arg.arg == "latent_dim")
    default_position = position - (len(init.args.args) - len(init.args.defaults))
    if ast.dump(init.args.defaults[default_position]) != ast.dump(ast.Constant(value=100)):
        raise ValueError("Unexpected Hub compatibility latent dimension")
    del init.args.args[position]
    del init.args.defaults[default_position]
    expected = {"self.latent_di = latent_dim", "self.fc1 = nn.Linear(latent_dim, 10 * model_size)"}
    removed = {
        ast.unparse(node)
        for node in init.body
        if isinstance(node, ast.Assign) and ast.unparse(node) in expected
    }
    if removed != expected:
        raise ValueError("Unexpected Hub compatibility layer declarations")
    init.body = [
        node for node in init.body if not (isinstance(node, ast.Assign) and ast.unparse(node) in expected)
    ]
    if any(
        isinstance(node, ast.Attribute)
        and node.attr in {"fc1", "latent_di"}
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        for node in ast.walk(normalized)
    ):
        raise ValueError("Hub compatibility attributes participate in the generator computation")
    if ast.dump(normalized) != ast.dump(mit["WaveGANGenerator"]):
        raise ValueError("MIT and Hub generator differ beyond the reviewed compatibility layer")
    return {"transpose_classes_exact": True, "generator_exact_except_unused_fc1": True}


def architecture_class(mit_source: str, hf_source: str) -> type[nn.Module]:
    """Load only reviewed, hash-pinned MIT class declarations with explicit runtime globals."""
    verify_architecture_equivalence(mit_source, hf_source)
    definitions = _classes(mit_source)
    selected = ast.Module(body=[definitions[name] for name in CLASS_NAMES], type_ignores=[])
    safe_builtins = {
        name: getattr(builtins, name) for name in ["__build_class__", "super", "isinstance", "int", "print"]
    }
    namespace = {
        "__builtins__": safe_builtins,
        "__name__": "reviewed_local_pulse2pulse",
        "torch": torch,
        "nn": nn,
        "F": functional,
    }
    exec(compile(selected, "<hash-pinned-MIT-class-subset>", "exec"), namespace)
    return namespace["WaveGANGenerator"]


class GeneratorState(nn.Module):
    """Expose the exact Hub state-key namespace over the reviewed MIT computation."""

    def __init__(self, generator: nn.Module) -> None:
        super().__init__()
        self.model = generator
        self.model.latent_di = 100
        self.model.fc1 = nn.Linear(100, 500)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Apply the unchanged reviewed generator computation."""
        return self.model(noise)


def load_generator(path: Path, generator_type: type[nn.Module]) -> GeneratorState:
    """Load a finite, exact-key float32 tensor state without unrestricted pickle fallback."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items())
    ):
        raise TypeError("Checkpoint must contain only a plain tensor state dictionary")
    model = GeneratorState(
        generator_type(
            model_size=50, ngpus=1, num_channels=8, post_proc_filt_len=512, verbose=False, upsample=True
        )
    )
    if list(model.named_buffers()):
        raise ValueError("Reviewed convolutional architecture unexpectedly declares tensor buffers")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Checkpoint tensor keys differ from the reviewed architecture")
    for name, value in state.items():
        if (
            value.dtype != torch.float32
            or value.shape != expected[name].shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Checkpoint tensor dtype, shape or finiteness check failed")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def canonical_leads(export: np.ndarray) -> np.ndarray:
    """Derive canonical twelve leads in the same unverified author-export numerical units."""
    if export.dtype != np.int32 or export.shape != (5000, 8):
        raise ValueError("Author export must be int32 with shape (5000,8)")
    values = export.T.astype(np.float32)
    lead_i, lead_ii = values[:2]
    return np.stack(
        [
            lead_i,
            lead_ii,
            lead_ii - lead_i,
            -(lead_i + lead_ii) / 2,
            lead_i - lead_ii / 2,
            lead_ii - lead_i / 2,
            *values[2:],
        ]
    ).astype(np.float32)


def output_checks(raw: np.ndarray, export: np.ndarray, canonical: np.ndarray) -> dict[str, bool]:
    """Check numerical geometry/export/algebra without clinical or physical-unit interpretation."""
    geometry = raw.dtype == np.float32 and raw.shape == (8, 5000)
    if not geometry:
        raise ValueError("Raw output must be float32 with shape (8,5000)")
    if not np.isfinite(raw).all() or export.dtype != np.int32 or export.shape != (5000, 8):
        raise ValueError("Output finiteness or author export geometry failed")
    expected = (raw * np.float32(6000)).astype(np.int32).T
    if not np.array_equal(export, expected):
        raise ValueError("Author export differs from multiply-6000/int32/transposition")
    if canonical.dtype != np.float32 or canonical.shape != (12, 5000) or not np.isfinite(canonical).all():
        raise ValueError("Canonical numerical geometry failed")
    if not np.allclose(canonical, canonical_leads(export), rtol=0, atol=1e-5):
        raise ValueError("Canonical lead equations failed")
    return {
        "native_geometry": True,
        "finite": True,
        "author_export_exact": True,
        "canonical_algebra_atol_1e_minus5": True,
        "every_raw_lead_varies": bool(np.all(np.ptp(raw, axis=1) > 0)),
    }


def generate(model: nn.Module, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate once from retained noise using CPU inference and explicit export arithmetic."""
    if noise.dtype != np.float32 or noise.shape != (1, 8, 5000):
        raise ValueError("Noise must have native float32 shape (1,8,5000)")
    with torch.inference_mode():
        output = model(torch.from_numpy(noise))
    if (
        output.dtype != torch.float32
        or tuple(output.shape) != (1, 8, 5000)
        or not torch.isfinite(output).all()
    ):
        raise ValueError("Native generator output dtype, shape or finiteness failed")
    raw = output[0].cpu().numpy().copy()
    export = (output[0] * 6000).to(torch.int32).T.cpu().numpy().copy()
    canonical = canonical_leads(export)
    return raw, export, canonical


def configure_cpu(config: dict[str, Any]) -> None:
    """Require frozen assets, hidden CUDA and one thread before any model operation."""
    if "public_assets" not in config:
        raise ValueError("Prospective public-assets amendment is required before any probe download")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This CPU-only probe requires CUDA_VISIBLE_DEVICES to be empty")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(name) != "1":
            raise ValueError("Numerical thread environment must be fixed to one")


def verify_snapshots(sources: dict[str, str], inputs: dict[str, str], guard: Callable[[], None]) -> None:
    """Verify all pinned local sources and bound inputs again at the end of the probe."""
    for label, catalog in [("source", sources), ("input", inputs)]:
        for name, digest in catalog.items():
            if file_sha256(ROOT / name, guard) != digest:
                raise ValueError(f"Pinned {label} changed during execution")


def download_assets(
    config: dict[str, Any], inputs: dict[str, str], guard: Callable[[], None]
) -> dict[str, Path]:
    """Download only the explicit prospective asset list and bind its verified digests."""
    assets = {}
    for asset in config["public_assets"]:
        path = download_asset(asset, STUDY, guard)
        assets[asset["filename"]] = path
        inputs[to_stored(path)] = asset["sha256"]
    return assets


def check_receipt_reserve(config: dict[str, Any]) -> None:
    """Reserve one MiB for the final immutable receipt within both storage ceilings."""
    if (
        directory_bytes(STUDY) + 1024**2 >= config["output_bytes_ceiling"]
        or directory_bytes(BASE) + 1024**2 >= config["cache_bytes_ceiling"]
    ):
        raise ValueError("Final receipt storage reserve exhausted")


def run_probe() -> dict[str, Any]:
    """Execute one immutable technical-only run after explicit coordinator admission."""
    clock = {"wall": time.perf_counter(), "cpu": time.process_time()}
    STUDY.mkdir(exist_ok=False)
    config: dict[str, Any] = {}
    sources: dict[str, str] = {}
    inputs: dict[str, str] = {}
    completed: list[int] = []
    prior_cpu = 0.0
    prior_gpu = 0.0
    forward_attempts = 0
    completed_forwards = 0

    def guard() -> None:
        check_limits(
            config,
            clock,
            prior_cpu,
            study_bytes=directory_bytes(STUDY),
            stream_bytes=directory_bytes(BASE),
            reserve=config["failure_reserve_seconds"],
        )

    try:
        config = json.loads(CONFIG.read_text())
        configure_cpu(config)
        guard()
        if file_sha256(PARENT, guard) != PARENT_SHA256:
            raise ValueError("Passed parent numerical audit binding changed")
        parent = json.loads(PARENT.read_text())
        if not parent["all_passed"] or parent["mismatch_count"] != 0:
            raise ValueError("Parent numerical audit did not pass")
        prior_cpu = parent["cumulative_cpu_seconds"]
        prior_gpu = parent["global_gpu_wall_seconds"]
        if prior_gpu >= config["gpu_seconds_ceiling"]:
            raise ValueError("Cumulative GPU-wall ceiling already exhausted")
        inputs[to_stored(PARENT)] = PARENT_SHA256
        sources = {name: file_sha256(ROOT / name, guard) for name in SOURCE_NAMES}
        if (
            directory_bytes(BASE)
            + config["output_bytes_ceiling"]
            + config["parallel_mode_diagnostic_reserve_bytes"]
            >= config["cache_bytes_ceiling"]
        ):
            raise ValueError("Whole-stream remaining study/diagnostic storage forecast fails")
        write_new_json(
            STUDY / "admission.json",
            {
                "source_sha256": sources,
                "input_sha256": inputs,
                "config": config,
                "prior_cpu_seconds": prior_cpu,
                "prior_gpu_wall_seconds": prior_gpu,
            },
        )
        assets = download_assets(config, inputs, guard)
        mit = assets["mit_pulse2pulse.py"].read_text()
        hf = assets["hf_modeling_deepfake.py"].read_text()
        architecture_proof = verify_architecture_equivalence(mit, hf)
        model = load_generator(assets[config["weight_filename"]], architecture_class(mit, hf))
        initial_state_sha256 = tensor_state_sha256(model)
        guard()
        noises = np.stack(
            [
                torch.empty((1, 8, 5000), dtype=torch.float32)
                .uniform_(-1, 1, generator=torch.Generator(device="cpu").manual_seed(seed))
                .numpy()
                for seed in config["seeds"]
            ]
        )
        profile_wall, profile_cpu = time.perf_counter(), time.process_time()
        forward_attempts += 1
        generate(model, noises[0])
        completed_forwards += 1
        profile_wall, profile_cpu = time.perf_counter() - profile_wall, time.process_time() - profile_cpu
        spent_wall, spent_cpu = time.perf_counter() - clock["wall"], time.process_time() - clock["cpu"]
        forecast_wall = (
            spent_wall
            + config["profile_forward_multiplier"] * profile_wall
            + config["report_reserve_seconds"]
        )
        forecast_cpu = (
            spent_cpu + config["profile_forward_multiplier"] * profile_cpu + config["report_reserve_seconds"]
        )
        write_new_json(
            STUDY / "profile.json",
            {
                "wall_seconds": profile_wall,
                "cpu_seconds": profile_cpu,
                "forecast_wall_seconds": forecast_wall,
                "forecast_cpu_seconds": forecast_cpu,
                "source_sha256": sources,
                "input_sha256": inputs,
            },
        )
        if max(forecast_wall, forecast_cpu) >= config["cpu_wall_limit_seconds"]:
            raise TimeoutError("Actual-forward conservative technical generation forecast fails")
        raw_bank, export_bank, canonical_bank, checks = [], [], [], []
        for seed, noise in zip(config["seeds"], noises, strict=True):
            guard()
            forward_attempts += 1
            raw, export, canonical = generate(model, noise)
            completed_forwards += 1
            guard()
            checks.append(output_checks(raw, export, canonical))
            raw_bank.append(raw)
            export_bank.append(export)
            canonical_bank.append(canonical)
            completed.append(seed)
        guard()
        forward_attempts += 1
        replay = generate(model, noises[0])
        completed_forwards += 1
        guard()
        replay_exact = all(
            np.array_equal(a, b)
            for a, b in zip(replay, [raw_bank[0], export_bank[0], canonical_bank[0]], strict=True)
        )
        distinct = all(not np.array_equal(raw_bank[i], raw_bank[j]) for i in range(4) for j in range(i))
        gates = {
            "all_case_checks": all(all(row.values()) for row in checks),
            "four_fixed_inputs": len(completed) == 4,
            "exact_replay": replay_exact,
            "distinct_fixed_seed_outputs": distinct,
            "strict_finite_tensor_loading": True,
            "architecture_equivalence": all(architecture_proof.values()),
            "model_state_unchanged": tensor_state_sha256(model) == initial_state_sha256,
        }
        for name, value in {
            "noise.npy": noises,
            "raw.npy": np.stack(raw_bank),
            "author_export.npy": np.stack(export_bank),
            "canonical12.npy": np.stack(canonical_bank),
        }.items():
            with (STUDY / name).open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
        guard()
        verify_snapshots(sources, inputs, guard)
        outputs = {p.name: file_sha256(p, guard) for p in STUDY.iterdir() if p.is_file()}
        result = {
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "technical_generation_feasible": all(gates.values()),
            "gates": gates,
            "case_checks": checks,
            "completed_seeds": completed,
            "architecture_proof": architecture_proof,
            "forward_attempts": forward_attempts,
            "completed_forwards": completed_forwards,
            "forward_allocation": {"profile": 1, "fixed_seed_bank": 4, "replay": 1},
            "model_buffers": 0,
            "model_state_sha256": initial_state_sha256,
            "model_parameters": sum(value.numel() for value in model.parameters()),
            "source_sha256": sources,
            "input_sha256": inputs,
            "output_sha256": outputs,
            "profile": {"cpu_seconds": profile_cpu, "wall_seconds": profile_wall},
            "prior_cpu_seconds": prior_cpu,
            "cpu_seconds": time.process_time() - clock["cpu"],
            "wall_seconds": time.perf_counter() - clock["wall"],
            "global_gpu_wall_seconds": prior_gpu,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "study_bytes_before_receipt": directory_bytes(STUDY),
            "stream_bytes_before_receipt": directory_bytes(BASE),
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
            "input_array_sha256": array_sha256(noises),
            "raw_array_sha256": array_sha256(np.stack(raw_bank)),
        }
        result["cumulative_cpu_seconds"] = prior_cpu + result["cpu_seconds"]
        check_receipt_reserve(config)
        guard()
        write_new_json(STUDY / "summary.json", result)
        return result
    except Exception as error:
        write_new_json(
            STUDY / "failure.json",
            {
                "source_sha256": sources,
                "input_sha256": inputs,
                "completed_seeds": completed,
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "forward_attempts": forward_attempts,
                "completed_forwards": completed_forwards,
                "cpu_seconds": time.process_time() - clock["cpu"],
                "wall_seconds": time.perf_counter() - clock["wall"],
                "prior_cpu_seconds": prior_cpu,
                "prior_gpu_wall_seconds": prior_gpu,
                "gpu_used": False,
                "clinical_performance_scored": False,
            },
        )
        raise RuntimeError("Technical generator probe stopped; retained local failure receipt") from None
