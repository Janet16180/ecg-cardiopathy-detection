"""Profile and evaluate frozen native ECG reference matching without image inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import wfdb
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.eda.signals import canonical_order
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.localization.jepa_native import (
    CHECKPOINT_SHA256,
    CONTEXT_SCOPE,
    EMBEDDING_SPACE,
    NATIVE_LEADS,
    SOURCE_REVISION,
    encode_signal,
    load_encoder,
    preprocess_signal,
)
from ecg_experiment.localization.native_study import (
    automatic_regions,
    descriptors,
    native_predictions,
    native_summary,
    predecessor_inputs,
    region_matches,
    regions_from_saved,
    reproduce_predecessor,
    validate_profile,
)
from ecg_experiment.paths import to_stored

SOURCE_DIR = ROOT / f"third_party/ntt_native/sources/ECG_JEPA-{SOURCE_REVISION}"
CHECKPOINT = ROOT / "third_party/ntt_native/checkpoints/ecg-jepa-multiblock-epoch100.pth"
PROTOCOL = "docs/research/localization/ntt06-native-encoder/protocol.md"
SOURCES = (
    PROTOCOL,
    "ecg_experiment/localization/jepa_native.py",
    "ecg_experiment/localization/native_study.py",
    "scripts/experiments/localization/run_ntt06.py",
    "tests/localization/test_jepa_native.py",
    "tests/localization/test_native_study.py",
    "ecg_experiment/native_reference_ntt05.py",
    "ecg_experiment/same_wave_ntt04.py",
    "ecg_experiment/clustering01.py",
    "ecg_experiment/visual_reference_ntt03.py",
    "ecg_experiment/reference_transfer_ntt02.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/localization/__init__.py",
    "ecg_experiment/eda/avance2_features.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/__init__.py",
    "pyproject.toml",
    "uv.lock",
)


def frozen_sources() -> dict[str, str]:
    """Bind committed implementation bytes, including the exact external architecture."""
    hashes = {}
    for name in SOURCES:
        path = ROOT / name
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        digest = sha256_file(path)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"Uncommitted experiment source: {name}")
        hashes[name] = digest
    for name in ("ecg_jepa.py", "pos_encoding.py", "models.py"):
        hashes[to_stored(SOURCE_DIR / name)] = sha256_file(SOURCE_DIR / name)
    return hashes


def runtime_fingerprint() -> dict:
    """Bind installed numerical libraries and the CPU execution environment."""
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "numpy", "scipy", "timm", "wfdb")},
        "device": "cpu",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def budget(deadline: float) -> None:
    """Stop at a stage boundary when the prospectively declared deadline is exceeded."""
    if time.perf_counter() > deadline:
        raise TimeoutError("Prospective native inference time budget exceeded")


def profile(output: Path, sources: dict) -> dict:
    """Measure actual pretrained CPU inference on a synthetic non-diagnostic input."""
    started = time.perf_counter()
    deadline = started + 300
    load_start = time.perf_counter()
    encoder = load_encoder(SOURCE_DIR, CHECKPOINT)
    load_seconds = time.perf_counter() - load_start
    budget(deadline)
    signal = np.random.default_rng(20261019).normal(0, 0.1, (12, 5000)).astype(np.float32)
    elapsed, tokens = [], []
    for index in range(2):
        tick = time.perf_counter()
        tokens.append(encode_signal(encoder, signal, f"synthetic-profile-{index}"))
        elapsed.append(time.perf_counter() - tick)
        budget(deadline)
    tensor = torch.from_numpy(preprocess_signal(signal))[None]
    tick = time.perf_counter()
    with torch.inference_mode():
        pooled = encoder.representation(tensor).cpu().numpy()[0]
    pooled_seconds = time.perf_counter() - tick
    budget(deadline)
    mean_error = float(np.max(np.abs(tokens[0].values.mean(axis=0) - pooled)))
    repeat_error = float(np.max(np.abs(tokens[0].values - tokens[1].values)))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gates = {
        "finite_dense_shape": tokens[0].values.shape == (400, 768)
        and bool(np.isfinite(tokens[0].values).all()),
        "official_mean_equivalence": mean_error <= 1e-5,
        "deterministic_repeat": repeat_error <= 1e-6,
        "forward_budget": max(*elapsed, pooled_seconds) <= 20,
        "rss_budget": rss <= 6 * 1024**3,
        "profile_budget": time.perf_counter() - started <= 300,
        "all_parameters_frozen": all(not p.requires_grad for p in encoder.parameters())
        and not encoder.training,
    }
    result = {
        "study": "NTT-06",
        "stage": "synthetic_pretrained_cpu_profile",
        "profile_gate_passed": all(gates.values()),
        "gates": gates,
        "load_seconds": load_seconds,
        "forward_seconds": elapsed,
        "official_pooled_seconds": pooled_seconds,
        "mean_absolute_max_error": mean_error,
        "repeat_absolute_max_error": repeat_error,
        "peak_rss_bytes": rss,
        "parameters": sum(p.numel() for p in encoder.parameters()),
        "dense_shape": list(tokens[0].values.shape),
        "seconds": time.perf_counter() - started,
        "waveform_reads": 0,
        "clinical_score": False,
        "source_sha256": sources,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime": runtime_fingerprint(),
    }
    write_json_atomic(output / "result.json", result)
    if not all(gates.values()):
        raise RuntimeError("Native pretrained CPU profile did not pass all prospective gates")
    return result


def plot_local(signal: np.ndarray, regions: list[dict], output: Path) -> None:
    """Write an output-only waveform overlay for the prospectively fixed first target."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True, constrained_layout=True)
    time_axis = np.arange(5000) / 500
    colors = {"NORM": "gray", "CRBBB": "tab:orange", "CLBBB": "tab:blue"}
    for ax, lead, column in zip(axes, ("V1", "V6"), (6, 11), strict=True):
        ax.plot(time_axis, signal[:, column], color="black", linewidth=0.65)
        for row in regions:
            if row["proposal"] == "automatic_qrs":
                ax.axvspan(row["start_seconds"], row["stop_seconds"], color=colors[row["finding"]], alpha=0.2)
        ax.set(ylabel=f"{lead} (mV)")
    axes[-1].set(xlabel="Original recording time (seconds)", xlim=(0, 10))
    figure.suptitle("Contextual nearest-prototype matches; 200 ms anchors, no expert localization validation")
    figure.savefig(output / "fixed_first_target_local.png", dpi=150)
    plt.close(figure)


def evaluate(output: Path, sources: dict, profile_path: Path) -> dict:
    """Run the single frozen development comparison after the real CPU profile gate."""
    started = time.perf_counter()
    deadline = started + 1800
    saved_profile = json.loads(profile_path.read_text())
    validate_profile(saved_profile, sources, runtime_fingerprint(), CHECKPOINT_SHA256)
    profile_status = profile_path.with_name("status.json")
    status = json.loads(profile_status.read_text())
    if status.get("status") != "completed" or status.get("result_sha256") != sha256_file(profile_path):
        raise ValueError("Profile final status does not bind these result bytes")
    selected, old_features, previous, old_result, inputs = predecessor_inputs()
    inputs[to_stored(profile_path)] = sha256_file(profile_path)
    inputs[to_stored(profile_status)] = sha256_file(profile_status)
    integrity = reproduce_predecessor(selected, old_features, previous, old_result)
    budget(deadline)
    encoder = load_encoder(SOURCE_DIR, CHECKPOINT)
    rows = {r["ecg_id"]: r for r in selected["targets"]}
    rows.update({r["ecg_id"]: r for g in selected["support_draws"] for r in g["supports"]})
    if len(rows) != 87:
        raise ValueError("Unexpected selected waveform count")
    features, all_tokens, coordinates, hashes, forwards = {}, {}, {}, {}, []
    first_signal = None
    first_ecg = selected["targets"][0]["ecg_id"]
    for index, (ecg, row) in enumerate(sorted(rows.items())):
        budget(deadline)
        path = ROOT / "data/raw/ptb-xl/1.0.3" / row["filename_hr"]
        for suffix in (".hea", ".dat"):
            part = path.with_suffix(suffix)
            digest = sha256_file(part)
            if digest != old_result["input_sha256"].get(to_stored(part)):
                raise ValueError("Selected waveform differs from frozen predecessor")
            hashes[to_stored(part)] = digest
        signal, fields = wfdb.rdsamp(str(path))
        if fields["fs"] != 500 or set(fields["units"]) != {"mV"}:
            raise ValueError("Expected 500Hz physical mV data")
        signal = canonical_order(signal, fields["sig_name"])
        qrs, late = automatic_regions(signal)
        tick = time.perf_counter()
        tokens = encode_signal(encoder, signal.T, f"ptbxl:{ecg}:original-10s")
        forwards.append(time.perf_counter() - tick)
        features[ecg] = descriptors(tokens, qrs, late)
        all_tokens[ecg] = tokens
        coordinates[ecg] = {"qrs": [asdict(r) for r in qrs], "late": [asdict(r) for r in late]}
        if ecg == first_ecg:
            first_signal = signal
        write_json_atomic(
            output / "status.json",
            {"status": "extracting_native_tokens", "completed": index + 1, "total": 87},
        )
        print(f"Native signal features: {index + 1}/87", flush=True)
    budget(deadline)
    predictions, banks = native_predictions(features, selected, previous, EMBEDDING_SPACE)
    summary = native_summary(predictions)
    matches, total, qrs_count = {}, 0, 0
    for target in selected["targets"]:
        ecg = target["ecg_id"]
        tokens = all_tokens[ecg]
        qrs = regions_from_saved(coordinates[ecg]["qrs"])
        matches[ecg] = {}
        for seed, bank in banks.items():
            matched = region_matches(tokens, qrs, bank, str(target["patient_id"]))
            matches[ecg][str(seed)] = matched
            total += len(matched)
            qrs_count += len(qrs)
        budget(deadline)
    selected_path = output / "selection_local.json"
    write_json_atomic(selected_path, selected)
    write_json_atomic(output / "predictions_local.json", predictions)
    write_json_atomic(output / "region_matches_local.json", matches)
    write_json_atomic(output / "automatic_coordinates_local.json", coordinates)
    ecgs = sorted(rows)
    np.savez_compressed(
        output / "dense_tokens_local.npz",
        ecg_ids=np.array(ecgs),
        values=np.stack([all_tokens[e].values for e in ecgs]),
    )
    write_json_atomic(
        output / "feature_vectors_local.json",
        {str(e): {m: v.tolist() for m, v in row.items()} for e, row in features.items()},
    )
    plot_local(first_signal, matches[first_ecg][str(min(banks))], output)
    budget(deadline)
    target_features = [features[r["ecg_id"]] for r in selected["targets"]]
    agreements = {
        m: {"mean": float(np.mean(v)), "minimum": float(min(v)), "maximum": float(max(v))}
        for m in ("native_late", "native_global_pair", "native_global_eight")
        for v in [[float(f["native_qrs"] @ f[m]) for f in target_features]]
    }
    result = {
        "study": "NTT-06",
        "stage": "native_development_comparison",
        "status": "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sources,
        "input_sha256": inputs,
        "waveform_sha256": hashes,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "predecessor_reproduction": integrity,
        "profile_gate_passed": True,
        "targets": 60,
        "support_patients": 27,
        "unique_waveforms": 87,
        "support_draws": 3,
        "summary": summary,
        "target_vector_cosines": agreements,
        "regional_candidates": total,
        "automatic_qrs_candidates": qrs_count,
        "candidate_assignments_are_calibrated": False,
        "runtime": runtime_fingerprint(),
        "embedding_space": EMBEDDING_SPACE,
        "context_scope": CONTEXT_SCOPE,
        "native_leads": list(NATIVE_LEADS),
        "token_anchor_seconds": 0.2,
        "clinical_localization_accuracy": None,
        "pretraining_eligibility": (
            "Documented Chapman/Ningbo and CODE15 sources exclude PTB; "
            "exact checkpoint patient manifest unavailable."
        ),
        "coexisting_annotations": old_result["annotation_cooccurrence"],
        "inference_seconds": {
            "total": sum(forwards),
            "mean": float(np.mean(forwards)),
            "maximum": max(forwards),
        },
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_used": False,
        "weight_training": False,
        "image_model_input": False,
        "artifact_sha256": {
            to_stored(p): sha256_file(p)
            for p in sorted(output.iterdir())
            if p.name not in ("status.json", "result.json", "failure.json")
        },
    }
    write_json_atomic(output / "result.json", result)
    return result


def main() -> None:
    """Execute one immutable profile or development stage and preserve typed failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if args.stage == "evaluate" and args.profile is None:
        parser.error("evaluation requires a passed --profile receipt")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    write_json_atomic(output / "status.json", {"status": "running", "stage": args.stage})
    try:
        sources = frozen_sources()
        with threadpool_limits(limits=2):
            result = (
                profile(output, sources)
                if args.stage == "profile"
                else evaluate(
                    output, sources, args.profile if args.profile.is_absolute() else ROOT / args.profile
                )
            )
        write_json_atomic(
            output / "status.json",
            {"status": "completed", "result_sha256": sha256_file(output / "result.json")},
        )
        print(
            json.dumps({"stage": args.stage, "status": "completed", "seconds": result["seconds"]}), flush=True
        )
    except Exception as error:
        write_json_atomic(
            output / "failure.json",
            {"type": type(error).__name__, "message": str(error), "stage": args.stage},
        )
        write_json_atomic(output / "status.json", {"status": "failed", "stage": args.stage})
        raise


if __name__ == "__main__":
    main()
