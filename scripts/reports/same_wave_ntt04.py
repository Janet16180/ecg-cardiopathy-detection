"""Run the frozen NTT-04 finding-discrimination comparison on cached public ECGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import wfdb
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.eda.signals import canonical_order
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.same_wave_ntt04 import (
    FINDINGS,
    METHODS,
    classify,
    image_descriptors,
    labeled_cohort,
    load_dino,
    physical_median,
    raw_descriptors,
    select_patients,
    summarize,
    verify_reservations,
)
from ecg_experiment.visual_reference_ntt03 import DINO_REVISION, interval_iou

matplotlib.use("Agg")

SOURCES = (
    "ecg_experiment/same_wave_ntt04.py",
    "scripts/reports/same_wave_ntt04.py",
    "tests/test_same_wave_ntt04.py",
    "docs/ntt-04-same-wave-finding-discrimination.md",
    "docs/ntt-04-amendment-context-control.md",
    "ecg_experiment/visual_reference_ntt03.py",
    "ecg_experiment/reference_transfer_ntt02.py",
    "ecg_experiment/clustering01.py",
    "ecg_experiment/eda/avance2_features.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/__init__.py",
    "pyproject.toml",
    "uv.lock",
)
PREDECESSOR_SHA = "aa9b76d825a9aeff067d20521dc46d4013a637b8601c04573978fa3c3cdbce5c"


def write_json(path: Path, value: Any) -> None:
    """Write a strict JSON artifact without silent nonfinite values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frozen_sources() -> dict[str, str]:
    """Require committed source bytes and bind the cached external DINO Python tree."""
    hashes = {}
    for name in SOURCES:
        committed = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, check=True, capture_output=True)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"Uncommitted source: {name}")
        hashes[name] = digest
    external = ROOT / f"third_party/ntt_vision/sources/dinov2-{DINO_REVISION}"
    previous = ROOT / "outputs/ntt03_visual_reference/result.json"
    if sha256_file(previous) != PREDECESSOR_SHA:
        raise ValueError("Predecessor source-binding receipt changed")
    prefix = to_stored(external) + "/"
    expected_external = {
        name: digest
        for name, digest in json.loads(previous.read_text())["source_sha256"].items()
        if name.startswith(prefix) and name.endswith(".py")
    }
    actual_external = {}
    for path in sorted(external.rglob("*.py")):
        actual_external[to_stored(path)] = sha256_file(path)
    if not actual_external or actual_external != expected_external:
        raise ValueError("Cached DINO Python files differ from the frozen NTT-03 source")
    hashes.update(actual_external)
    return hashes


def reproduce_ntt03(manifest_path: Path) -> dict:
    """Rebuild saved predecessor counts without model inference or a broad clinical replay."""
    base = ROOT / "outputs/ntt03_visual_reference"
    result_path, selection_path = base / "result.json", base / "selection_local.json"
    if sha256_file(result_path) != PREDECESSOR_SHA:
        raise ValueError("Predecessor result changed")
    selected = json.loads(selection_path.read_text())
    roles = json.loads(manifest_path.read_text())["records"]
    permitted = {int(row["ecg_id"]) for row in roles if row["role"] == "train"}
    if any(
        row["ecg_id"] not in permitted for group in selected for row in group["supports"] + group["targets"]
    ):
        raise ValueError("Reserved predecessor source record")
    result = json.loads(result_path.read_text())
    views = {}
    for name, cropped in (("full", False), ("crop", True)):
        rows = [row for row in result["candidate_results"] if row["cropped"] == cropped]
        if len(rows) != 6:
            raise ValueError("Predecessor target count differs")
        correct = [
            sum(int(np.argmax(row[f"{kind}_scores"])) == row["answer"] for row in rows)
            for kind in ("real", "wrong")
        ]
        views[name] = correct
    targets = {row["qid"]: row for group in selected for row in group["targets"]}
    oracles = []
    for proposal in result["proposal_results"]:
        row = targets[proposal["qid"]]
        oracle = max(
            interval_iou(c["interval"], row["intervals"][row["answer"]]) for c in proposal["candidates"]
        )
        if abs(oracle - proposal["oracle_iou"]) > 1e-12:
            raise ValueError("Predecessor proposal oracle does not reproduce")
        oracles.append(oracle)
    if views != {"full": [6, 2], "crop": [6, 0]} or len(oracles) != 3:
        raise ValueError("Predecessor candidate counts do not reproduce")
    return {
        "candidate_correct_real_wrong": views,
        "targets": 6,
        "proposal_oracles": oracles,
        "saved_results_reproduced": True,
        "input_sha256": {to_stored(p): sha256_file(p) for p in (result_path, selection_path)},
    }


def extract_physical(rows: list[dict], output: Path, deadline: float) -> tuple[dict, dict, dict]:
    """Read and quality-check every fixed selected waveform before visual/classification scores."""
    features, beats, hashes = {}, {}, {}
    for row in rows:
        if time.perf_counter() >= deadline:
            raise TimeoutError("NTT-04 budget exhausted before physical extraction")
        path = ROOT / "data/raw/ptb-xl/1.0.3" / row["filename_hr"]
        try:
            for suffix in (".hea", ".dat"):
                source = path.with_suffix(suffix)
                hashes[to_stored(source)] = sha256_file(source)
            signal, fields = wfdb.rdsamp(str(path))
            if fields["fs"] != 500 or set(fields["units"]) != {"mV"}:
                raise ValueError("Expected 500 Hz physical mV recording")
            beat, quality = physical_median(canonical_order(signal, fields["sig_name"]))
            raw, full, scalars = raw_descriptors(beat, quality["qrs_duration_seconds"])
        except (ValueError, OSError) as error:
            write_json(
                output / "quality_failure_local.json",
                {"ecg_id": row["ecg_id"], "error": str(error), "replacement": False},
            )
            raise
        features[row["ecg_id"]] = {
            "raw": raw,
            "raw_full_context": full,
            "scalars": scalars,
            "quality": quality,
        }
        beats[row["ecg_id"]] = beat
    return features, beats, hashes


def figures(summary: dict, predictions: list[dict], beat: np.ndarray, output: Path) -> None:
    """Save aggregate classification bars/confusions and the fixed first local median-beat example."""
    import matplotlib.pyplot as plt

    from ecg_experiment.same_wave_ntt04 import render_median

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    values = [summary["balanced_accuracy"][method] for method in METHODS]
    axis.bar(METHODS, values)
    axis.axhline(1 / 3, color="black", linestyle="--", linewidth=0.8)
    axis.set(
        ylabel="Balanced record-finding accuracy",
        ylim=(0, 1),
        title="Fixed 60 targets; mean over three reference draws",
    )
    axis.tick_params(axis="x", rotation=25)
    figure.savefig(output / "aggregate_accuracy.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, method in zip(axes, ("dino", summary["best_baseline"]), strict=True):
        matrix = np.zeros((3, 3))
        for row in predictions:
            matrix[row["truth"], row["chosen"][method]] += 1
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
        axis.set_xticks(range(3), FINDINGS, rotation=20)
        axis.set_yticks(range(3), FINDINGS)
        axis.set(title=method, xlabel="Reference-class choice", ylabel="Source annotation")
    figure.savefig(output / "aggregate_confusion.png", dpi=160)
    plt.close(figure)
    rgb, qrs, late, coordinates = render_median(beat)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, mask, title in zip(
        axes, (qrs, late), ("Programmed QRS pooling", "Late-reference control"), strict=True
    ):
        axis.imshow(rgb)
        overlay = np.zeros((518, 518, 4))
        overlay[:, :, 1], overlay[:, :, 3] = 0.8, (mask > 0) * 0.9
        axis.imshow(overlay)
        axis.set(title=title)
        axis.axis("off")
    figure.suptitle("Fixed first target: V1 above V6, same physical scale for every finding")
    figure.savefig(output / "fixed_example_local.png", dpi=160)
    plt.close(figure)
    write_json(output / "fixed_example_coordinates_local.json", coordinates)


def run(output: Path, manifest: Path, policy: Path) -> dict:
    """Execute one score-frozen patient-disjoint reference comparison."""
    started = time.perf_counter()
    sources = frozen_sources()
    for original, name in (
        (manifest, "reservation_manifest_local.json"),
        (policy, "reservation_policy_local.json"),
    ):
        (output / name).write_bytes(original.read_bytes())
    metadata = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
    first, reservation = verify_reservations(
        metadata, output / "reservation_manifest_local.json", output / "reservation_policy_local.json"
    )
    replay = reproduce_ntt03(output / "reservation_manifest_local.json")
    cohort, counts = labeled_cohort(first, metadata)
    targets, manifests, inventory = select_patients(cohort)
    selected = {row["ecg_id"]: row for row in targets}
    for draw in manifests:
        selected.update({row["ecg_id"]: row for row in draw["supports"]})
    rows = sorted(selected.values(), key=lambda row: row["ecg_id"])
    write_json(output / "cohort_local.json", cohort)
    write_json(output / "selection_local.json", {"targets": targets, "support_draws": manifests})
    features, beats, wave_hashes = extract_physical(rows, output, started + 600)
    model = load_dino(ROOT / "third_party/ntt_vision")
    for row in rows:
        dino, late, coordinates = image_descriptors(model, beats[row["ecg_id"]], started + 600)
        features[row["ecg_id"]].update(dino=dino, late=late, coordinates=coordinates)
    if time.perf_counter() >= started + 600:
        raise TimeoutError("NTT-04 budget exhausted before classification")
    predictions = classify(features, targets, manifests)
    summary = summarize(predictions)
    write_json(output / "predictions_local.json", predictions)
    serialized = {
        str(ecg): {
            key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in item.items()
        }
        for ecg, item in features.items()
    }
    write_json(output / "features_local.json", serialized)
    figures(summary, predictions, beats[targets[0]["ecg_id"]], output)
    checkpoint = ROOT / "third_party/ntt_vision/checkpoints/dinov2_vits14_pretrain.pth"
    inputs = {
        to_stored(path): sha256_file(path)
        for path in (
            metadata,
            metadata.with_name("scp_statements.csv"),
            output / "reservation_manifest_local.json",
            output / "reservation_policy_local.json",
            checkpoint,
        )
    }
    inputs.update(wave_hashes)
    inputs.update(replay["input_sha256"])
    quality = [item["quality"] for item in features.values()]
    clipping = [item["coordinates"]["clipped_fraction"] for item in features.values()]
    return {
        "study": "NTT-04",
        "status": "complete_same_wave_finding_discrimination",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sources,
        "input_sha256": inputs,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "predecessor_reproduction": replay,
        "reservation_integrity": reservation,
        "cohort_counts": counts,
        "finding_inventory": inventory,
        "targets": 60,
        "target_patients": len({r["patient_id"] for r in targets}),
        "unique_selected_waveforms": len(features),
        "unique_support_patients": len({r["patient_id"] for draw in manifests for r in draw["supports"]}),
        "selected_quality": {
            "complete_beats_min_median_max": np.percentile(
                [q["complete_beats"] for q in quality], [0, 50, 100]
            ).tolist(),
            "consistency_min_median_max": np.percentile(
                [q["beat_consistency"] for q in quality], [0, 50, 100]
            ).tolist(),
            "clipped_fraction_max": max(clipping),
        },
        "annotation_cooccurrence": {
            label: dict(
                Counter(
                    code
                    for row in targets
                    if row["finding"] == label
                    for code in row["diagnostic_superclasses"]
                )
            )
            for label in FINDINGS
        },
        "summary": summary,
        "gpu_used": False,
        "weight_training": False,
        "provided_qa_windows_or_leads_used": False,
        "clinical_localization_accuracy": None,
        "model": "frozen DINOv2-S14; full median-beat context, programmed QRS-ink pooling",
        "versions": {name: version(name) for name in ("torch", "numpy", "scipy", "wfdb", "pillow")},
        "seconds": time.perf_counter() - started,
        "artifact_sha256": {
            to_stored(path): sha256_file(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "status.json"
        },
    }


def main() -> None:
    """Run once into a fresh output directory and preserve a failed execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ntt04_same_wave")
    parser.add_argument("--manifest", type=Path, default=Path("/tmp/ecg-shared-research-partition.json"))
    parser.add_argument("--policy", type=Path, default=Path("/tmp/ecg-shared-research-policy.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "running",
        "command": ["python", "-m", "scripts.reports.same_wave_ntt04", "--output", to_stored(args.output)],
        "thread_environment": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(args.output / "status.json", status)
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        with threadpool_limits(limits=2):
            result = run(args.output, args.manifest, args.policy)
        result.update(command=status["command"], thread_environment=status["thread_environment"])
        write_json(args.output / "result.json", result)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__)
        write_json(args.output / "status.json", status)
        raise
    status.update(status="complete", result_sha256=sha256_file(args.output / "result.json"))
    write_json(args.output / "status.json", status)
    print(json.dumps({"seconds": result["seconds"], "summary": result["summary"]}))


if __name__ == "__main__":
    main()
