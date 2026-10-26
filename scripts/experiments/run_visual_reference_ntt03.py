"""Execute the frozen compact visual reference pilot locally on CPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment import visual_reference_ntt03 as visual
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored

EXPECTED_INPUTS = {
    "result.json": "5e7f4052b60d3bfb3ae589bb6cc482c32c7af49d6858d8cc5ae37e6cb044047b",
    "questions_local.json": "3cc05a82cc21732ba7c2046dc8ee8276947d133c4f7ae01e96b8d0b928f3a019",
    "predictions_local.json": "29c53ae3dfe6fcd6bb721e32147be58ae43a3e3294c8e651e5baf5b21c595007",
    "partition_local.json": "efaa51a62f52fc2d10881a58994d2da206ae5683cff08872c09cc5ec08ef253c",
}
SOURCES = (
    "ecg_experiment/visual_reference_ntt03.py",
    "scripts/experiments/run_visual_reference_ntt03.py",
    "tests/test_visual_reference_ntt03.py",
    "docs/ntt-03-visual-reference-pilot.md",
    "docs/ntt-03-amendment-selection.md",
    "ecg_experiment/reference_transfer_ntt02.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "pyproject.toml",
    "uv.lock",
)


def load_inputs() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, str]]:
    """Verify predecessor bytes and training identities without rerunning its label replay.

    Returns
    -------
    tuple
        Selected classes, contextual phase predictions and input hashes.
    """
    base = ROOT / "outputs/ntt02_native_reference_transfer"
    hashes = {to_stored(base / name): sha256_file(base / name) for name in EXPECTED_INPUTS}
    if any(hashes[to_stored(base / name)] != digest for name, digest in EXPECTED_INPUTS.items()):
        raise ValueError("Predecessor input changed")
    rows = json.loads((base / "questions_local.json").read_text())
    predecessor = json.loads((base / "result.json").read_text())
    if len(rows) != 1507:
        raise ValueError("Predecessor inventory count did not reproduce")
    manifest = json.loads((base / "partition_local.json").read_text())
    permitted = {int(row["ecg_id"]) for row in manifest["records"] if row["role"] == "train"}
    if any(row["ecg_id"] not in permitted for row in rows):
        raise ValueError("Reserved question in predecessor inventory")
    selected = visual.select_cases(rows)
    if not selected:
        raise ValueError("No class meets the prospective reference rule")
    targets = [row for group in selected for row in group["targets"]]
    if len({row["qid"] for row in targets}) != len(targets):
        raise ValueError("Repeated selected target question")
    predictions = json.loads((base / "predictions_local.json").read_text())
    phase = {
        row["qid"]: row
        for row in predictions
        if row["fold"] == 0 and row["seed"] == 20261018 and row["budget"] == 3
    }
    for group in selected:
        for row in [*group["supports"], *group["targets"]]:
            basepath = ROOT / "data/raw/ptb-xl/1.0.3" / row["waveform"]
            for suffix in (".hea", ".dat"):
                path = basepath.with_suffix(suffix)
                hashes[to_stored(path)] = sha256_file(path)
                if hashes[to_stored(path)] != predecessor["waveform_sha256"][to_stored(path)]:
                    raise ValueError("Selected waveform changed since the frozen predecessor")
    return selected, phase, hashes


def plot_proposal(
    output: Path, row: dict[str, Any], proposal: dict[str, Any], rgb: np.ndarray, mask: np.ndarray | None
) -> None:
    """Save the prospectively selected first-target temporal overlay locally.

    Parameters
    ----------
    output : Path
        Local figure destination.
    row : dict
        Selected target and nominated silver interval.
    proposal : dict
        SAM automatic proposals and reference choice.
    rgb : np.ndarray
        Full ten-second rendered signal.
    mask : np.ndarray or None
        Chosen proposal mask.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    axes[0].imshow(rgb, extent=(0, 10, -3, 3), aspect="auto")
    if mask is not None:
        axes[0].imshow(
            np.ma.masked_where(~mask, mask),
            extent=(0, 10, -3, 3),
            aspect="auto",
            alpha=0.3,
            cmap="Oranges",
            vmin=0,
            vmax=1,
        )
    truth = row["intervals"][row["answer"]]
    axes[0].axvspan(*truth, color="tab:blue", alpha=0.2, label="Nominated silver interval")
    axes[0].set(
        title=f"{row['diagnosis']} · supplied lead {row['lead']} · SAM proposals",
        xlabel="Seconds",
        ylabel="mV",
    )
    axes[0].legend(loc="upper right")
    signal = visual.physical_lead(row)
    axes[1].plot(np.arange(len(signal)) / 100, signal, color="black", linewidth=0.8)
    axes[1].axvspan(*truth, color="tab:blue", alpha=0.2)
    for interval in row["intervals"]:
        axes[1].axvline(interval[0], color="gray", linewidth=0.5)
        axes[1].axvline(interval[1], color="gray", linewidth=0.5)
    axes[1].set(
        xlim=(max(0, truth[0] - 0.6), min(10, truth[1] + 0.6)),
        xlabel="Seconds",
        ylabel="mV",
        title="Physical trace and supplied candidate boundaries",
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def evaluate(output: Path) -> dict[str, Any]:
    """Run the fixed cohort once, preserving every per-target result.

    Parameters
    ----------
    output : Path
        New local output directory.

    Returns
    -------
    dict
        Aggregate engineering result, stage times and provenance.
    """
    started = time.perf_counter()
    source_hashes = {}
    for name in SOURCES:
        committed = subprocess.run(
            ["git", "show", f"HEAD:{name}"], cwd=ROOT, check=True, capture_output=True
        ).stdout
        if committed != (ROOT / name).read_bytes():
            raise ValueError(f"Uncommitted source {name}")
        source_hashes[name] = sha256_file(ROOT / name)
    selected, phase, input_hashes = load_inputs()
    source_rows = json.loads(
        (ROOT / "outputs/ntt02_native_reference_transfer/questions_local.json").read_text()
    )
    lead_links = Counter(visual.linked_lead(row)[1] for row in source_rows)
    (output / "selection_local.json").write_text(json.dumps(selected, indent=2) + "\n")
    checkpoints = ROOT / "third_party/ntt_vision/checkpoints"
    for path in sorted(checkpoints.glob("*.pth")) + sorted(checkpoints.glob("*.pt")):
        input_hashes[to_stored(path)] = sha256_file(path)
    for path in sorted((ROOT / "third_party/ntt_vision/sources").rglob("*.py")):
        source_hashes[to_stored(path)] = sha256_file(path)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model, generator = visual.load_models()
    results, proposals, unprocessed = [], [], []
    for index, group in enumerate(selected):
        if time.perf_counter() - started >= 900:
            unprocessed.append(group["diagnosis"])
            continue
        try:
            tick = time.perf_counter()
            prototypes = {
                view: visual.class_prototypes(model, group["supports"], view, started + 900)
                for view in (False, True)
            }
            widths = [np.diff(row["intervals"][row["answer"]])[0] for row in group["supports"]]
            for row in group["targets"]:
                for view in (False, True):
                    result = visual.match_candidates(model, row, prototypes[view], view, started + 900)
                    candidate_widths = np.diff(np.asarray(row["intervals"]), axis=1).ravel()
                    duration = int(np.argmin(np.abs(np.log(candidate_widths / np.median(widths)))))
                    result.update(
                        diagnosis=group["diagnosis"],
                        duration_choice=duration,
                        duration_accuracy=float(duration == row["answer"]),
                        phase_context_accuracy=phase[row["qid"]]["accuracy"]["qrs_phase"],
                    )
                    results.append(result)
            target = group["targets"][0]
            proposal, rgb, mask = visual.proposal_trial(
                model, generator, target, prototypes[False][0], started + 900
            )
            proposal.update(diagnosis=group["diagnosis"], class_seconds=time.perf_counter() - tick)
            proposals.append(proposal)
            plot_proposal(output / f"proposal_{index}.png", target, proposal, rgb, mask)
            (output / "progress_local.json").write_text(
                json.dumps({"candidate_results": results, "proposal_results": proposals}, indent=2) + "\n"
            )
        except TimeoutError:
            unprocessed.extend(item["diagnosis"] for item in selected[index:])
            break
    targets = [row for group in selected for row in group["targets"]]
    return {
        "study": "NTT-03",
        "status": "partial_budget_stop" if unprocessed else "complete",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "model_variant": "DINOv2-S14 plus SAM2-Tiny; not original large/large recipe",
        "classes": len(selected),
        "target_questions": len(targets),
        "unique_target_patients": len({row["patient_id"] for row in targets}),
        "lead_link_inventory": dict(lead_links),
        "candidate_results": results,
        "proposal_results": proposals,
        "unprocessed_classes": unprocessed,
        "seconds": time.perf_counter() - started,
        "gpu_used": False,
        "training_performed": False,
        "reserved_patients_used": False,
        "phase_comparator": "NTT02 all12leads and its separate fixed support draw; contextual only",
        "clinical_localization_accuracy": None,
        "primary_visual_view": "one_second_candidate_crop",
        "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "pillow")},
    }


def main() -> None:
    """Preserve completion/failure evidence for one source-frozen CPU run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ntt03_visual_reference")
    args = parser.parse_args()
    to_stored(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "threads": 2, "gpu": False}
    status_path = args.output / "status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        result = evaluate(args.output)
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        raise
    status.update(status=result["status"], result_sha256=sha256_file(args.output / "result.json"))
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(
        json.dumps({"status": result["status"], "classes": result["classes"], "seconds": result["seconds"]})
    )


if __name__ == "__main__":
    main()
