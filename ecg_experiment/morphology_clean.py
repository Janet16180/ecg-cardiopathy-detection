#!/usr/bin/env python3
"""Clean-transfer and second-seed Experiment 017 development replication."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F  # noqa: N812 - conventional alias

from ecg_experiment.cpc import CPCClassifier
from ecg_experiment.cpc_morphology import TEMPLATES, MorphologyCPCClassifier
from ecg_experiment.evaluation import LIMITED_LABELS
from ecg_experiment.files import (
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_npz_atomic,
    write_torch_atomic,
)
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.morphology_clean_inputs import (
    ARTIFACTS,
    CACHE,
    CANONICAL,
    CLEAN,
    DROPOUT_SEED_OFFSET,
    ENCODER_LR,
    EPOCHS,
    FULL_CLEAN_LABELS,
    HEAD_LR,
    INITIAL_BEST,
    KINDS,
    MANIFEST,
    NORMALIZATION,
    OUTPUT,
    PLANNING_GATE_SECONDS,
    POOL_VERIFICATION,
    REQUIRED_GAIN,
    ROOT,
    SEED,
    SEEDS,
    SENSITIVITY_TOLERANCE,
    SSL,
    WEIGHT_DECAY,
    PilotData,
    clean_exposed_labels,
    load_inputs,
    preload,
    require_clean_transform,
    verify_clean_transform,
)
from ecg_experiment.pilot import (
    BUDGETS,
    DEFAULT_MAX_CACHE_BYTES,
    DEFAULT_RESERVE_BYTES,
    INTERRUPTED_EXIT,
    MIN_CACHE_BYTES,
    SAVE_EVERY,
    Progress,
    check_roundtrip,
    check_waveform_sample,
    clipped_sigmoid,
    development_batches,
    fixed_batches,
    fold_operating_point,
    install_stop_handler,
    interrupted,
    labels_and_patients,
    load_state,
    normalized_batch,
    profile_arms,
    save_state,
)
from ecg_experiment.receipts import check_completion, check_existing_config, require_receipt, write_completion
from ecg_experiment.reproducibility import capture_rng_state, cpu_state, seed_everything
from ecg_experiment.training import checked_step, parameter_count


def _prefixed(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}


def make_model(ssl: Path, kind: str, bank: torch.Tensor | None, device: str,
               seed: int) -> MorphologyCPCClassifier:
    """
    Build one arm from the Experiment 004 encoder with a head shared across arms.

    Parameters
    ----------
    ssl : Path
        Experiment 004 ``encoder.pt``.
    kind : str
        ``"none"``, ``"conv"`` or ``"template"``.
    bank : torch.Tensor | None
        Initial template bank, unused by ``"none"``.
    device : str
        Target device.

    Returns
    -------
    MorphologyCPCClassifier
        Model whose first training dropout mask is common to every arm.

    Raises
    ------
    ValueError
        If the checkpoint's encoder keys differ from the original CPC encoder.
    """
    seed_everything(seed)
    # Construct the common original classifier first: adding a branch must not
    # shift the classifier-head RNG stream across comparison arms.
    reference = CPCClassifier()
    reference_head = cpu_state(reference.head)
    encoder_keys = set(reference.encoder.state_dict())
    model = MorphologyCPCClassifier(kind, bank if kind != "none" else None)
    model.head.load_state_dict(reference_head)
    saved = torch.load(ssl, map_location="cpu", weights_only=True)
    model.encoder.convs.load_state_dict(_prefixed(saved["encoder"], "convs."), strict=True)
    model.encoder.context.load_state_dict(_prefixed(saved["encoder"], "context."), strict=True)
    if set(saved["encoder"]) != encoder_keys:
        raise ValueError("Bootstrap encoder source keys changed")
    model = model.to(device)
    # The first training dropout mask is also common across arms. Resume state
    # overrides this with its exact saved RNG state.
    seed_everything(seed + DROPOUT_SEED_OFFSET)
    return model


def optimizer_for(model: MorphologyCPCClassifier) -> torch.optim.AdamW:
    """
    AdamW with a lower rate for the pretrained encoder than for new parameters.

    Parameters
    ----------
    model : MorphologyCPCClassifier
        Arm to optimize.

    Returns
    -------
    torch.optim.AdamW
        Optimizer over encoder, head and any morphology branch.
    """
    encoder = model.encoder
    base = list(encoder.convs.parameters()) + list(encoder.context.parameters())
    groups = [{"params": base, "lr": ENCODER_LR}, {"params": model.head.parameters(), "lr": HEAD_LR}]
    if encoder.kind != "none":
        groups.append({"params": [encoder.bank, *encoder.branch_hidden.parameters(),
                                  *encoder.branch_final.parameters()], "lr": HEAD_LR})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def rng_matches(saved: dict[str, Any]) -> bool:
    """
    Compare the active global random states with a saved checkpoint's.

    Parameters
    ----------
    saved : dict[str, Any]
        States from ``capture_rng_state``.

    Returns
    -------
    bool
        True when every generator state is identical.
    """
    active = capture_rng_state()
    return (active["python"] == saved["python"]
            and active["numpy"][0] == saved["numpy"][0]
            and np.array_equal(active["numpy"][1], saved["numpy"][1])
            and active["numpy"][2:] == saved["numpy"][2:]
            and torch.equal(active["torch"], saved["torch"])
            and len(active["cuda"]) == len(saved["cuda"])
            and all(torch.equal(a, b) for a, b in zip(active["cuda"], saved["cuda"], strict=True)))


@torch.inference_mode()
def predict_development(model: MorphologyCPCClassifier, data: PilotData, device: str) -> np.ndarray:
    """
    Compute development logits in manifest order.

    Parameters
    ----------
    model : MorphologyCPCClassifier
        Arm to evaluate.
    data : PilotData
        Preloaded inputs; development rows follow training rows.
    device : str
        Device holding ``model``.

    Returns
    -------
    np.ndarray
        Float64 development logits.
    """
    model.eval()
    logits = []
    for x in development_batches(data.waveforms, len(data.partitions.full), len(data.partitions.development),
                                 data.mean, data.std, device):
        logits.extend(model(x).cpu().numpy().tolist())
    return np.asarray(logits, dtype=np.float64)


def development_screen(rows: list[dict[str, str]], logits: np.ndarray) -> dict[str, Any]:
    """
    Development AUROC and cross-fitted patient-fold operating point.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Development rows.
    logits : np.ndarray
        Development logits.

    Returns
    -------
    dict[str, Any]
        AUROC, pooled fold sensitivity and specificity, and confusion counts.
    """
    labels, groups = labels_and_patients(rows)
    probabilities = clipped_sigmoid(logits)
    return {"auroc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            **fold_operating_point(labels, groups, probabilities, SEED)}


def identity(data: PilotData, budget: str, kind: str) -> dict[str, Any]:
    """
    Everything that determines one arm's result.

    Parameters
    ----------
    data : PilotData
        Preloaded inputs.
    budget : str
        Label budget.
    kind : str
        Arm.

    Returns
    -------
    dict[str, Any]
        Identity whose digest fingerprints the arm.
    """
    return {"provenance": data.provenance, "budget": budget, "arm": kind,
            "optimization_seed": data.seed,
            "train_ids": [row["ecg_id"] for row in data.partitions.full],
            "development_ids": [row["ecg_id"] for row in data.partitions.development],
            "template_selection": data.template_receipt}


def train_epoch(args: argparse.Namespace, data: PilotData, model: MorphologyCPCClassifier,
                optimizer: torch.optim.Optimizer, progress: Progress, labels: tuple[np.ndarray, np.ndarray],
                checkpoint: tuple[Path, str], clock: tuple[float, float | None]) -> None:
    """
    Train the remaining batches of the current epoch, checkpointing periodically.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``.
    data : PilotData
        Preloaded inputs.
    model : MorphologyCPCClassifier
        Arm being trained.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    progress : Progress
        Position, updated in place.
    labels : tuple[np.ndarray, np.ndarray]
        Exposure mask and targets from ``exposed_labels``.
    checkpoint : tuple[Path, str]
        Arm directory and fingerprint.
    clock : tuple[float, float | None]
        ``time.monotonic`` start of this invocation and the deadline.

    Raises
    ------
    SystemExit
        With ``INTERRUPTED_EXIT`` after checkpointing on SIGTERM or the deadline.
    """
    exposed, targets = labels
    directory, fingerprint = checkpoint
    started, deadline = clock
    model.train()
    batches = fixed_batches(len(data.partitions.full), exposed, progress.epoch, data.seed)
    for position in range(progress.batch, len(batches)):
        indices = batches[position]
        x = normalized_batch(data.waveforms, indices, data.mean, data.std, args.device)
        y = torch.from_numpy(targets[indices]).to(args.device)
        mask = torch.from_numpy(exposed[indices]).to(args.device)
        loss = F.binary_cross_entropy_with_logits(model(x)[mask], y[mask])
        checked_step(loss, model, optimizer, "training")
        totals = progress.totals
        totals["updates"] = totals.get("updates", 0) + 1
        totals["record_exposures"] = totals.get("record_exposures", 0) + len(indices)
        totals["label_exposures"] = totals.get("label_exposures", 0) + int(mask.sum())
        totals["loss_sum"] = totals.get("loss_sum", 0.0) + float(loss.detach())
        progress.batch = position + 1
        stop = interrupted(deadline)
        if progress.batch % SAVE_EVERY == 0 or stop:
            save_state(directory, fingerprint, model, optimizer, progress,
                       progress.elapsed_seconds + time.monotonic() - started)
        if stop:
            raise SystemExit(INTERRUPTED_EXIT)


def finish_epoch(args: argparse.Namespace, data: PilotData, model: MorphologyCPCClassifier,
                 optimizer: torch.optim.Optimizer, progress: Progress, arm: tuple[str, str],
                 checkpoint: tuple[Path, str], started: float) -> None:
    """
    Screen development data, keep the best model and save the epoch boundary.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``.
    data : PilotData
        Preloaded inputs.
    model : MorphologyCPCClassifier
        Arm being trained.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    progress : Progress
        Position, updated in place.
    arm : tuple[str, str]
        Label budget and arm kind.
    checkpoint : tuple[Path, str]
        Arm directory and fingerprint.
    started : float
        ``time.monotonic`` start of this invocation.

    Raises
    ------
    RuntimeError
        If the weights became nonfinite.
    """
    budget, kind = arm
    directory, fingerprint = checkpoint
    logits = predict_development(model, data, args.device)
    screen = development_screen(data.partitions.development, logits)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Nonfinite model weights")
    if screen["auroc"] > progress.best["auc"]:
        progress.best = {"auc": screen["auroc"], "epoch": progress.epoch + 1}
        write_torch_atomic(directory / "best_model.pt", {"fingerprint": fingerprint,
                           "epoch": progress.epoch + 1, "model": cpu_state(model)})
        write_npz_atomic(directory / "best_logits.npz", logits=logits)
    totals = progress.totals
    row = {"epoch": progress.epoch + 1, "budget": budget, "arm": kind,
           "mean_batch_loss": totals["loss_sum"] / totals["updates"],
           "optimizer_updates": totals["updates"], "record_exposures": totals["record_exposures"],
           "label_exposures": totals["label_exposures"], "development": screen,
           "best_epoch": progress.best["epoch"],
           "elapsed_seconds": progress.elapsed_seconds + time.monotonic() - started}
    progress.history.append(row)
    print(json.dumps(row), flush=True)
    progress.epoch += 1
    progress.batch, progress.totals = 0, {}
    save_state(directory, fingerprint, model, optimizer, progress,
               progress.elapsed_seconds + time.monotonic() - started)


def run_arm(args: argparse.Namespace, data: PilotData, budget: str, kind: str, directory: Path,
            max_epochs: int, deadline: float | None) -> list[dict[str, Any]]:
    """
    Train one arm resumably, or verify it when already complete.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``ssl`` and ``device``.
    data : PilotData
        Preloaded inputs.
    budget : str
        Label budget.
    kind : str
        Arm.
    directory : Path
        Arm output directory.
    max_epochs : int
        Stop after this many completed epochs; only ``EPOCHS`` completes the arm.
    deadline : float | None
        ``time.monotonic`` deadline for checkpoint-and-exit.

    Returns
    -------
    list[dict[str, Any]]
        Per-epoch history.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_json(identity(data, budget, kind))
    if (directory / "completion.json").exists():
        return check_completion(directory, fingerprint, ARTIFACTS)
    config = directory / "config.json"
    check_existing_config(config, fingerprint)
    model = make_model(args.ssl, kind, data.bank, args.device, data.seed)
    optimizer = optimizer_for(model)
    progress = load_state(directory, fingerprint, model, optimizer, dict(INITIAL_BEST))
    write_json_atomic(config, {"fingerprint": fingerprint, "identity": identity(data, budget, kind),
                               "parameter_count": parameter_count(model),
                               "exposed_labels": FULL_CLEAN_LABELS if budget == "1" else LIMITED_LABELS})
    labels = clean_exposed_labels(data.partitions, budget)
    checkpoint = (directory, fingerprint)
    started = time.monotonic()
    while progress.epoch < max_epochs:
        train_epoch(args, data, model, optimizer, progress, labels, checkpoint, (started, deadline))
        finish_epoch(args, data, model, optimizer, progress, (budget, kind), checkpoint, started)
    if progress.epoch == EPOCHS and max_epochs == EPOCHS:
        write_completion(directory, fingerprint, progress.best, ARTIFACTS)
    return progress.history


def verify_roundtrip(args: argparse.Namespace, data: PilotData, kind: str, directory: Path,
                     device: str) -> dict[str, Any]:
    """
    Reload a one-epoch profile checkpoint into a fresh model and compare every state.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``ssl``.
    data : PilotData
        Preloaded inputs.
    kind : str
        Profiled arm.
    directory : Path
        Profile arm directory holding ``resume.pt``.
    device : str
        Device of the fresh model.

    Returns
    -------
    dict[str, Any]
        Roundtrip receipt.

    Raises
    ------
    ValueError
        If the model, optimizer or random states differ after reloading.
    """
    original = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    probe = make_model(args.ssl, kind, data.bank, device, data.seed)
    optimizer = optimizer_for(probe)
    progress = load_state(directory, sha256_json(identity(data, "1", kind)), probe, optimizer,
                          dict(INITIAL_BEST))
    receipt = check_roundtrip(original, progress, probe, optimizer)
    if not rng_matches(original["rng"]):
        raise ValueError("Profile RNG roundtrip differs")
    return {**receipt, "rng_roundtrip": True}


def best_rows(output: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]] | None:
    """
    Summarize each completed arm's best development epoch.

    Parameters
    ----------
    output : Path
        Pilot output directory.

    Returns
    -------
    tuple[list[str], dict[tuple[str, str], dict[str, Any]]] | None
        Report table rows and best development screens keyed by
        ``(budget, kind)``, or ``None`` while an arm is incomplete.

    Raises
    ------
    ValueError
        If a completed arm lacks epochs.
    """
    lines, selected = [], {}
    for seed in SEEDS:
      for budget in BUDGETS:
        for kind in KINDS:
            directory = output / f"{kind}_fraction{budget}_seed{seed}"
            if not (directory / "completion.json").exists():
                return None
            history = json.loads((directory / "history.json").read_text())
            if len(history) != EPOCHS:
                raise ValueError("Completed arm lacks five epochs")
            epoch = json.loads((directory / "completion.json").read_text())["best_epoch"]
            dev = history[epoch - 1]["development"]
            selected[(str(seed), budget, kind)] = dev
            lines.append(f"| {seed} | {budget} | {kind} | {epoch} | {dev['auroc']:.4f} | "
                         f"{dev['fold_specificity']:.4f} | "
                         f"{dev['fold_sensitivity']:.4f} | {sum(r['optimizer_updates'] for r in history)} | "
                         f"{sum(r['label_exposures'] for r in history)} | "
                         f"{history[-1]['elapsed_seconds']:.1f} |")
    return lines, selected


def report_pilot(output: Path) -> None:
    """
    Write the development-only report once all six arms are complete.

    Parameters
    ----------
    output : Path
        Pilot output directory.
    """
    summary = best_rows(output)
    if summary is None:
        return
    rows, selected = summary
    lines = ["# Experiment 017 development-only morphology pilot", "",
             "Five fixed epochs per arm. No calibration or test predictions were used.", "",
             "| Seed | Label budget | Arm | Best epoch | Development AUROC | Patient-fold specificity | "
             "Patient-fold sensitivity | Updates | Labeled exposures | Wall seconds |",
             "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *rows,
             "", "## Prespecified comparison", ""]
    advance = True
    for seed in SEEDS:
      for budget in BUDGETS:
        template = selected[(str(seed), budget, "template")]
        conv = selected[(str(seed), budget, "conv")]
        plain = selected[(str(seed), budget, "none")]
        gain_conv = template["auroc"] - conv["auroc"]
        gain_plain = template["auroc"] - plain["auroc"]
        sens_delta = template["fold_sensitivity"] - conv["fold_sensitivity"]
        passed = ((gain_conv >= REQUIRED_GAIN and gain_plain >= REQUIRED_GAIN
                   if budget == "0.1" else gain_conv >= -REQUIRED_GAIN)
                  and sens_delta >= -SENSITIVITY_TOLERANCE)
        advance &= passed
        lines.append(f"Seed {seed}, budget {budget}: template minus convolution AUROC {gain_conv:+.4f}, "
                     f"template minus no-branch AUROC {gain_plain:+.4f}, "
                     f"template minus convolution patient-fold sensitivity {sens_delta:+.4f}; "
                     f"positive criteria {'met' if passed else 'not met'}.")
    decision = ("Both optimization seeds passed the conditional development screen." if advance else
                "The conditional two-seed development screen did not pass.")
    audit_path = output / "audit.json"
    if not audit_path.exists():
        raise ValueError("Artifact audit is required before final report")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "complete":
        raise ValueError("Artifact audit is incomplete")
    lines.extend(["", decision, "", f"Artifact audit: {audit_path.name}.", "",
                  "Template matches are not validated explanations. The binary endpoint is an ECG diagnostic "
                  "annotation proxy, not verified health or referral need.", ""])
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text("\n".join(lines))
    write_json_atomic(output / "completion.json", {
        "status": "development_pilot_complete", "report_sha256": sha256_file(output / "report.md"),
        "audit_sha256": sha256_file(audit_path),
        "replication_screen_passed": bool(advance),
        "arm_completions_sha256": {
            f"{kind}_fraction{budget}_seed{seed}":
                sha256_file(output / f"{kind}_fraction{budget}_seed{seed}/completion.json")
            for seed in SEEDS for budget in BUDGETS for kind in KINDS}})


def write_check(args: argparse.Namespace, data: PilotData) -> None:
    """
    Record the CPU input check that must precede any GPU stage.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir`` and ``pool_verification``.
    data : PilotData
        Verified inputs.
    """
    check_waveform_sample(data.pool, data.partitions.full[0])
    preload(args, data)
    transform = verify_clean_transform(data, args.canonical_dir)
    if len(data.template_receipt) != TEMPLATES:
        raise ValueError("Template donor count changed")
    donor_ids = {row["ecg_id"] for row in data.partitions.full}
    if any(donor["ecg_id"] not in donor_ids for donor in data.template_receipt):
        raise ValueError("Template donor left the clean training set")
    sample = normalized_batch(data.waveforms, [0, 1], data.mean, data.std, "cpu")
    for seed in SEEDS:
        starts = [make_model(args.ssl, kind, data.bank, "cpu", seed)(sample).detach() for kind in KINDS]
        if any(not torch.equal(starts[0], value) for value in starts[1:]):
            raise ValueError("Matched arms have different initial logits")
    fingerprint = sha256_json(data.provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "provenance/verification.json", {
        "stage": "check", "fingerprint": fingerprint, "provenance": data.provenance,
        "clean_transform": transform, "template_selection": data.template_receipt,
        "common_initial_logits": True,
        "source_015_receipt_sha256_for_audit": data.pool_receipt_sha256,
        "source_015_receipt": str(args.pool_verification.resolve())})
    partitions = data.partitions
    print(json.dumps({"stage": "check", "train": len(partitions.full), "limited": len(partitions.limited),
                      "development": len(partitions.development), "calibration": len(partitions.calibration),
                      "test": len(partitions.test), "precheck_seconds": data.precheck_seconds,
                      "fingerprint": fingerprint}), flush=True)


def profile(args: argparse.Namespace, data: PilotData, deadline: float, roundtrip_device: str) -> None:
    """
    Time one real full-data epoch per arm and write the two-hour planning gate.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    data : PilotData
        Preloaded inputs.
    deadline : float
        ``time.monotonic`` deadline.
    roundtrip_device : str
        Device on which each profile checkpoint is reloaded and compared.
    """
    def profile_arm(kind: str, directory: Path) -> dict[str, Any]:
        run_arm(args, data, "1", kind, directory, 1, deadline)
        roundtrip = verify_roundtrip(args, data, kind, directory, roundtrip_device)
        model = make_model(args.ssl, kind, data.bank, args.device, data.seed)
        saved = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        began = time.monotonic()
        predict_development(model, data, args.device)
        return {**roundtrip, "development_inference_seconds": time.monotonic() - began}

    durations, roundtrips = profile_arms("experiment017_profile_", KINDS, profile_arm)
    overhead = data.precheck_seconds + data.preload_seconds
    training_estimate = overhead + len(SEEDS) * len(BUDGETS) * EPOCHS * sum(durations.values())
    from ecg_experiment.morphology_clean_audit import (
        ANALYSIS_SEED,
        paired_bootstrap,
        raw_development_flags,
    )

    labels, groups = labels_and_patients(data.partitions.development)
    rng = np.random.default_rng(ANALYSIS_SEED)
    predictions = {kind: rng.random(len(labels)) for kind in KINDS}
    began = time.monotonic()
    paired_bootstrap(labels, groups, predictions)
    bootstrap_seconds = time.monotonic() - began
    verification = json.loads((args.output_dir / "provenance/verification.json").read_text())
    began = time.monotonic()
    raw_development_flags(ROOT, args.clean_dir, verification["clean_transform"],
                          data.partitions.development)
    raw_audit_seconds = time.monotonic() - began
    dev_seconds = max(item["development_inference_seconds"] for item in roundtrips.values())
    audit_estimate = (70 * dev_seconds + 4 * bootstrap_seconds + raw_audit_seconds + 60)
    estimate = 1.25 * (overhead + sum(durations.values()) + training_estimate + audit_estimate)
    receipt = {"stage": "profile", "device": args.device, "full_epoch_seconds": durations,
               "checkpoint_roundtrips": roundtrips,
               "precheck_seconds": data.precheck_seconds, "preload_seconds": data.preload_seconds,
               "projected_twelve_run_training_seconds": training_estimate,
               "projected_audit_seconds": audit_estimate,
               "bootstrap_profile_seconds": bootstrap_seconds,
               "raw_audit_profile_seconds": raw_audit_seconds,
               "conservative_profile_plus_twelve_run_seconds": estimate,
               "planning_gate_seconds": PLANNING_GATE_SECONDS,
               "gate_passed": estimate <= PLANNING_GATE_SECONDS,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
               "fingerprint": sha256_json(data.provenance)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "profile.json", receipt)
    print(json.dumps(receipt), flush=True)


def require_profile(output_dir: Path, fingerprint: str) -> None:
    """
    Require a passing real-GPU profile of these exact inputs before training.

    Parameters
    ----------
    output_dir : Path
        Pilot output directory holding ``profile.json``.
    fingerprint : str
        Current provenance digest.

    Raises
    ------
    ValueError
        If the profile is missing, from another device or inputs, or failed its gate.
    """
    path = output_dir / "profile.json"
    if not path.exists():
        raise ValueError("Real complete-pass GPU profile required before training")
    receipt = json.loads(path.read_text())
    if (receipt.get("stage") != "profile" or receipt.get("device") != "cuda"
            or receipt["fingerprint"] != fingerprint or not receipt["gate_passed"]):
        raise ValueError("GPU profile fingerprint or two-hour planning gate failed")


def parse_args(argv: list[str] | None, output: Path) -> argparse.Namespace:
    """
    Parse and validate the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    output : Path
        Default output directory.

    Returns
    -------
    argparse.Namespace
        Validated arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "train", "report"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST)
    parser.add_argument("--clean-dir", type=Path, default=CLEAN)
    parser.add_argument("--canonical-dir", type=Path, default=CANONICAL)
    parser.add_argument("--ssl", type=Path, default=SSL)
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION)
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--pool-verification", type=Path, default=POOL_VERIFICATION)
    parser.add_argument("--max-cache-bytes", type=int, default=DEFAULT_MAX_CACHE_BYTES)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    parser.add_argument("--max-wall-seconds", type=int, default=PLANNING_GATE_SECONDS)
    args = parser.parse_args(argv)
    if (args.threads < 1 or args.max_cache_bytes < MIN_CACHE_BYTES or args.reserve_bytes < 0
            or args.max_wall_seconds < 1):
        parser.error("Invalid thread, memory, or time limits")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.stage in ("profile", "report") and args.device != "cuda":
        parser.error("Profile and artifact report must use the real V100 GPU data path")
    return args


def main(argv: list[str] | None = None, *, output: Path = OUTPUT, extra_code: Sequence[str] = (),
         roundtrip_device: str | None = None) -> None:
    """
    Run the check, profile or training stage.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    output : Path
        Default output directory.
    extra_code : Sequence[str]
        Additional repository files whose digests enter the provenance.
    roundtrip_device : str | None
        Device for profile checkpoint comparisons; defaults to ``--device``.
    """
    args = parse_args(argv, output)
    torch.set_num_threads(args.threads)
    install_stop_handler()
    deadline = time.monotonic() + args.max_wall_seconds
    with gpu_lock(args.device):
        data = load_inputs(args, extra_code)
        if args.stage == "check":
            write_check(args, data)
            return
        if args.stage == "report":
            require_receipt(args.output_dir / "provenance/verification.json", sha256_json(data.provenance),
                            "Verified clean replication CPU check is missing or changed")
            require_clean_transform(args.output_dir, args.canonical_dir)
            require_profile(args.output_dir, sha256_json(data.provenance))
            preload(args, data)
            from ecg_experiment.morphology_clean_audit import run_audit

            run_audit(args, data)
            report_pilot(args.output_dir)
            return
        preload(args, data)
        fingerprint = sha256_json(data.provenance)
        if args.device == "cuda":
            require_receipt(args.output_dir / "provenance/verification.json", fingerprint,
                            "Verified Experiment 017 CPU check is missing or changed")
            require_clean_transform(args.output_dir, args.canonical_dir)
        print(json.dumps({"stage": args.stage, "precheck_seconds": data.precheck_seconds,
                          "preload_seconds": data.preload_seconds,
                          "cache_bytes": int(data.waveforms.signals.nbytes),
                          "template_selection": data.template_receipt}), flush=True)
        if args.stage == "profile":
            profile(args, data, deadline, roundtrip_device or args.device)
            return
        if args.device == "cuda":
            require_profile(args.output_dir, fingerprint)
        for seed in SEEDS:
            data.seed = seed
            for budget in BUDGETS:
                for kind in KINDS:
                    run_arm(args, data, budget, kind,
                            args.output_dir / f"{kind}_fraction{budget}_seed{seed}", EPOCHS, deadline)
        print(json.dumps({"stage": "train", "status": "all_arms_complete"}), flush=True)


if __name__ == "__main__":
    main()
