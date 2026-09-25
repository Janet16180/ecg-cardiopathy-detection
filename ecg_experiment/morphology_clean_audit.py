"""Development-only artifact and concentration checks for clean Experiment 017."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment.cpc import split_halves
from ecg_experiment.cpc_morphology import TEMPLATES, local_response
from ecg_experiment.files import read_csv, sha256_file, write_json_atomic
from ecg_experiment.pilot import BATCH, clipped_sigmoid, labels_and_patients, normalized_batch
from ecg_experiment.waveforms import LEADS, read_record

BOOTSTRAPS = 2000
ANALYSIS_SEED = 17017
MEAN_SAMPLE = 1024
PLATEAU_MIN_SAMPLES = 5


def auc_or_none(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return AUROC when both diagnostic proxy classes occur."""
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def paired_bootstrap(labels: np.ndarray, groups: np.ndarray,
                     predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    """Draw whole patients once and reuse indices across every arm contrast."""
    patients, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(patients))]
    rng = np.random.default_rng(ANALYSIS_SEED)
    contrasts = {"template_minus_conv": ("template", "conv"),
                 "template_minus_none": ("template", "none")}
    draws: dict[str, list[float]] = {name: [] for name in contrasts}
    single_class = 0
    for _ in range(BOOTSTRAPS):
        chosen = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[index] for index in chosen])
        if len(np.unique(labels[rows])) != 2:
            single_class += 1
            continue
        for name, (left, right) in contrasts.items():
            draws[name].append(float(roc_auc_score(labels[rows], predictions[left][rows])
                                     - roc_auc_score(labels[rows], predictions[right][rows])))
    return {"draws_requested": BOOTSTRAPS, "single_class_skipped": single_class,
            "effective_count": BOOTSTRAPS - single_class,
            "contrasts": {name: {"observed": float(roc_auc_score(labels, predictions[left])
                                                - roc_auc_score(labels, predictions[right])),
                                 "interval_95": np.quantile(draws[name], [0.025, 0.975]).tolist()}
                          for name, (left, right) in contrasts.items()}}


def leave_one_patient_out(labels: np.ndarray, groups: np.ndarray,
                          template: np.ndarray, conv: np.ndarray) -> dict[str, Any]:
    """Measure the selected limited-label AUROC delta after each patient deletion."""
    observed = float(roc_auc_score(labels, template) - roc_auc_score(labels, conv))
    deltas = []
    for patient in np.unique(groups):
        keep = groups != patient
        if len(np.unique(labels[keep])) == 2:
            deltas.append(float(roc_auc_score(labels[keep], template[keep])
                                - roc_auc_score(labels[keep], conv[keep])))
    return {"observed_delta": observed, "patients_checked": len(deltas),
            "minimum_delta": min(deltas), "maximum_delta": max(deltas),
            "any_reversal": any(np.sign(delta) != np.sign(observed) for delta in deltas)}


def _extreme_plateau(signal: np.ndarray, threshold: float) -> bool:
    """Find five identical consecutive samples at an amplitude above the frozen threshold."""
    for lead in signal:
        if not np.any(np.abs(lead) >= threshold):
            continue
        same = np.diff(lead) == 0
        runs = np.convolve(same.astype(np.int8), np.ones(PLATEAU_MIN_SAMPLES - 1, dtype=np.int8),
                           mode="valid")
        starts = np.flatnonzero(runs == PLATEAU_MIN_SAMPLES - 1)
        if any(abs(float(lead[start])) >= threshold for start in starts):
            return True
    return False


def raw_development_flags(root: Path, clean_dir: Path, train_thresholds: dict[str, float],
                          development: list[dict[str, str]]) -> dict[str, Any]:
    """Read only development raw ECGs and apply thresholds fitted on training ECGs."""
    references = {row["record_id"]: row for row in read_csv(clean_dir / "heldout_references.csv")
                  if row["split"] == "development"}
    raw_dir = root / "data/raw/ptb-xl/1.0.3"
    max_cut = train_thresholds["train_maxabs_p99_mV"]
    rms_cut = train_thresholds["train_rms_p99_mV"]
    records = []
    for row in development:
        record_id = f"ptbxl:{row['ecg_id']}"
        path = (root / references[record_id]["raw_path"]).resolve()
        signal = read_record(raw_dir, str(path.relative_to(raw_dir.resolve())))
        maximum = float(np.max(np.abs(signal)))
        rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
        flags = {"maxabs_gt_train_p99": maximum > max_cut, "rms_gt_train_p99": rms > rms_cut,
                 "maxabs_gt_10mV": maximum > 10.0,
                 "constant_leads": [LEADS[i] for i in np.flatnonzero(np.ptp(signal, axis=1) == 0)],
                 "extreme_plateau": _extreme_plateau(signal, max(10.0, max_cut))}
        records.append({"record_id": record_id, "patient_id": row["patient_id"],
                        "target": int(row["target"]), "maxabs_mV": maximum, "rms_mV": rms,
                        "flags": flags, "flagged": any((flags["maxabs_gt_train_p99"],
                                                       flags["rms_gt_train_p99"],
                                                       flags["maxabs_gt_10mV"]))})
    if len(records) != len(development):
        raise ValueError("Development raw audit count changed")
    return {"definitions": {"maxabs_p99_train_mV": max_cut, "rms_p99_train_mV": rms_cut,
                            "absolute_review_mV": 10.0, "plateau_min_samples": PLATEAU_MIN_SAMPLES,
                            "plateau_threshold_mV": max(10.0, max_cut)},
            "records": records}


def subgroup_contrasts(labels: np.ndarray, scores: dict[str, np.ndarray],
                       flags: np.ndarray) -> dict[str, Any]:
    """Keep the complete development comparison primary and describe unflagged records."""
    result = {}
    for name, mask in (("all", np.ones(len(labels), dtype=bool)), ("unflagged", ~flags),
                       ("flagged", flags)):
        subset = labels[mask]
        values = {arm: auc_or_none(subset, prediction[mask]) for arm, prediction in scores.items()}
        missing = any(values[arm] is None for arm in ("template", "conv"))
        result[name] = {"records": int(mask.sum()), "positive": int(subset.sum()),
                        "negative": int(len(subset) - subset.sum()), "auroc": values,
                        "template_minus_conv": None if missing else values["template"] - values["conv"]}
    return result


def _quantiles(values: np.ndarray) -> dict[str, float]:
    """Compactly describe a response or matched-window amplitude distribution."""
    return {str(q): float(np.quantile(values, q)) for q in (0.0, 0.01, 0.5, 0.99, 1.0)}


def response_diagnostics(model: torch.nn.Module, data: Any, device: str,
                         training_sample: np.ndarray) -> dict[str, Any]:
    """Describe distance responses, strongest matches, and physical-mV window amplitudes."""
    train_top = [{"response": -float("inf")} for _ in range(TEMPLATES)]
    dev_top = [{"response": -float("inf")} for _ in range(TEMPLATES)]
    response_values = []
    record_best = []
    window_amplitudes = []
    train_count = len(data.partitions.full)
    with torch.inference_mode():
        for split, indices in (("training", training_sample),
                               ("development", np.arange(len(data.partitions.development)))):
            for offset in range(0, len(indices), BATCH):
                selected = indices[offset:offset + BATCH]
                cache_indices = selected if split == "training" else selected + train_count
                x = normalized_batch(data.waveforms, cache_indices, data.mean, data.std, device)
                response = local_response(split_halves(x), model.encoder.bank, "template")
                values = response.reshape(len(selected), 2, TEMPLATES, -1).cpu().numpy()
                if split == "development":
                    response_values.append(values.reshape(-1))
                top = train_top if split == "training" else dev_top
                rows = data.partitions.full if split == "training" else data.partitions.development
                for channel in range(TEMPLATES):
                    local = values[:, :, channel, :]
                    flat = int(np.argmax(local))
                    row_index, half, token = np.unravel_index(flat, local.shape)
                    score = float(local[row_index, half, token])
                    if score > top[channel]["response"]:
                        row = rows[int(selected[row_index])]
                        top[channel] = {"response": score, "ecg_id": row["ecg_id"],
                                        "patient_id": row["patient_id"],
                                        "half": int(half), "token": int(token)}
                if split == "development":
                    for local_index, source_index in enumerate(selected):
                        flat = int(np.argmax(values[local_index]))
                        half, channel, token = np.unravel_index(flat, values[local_index].shape)
                        end = 16 * int(token)
                        start = max(0, end - 49)
                        left = int(half) * 1250 + start
                        right = int(half) * 1250 + end + 1
                        physical = data.waveforms.signals[train_count + int(source_index), :, left:right]
                        window_amplitudes.append(float(np.max(np.abs(physical))))
                        record_best.append(float(values[local_index, half, channel, token]))
    return {"response_quantiles": _quantiles(np.concatenate(response_values)),
            "strongest_response_per_record_quantiles": _quantiles(np.asarray(record_best)),
            "strongest_match_window_maxabs_mV_quantiles": _quantiles(np.asarray(window_amplitudes)),
            "strongest_training_match_by_channel": train_top,
            "strongest_development_match_by_channel": dev_top}


def _predict_with_hook(model: torch.nn.Module, data: Any, device: str,
                       hook: Callable[[Any, tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]] | None,
                       branch_off: bool = False) -> np.ndarray:
    """Evaluate a frozen model with one response change or its whole branch removed."""
    from ecg_experiment.morphology_clean import predict_development

    handle = None
    if hook is not None:
        handle = model.encoder.branch_hidden.register_forward_pre_hook(hook)
    elif branch_off:
        handle = model.encoder.branch_final.register_forward_hook(
            lambda _module, _args, output: torch.zeros_like(output))
    try:
        with torch.inference_mode():
            return predict_development(model, data, device)
    finally:
        if handle is not None:
            handle.remove()


def replace_response_channel(response: torch.Tensor, channel: int, mean: float) -> torch.Tensor:
    """Substitute only one response channel while preserving all other channels."""
    if response.ndim != 3 or channel < 0 or channel >= response.shape[2]:
        raise ValueError("Invalid response or channel")
    result = response.clone()
    result[:, :, channel] = mean
    return result


def template_concentration(args: Any, data: Any, seed: int,
                           baseline: np.ndarray, conv: np.ndarray) -> dict[str, Any]:
    """Ablate every selected template channel at its unlabeled training mean."""
    from ecg_experiment.morphology_clean import make_model

    data.seed = seed
    directory = args.output_dir / f"template_fraction0.1_seed{seed}"
    selected = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=False)
    model = make_model(args.ssl, "template", data.bank, args.device, seed)
    model.load_state_dict(selected["model"])
    model.eval()
    rng = np.random.default_rng(42)
    sample = rng.choice(len(data.partitions.full),
                        size=min(MEAN_SAMPLE, len(data.partitions.full)), replace=False)
    sums = np.zeros(TEMPLATES, dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for offset in range(0, len(sample), BATCH):
            x = normalized_batch(data.waveforms, sample[offset:offset + BATCH],
                                 data.mean, data.std, args.device)
            response = local_response(split_halves(x), model.encoder.bank, "template")
            sums += response.sum(dim=(0, 2)).cpu().numpy()
            count += response.shape[0] * response.shape[2]
    means = sums / count
    diagnostics = response_diagnostics(model, data, args.device, sample)
    unmodified = _predict_with_hook(model, data, args.device, None)
    if not np.allclose(unmodified, baseline, rtol=0, atol=1e-5):
        raise ValueError("Selected template logits changed before ablation")
    labels, _ = labels_and_patients(data.partitions.development)
    baseline_auc = float(roc_auc_score(labels, clipped_sigmoid(baseline)))
    conv_auc = float(roc_auc_score(labels, clipped_sigmoid(conv)))
    gain = baseline_auc - conv_auc
    channels = []
    for channel in range(TEMPLATES):
        mean = float(means[channel])

        def replace(_module: Any, inputs: tuple[torch.Tensor, ...],
                    channel: int = channel, mean: float = mean) -> tuple[torch.Tensor, ...]:
            return (replace_response_channel(inputs[0], channel, mean),)

        modified = _predict_with_hook(model, data, args.device, replace)
        auc = float(roc_auc_score(labels, clipped_sigmoid(modified)))
        change = gain - (auc - conv_auc)
        channels.append({"channel": channel, "donor": data.template_receipt[channel],
                         "strongest_training_match":
                             diagnostics["strongest_training_match_by_channel"][channel],
                         "strongest_development_match":
                             diagnostics["strongest_development_match_by_channel"][channel],
                         "unlabeled_training_mean_response": mean, "auroc": auc,
                         "mean_abs_logit_change": float(np.mean(np.abs(modified - baseline))),
                         "gain_erased": change, "dominance_flag": gain > 0 and change >= 0.5 * gain})
    branch = _predict_with_hook(model, data, args.device, None, branch_off=True)
    branch_auc = float(roc_auc_score(labels, clipped_sigmoid(branch)))
    return {"training_sample_seed": 42, "training_sample_records": len(sample),
            "training_response_tokens": count, "selected_epoch": selected["epoch"],
            "baseline_auroc": baseline_auc, "conv_auroc": conv_auc, "baseline_gain": gain,
            "response_diagnostics": {key: value for key, value in diagnostics.items()
                                     if not key.endswith("_by_channel")},
            "channels": channels,
            "whole_branch": {"auroc": branch_auc,
                             "mean_abs_logit_change": float(np.mean(np.abs(branch - baseline))),
                             "gain_erased": gain - (branch_auc - conv_auc)}}


def run_audit(args: Any, data: Any) -> dict[str, Any]:
    """Verify selected artifacts and publish development-only analysis."""
    from ecg_experiment.morphology_clean import BUDGETS, KINDS, SEEDS, best_rows

    if best_rows(args.output_dir) is None:
        raise ValueError("All twelve completed arms are required before artifact audit")
    receipt = json.loads((args.output_dir / "provenance/verification.json").read_text())
    labels, groups = labels_and_patients(data.partitions.development)
    raw = raw_development_flags(args.output_dir.parents[1], args.clean_dir,
                                receipt["clean_transform"], data.partitions.development)
    flags = np.asarray([row["flagged"] for row in raw["records"]], dtype=bool)
    scores = {}
    for seed in SEEDS:
        for budget in BUDGETS:
            scores[(seed, budget)] = {}
            for kind in KINDS:
                path = args.output_dir / f"{kind}_fraction{budget}_seed{seed}/best_logits.npz"
                with np.load(path, allow_pickle=False) as archive:
                    logits = archive["logits"]
                if len(logits) != len(labels) or not np.isfinite(logits).all():
                    raise ValueError("Selected development logits are invalid")
                scores[(seed, budget)][kind] = clipped_sigmoid(logits)
    comparisons = {}
    for seed in SEEDS:
        for budget in BUDGETS:
            prediction = scores[(seed, budget)]
            comparisons[f"seed{seed}_fraction{budget}"] = {
                "subgroups": subgroup_contrasts(labels, prediction, flags),
                "paired_patient_bootstrap": paired_bootstrap(labels, groups, prediction),
                "average_precision": {kind: float(average_precision_score(labels, values))
                                      for kind, values in prediction.items()},
                "leave_one_patient_out": leave_one_patient_out(
                    labels, groups, prediction["template"], prediction["conv"])}
    concentration = {}
    for seed in SEEDS:
        with np.load(args.output_dir / f"template_fraction0.1_seed{seed}/best_logits.npz",
                     allow_pickle=False) as archive:
            baseline = archive["logits"]
        with np.load(args.output_dir / f"conv_fraction0.1_seed{seed}/best_logits.npz",
                     allow_pickle=False) as archive:
            conv = archive["logits"]
        concentration[str(seed)] = template_concentration(args, data, seed, baseline, conv)
    deltas = {budget: [comparisons[f"seed{seed}_fraction{budget}"]["subgroups"]["all"]
                       ["template_minus_conv"] for seed in SEEDS] for budget in BUDGETS}
    summary = {budget: {"mean_delta": float(np.mean(values)), "range": [min(values), max(values)]}
               for budget, values in deltas.items()}
    result = {"status": "complete", "scope": "development_only", "raw_flags": raw,
              "comparisons": comparisons, "concentration": concentration,
              "two_seed_delta_summary": summary,
              "source_hashes": {"verification.json": sha256_file(
                  args.output_dir / "provenance/verification.json")}}
    write_json_atomic(args.output_dir / "audit.json", result)
    return result
