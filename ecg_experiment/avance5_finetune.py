"""Bounded, fully trainable xECG successor to the completed Avance 5 notebook."""

from __future__ import annotations

import gc
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from . import ROOT, avance5
from .avance5_cache import delivery_run
from .cpu_screening041 import sensitivity_difference
from .cuda_runtime20261028 import configure_cuda_runtime
from .external_encoders import xecg_features
from .files import sha256_file, write_json_atomic
from .gpu import gpu_lock
from .paths import to_stored
from .pr74_screening_features import read_training_signal
from .pr74_screening_metrics import budget_threshold, point_metrics
from .xecg import XECGBinaryClassifier, load_xecg, preprocess_xecg

OUT = ROOT / "outputs/avance5_finetune"
REPORT = ROOT / "reports/avance5_finetune"
PROTOCOL = ROOT / "docs/avance5-finetune-protocol-2026-10-29.md"
EFFECTIVE_BATCH = 32
INFERENCE_BATCH = 16


def configure() -> None:
    """Use the installed CUDA runtime and deterministic FP32 arithmetic on an available GPU."""
    configure_cuda_runtime()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA GPU is required")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)


def source_binding() -> dict[str, Any]:
    """Capture scientific code and existing all-age training identity before any candidate score."""
    summary = avance5.json_read(avance5.REPORT / "summary.json")
    paths = [PROTOCOL, Path(__file__), ROOT / "scripts/experiments/run_avance5_finetune.py",
             ROOT / "ecg_experiment/pr74_screening_features.py", ROOT / "ecg_experiment/xecg.py"]
    return {"protocol": to_stored(PROTOCOL),
            "source_sha256": {to_stored(path): sha256_file(path) for path in paths},
            "training_input_sha256": summary["input_sha256"],
            "scope": "reused_development", "closed_test_scored": False}


def internal_split(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Divide only the training cohort into disjoint fit, normal-calibration and epoch-selection groups."""
    groups = np.where(rows.patient_id.ne(""), rows.source + ":patient:" + rows.patient_id, rows.record_id)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold = np.empty(len(rows), dtype=int)
    for number, (_, held) in enumerate(splitter.split(np.zeros(len(rows)), rows.target.astype(int), groups)):
        fold[held] = number
    result = {"fit": np.flatnonzero(fold >= 2), "calibration": np.flatnonzero(fold == 1),
              "validation": np.flatnonzero(fold == 0)}
    sets = [set(groups[index]) for index in result.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Internal training patients overlap")
    return result


def logistic_c() -> float:
    """Read the already selected training-only regularization, never a hospital-development result."""
    search = pd.read_csv(avance5.REPORT / "hiperparametros.csv")
    selected = search.loc[search.modelo.eq("xECG + logística")].sort_values("AUROC_VC", ascending=False)
    return float(json.loads(selected.iloc[0].parametros)["logisticregression__C"])


def fit_head(vectors: np.ndarray, y: np.ndarray, indices: np.ndarray) -> Any:
    """Fit a frozen readout exclusively on the supplied training indices."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=logistic_c(), max_iter=3000, tol=1e-6))
    model.fit(vectors[indices], y[indices])
    return model


def unscaled_head(model: Any) -> tuple[np.ndarray, float]:
    """Convert a fitted standardization/logistic pipeline exactly into raw-embedding logit weights."""
    scaler, logistic = model.steps[0][1], model.steps[1][1]
    weights = logistic.coef_[0].astype(np.float64) / scaler.scale_
    bias = float(logistic.intercept_[0] - np.dot(weights, scaler.mean_))
    return weights.astype(np.float32), bias


def initialized_model(head: Any) -> XECGBinaryClassifier:
    """Make every released backbone parameter trainable and initialize an equivalent linear head."""
    torch.manual_seed(42)
    backbone = load_xecg(device="cuda", backend="vanilla", drop_path_prob=0.0)
    model = XECGBinaryClassifier(backbone).to("cuda")
    weights, bias = unscaled_head(head)
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(weights[None]).to("cuda"))
        model.head.bias.fill_(bias)
    model.requires_grad_(True)
    return model


def optimizer(model: nn.Module, rate: float) -> torch.optim.Optimizer:
    """Update the full backbone and head at their fixed separate learning rates."""
    return torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": rate},
        {"params": model.head.parameters(), "lr": rate * 10},
    ], weight_decay=0.01, foreach=False)


def training_update(model: nn.Module, opt: torch.optim.Optimizer, inputs: np.ndarray,
                    targets: np.ndarray, indices: np.ndarray, microbatch: int) -> float:
    """Take one exact-size accumulated BCE step, including short final effective batches."""
    model.train()
    opt.zero_grad(set_to_none=True)
    total_loss = 0.0
    for start in range(0, len(indices), microbatch):
        part = indices[start:start + microbatch]
        tensor = torch.from_numpy(np.array(inputs[part], dtype=np.float32)).to("cuda")
        y = torch.from_numpy(targets[part].astype(np.float32)).to("cuda")
        logits = model(tensor)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum") / len(indices)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite fine-tuning loss")
        loss.backward()
        total_loss += float(loss.detach())
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite full-backbone gradients")
    opt.step()
    return total_loss


@torch.inference_mode()
def inference(model: nn.Module, inputs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Score only the explicitly supplied admitted records."""
    model.eval()
    result = []
    for start in range(0, len(indices), INFERENCE_BATCH):
        part = indices[start:start + INFERENCE_BATCH]
        tensor = torch.from_numpy(np.array(inputs[part], dtype=np.float32)).to("cuda")
        result.append(model(tensor).float().cpu().numpy())
    scores = expit(np.concatenate(result).astype(np.float64))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite fine-tuning predictions")
    return scores


def saved_weights(model: nn.Module, path: Path) -> str:
    """Save tensor-only safetensors and bind their exact digest."""
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
              str(path))
    return sha256_file(path)


def input_rows(rows: pd.DataFrame) -> np.ndarray:
    """Read admitted checksum-verified raw signals and preserve the established official input transform."""
    return np.stack([preprocess_xecg(read_training_signal(ROOT, row)) for _, row in rows.iterrows()])


def profile() -> dict[str, Any]:
    """Measure actual gradients, Adam states, numerical replay and a bounded complete-study forecast."""
    configure()
    binding = source_binding()
    rows, _, vectors = avance5.training_data()
    splits = internal_split(rows)
    y = rows.target.to_numpy(dtype=int)
    head = fit_head(vectors, y, splits["fit"])
    profile_indices = splits["fit"][:64]
    inputs = input_rows(rows.iloc[profile_indices])
    with gpu_lock("cuda", blocking=False):
        free, _ = torch.cuda.mem_get_info()
        model = initialized_model(head)
        model.eval()
        replay = xecg_features(model.backbone, inputs[:16])
        error = float(np.max(np.abs(replay - vectors[profile_indices[:16]])))
        if not np.allclose(replay, vectors[profile_indices[:16]], atol=1e-4, rtol=1e-5):
            raise ValueError("Training waveform/backbone replay differs from its frozen embeddings")
        expected = head.predict_proba(vectors[profile_indices])[:, 1]
        observed = inference(model, inputs, np.arange(len(inputs)))
        head_error = float(np.max(np.abs(expected - observed)))
        if head_error > 2e-4:
            raise ValueError("Converted head failed frozen prediction equivalence")
        torch.cuda.reset_peak_memory_stats()
        opt = optimizer(model, 1e-5)
        before = model.backbone.patch_embedding.conv.weight.detach().clone()
        started = time.perf_counter()
        losses = [training_update(model, opt, inputs, y[profile_indices],
                                  np.arange(start, start + EFFECTIVE_BATCH), 8)
                  for start in (0, EFFECTIVE_BATCH)]
        torch.cuda.synchronize()
        per_update = (time.perf_counter() - started) / 2
        changed = not torch.equal(before, model.backbone.patch_embedding.conv.weight.detach())
        if not changed:
            raise ValueError("The backbone's input projection did not update")
        started = time.perf_counter()
        inference(model, inputs, np.arange(len(inputs)))
        torch.cuda.synchronize()
        inference_seconds = (time.perf_counter() - started) / len(inputs)
        peak = torch.cuda.max_memory_allocated()
        digest = saved_weights(model, OUT / "profile.safetensors")
        state = load_file(str(OUT / "profile.safetensors"), device="cpu")
        if any(not torch.equal(value.detach().cpu(), state[name])
               for name, value in model.state_dict().items()):
            raise ValueError("Tensor-only checkpoint round trip failed")
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        del state, model, opt
        gc.collect()
        torch.cuda.empty_cache()
    forecasts = []
    chosen = None
    for rates, epochs in (((1e-5, 3e-6), 4), ((1e-5,), 4), ((1e-5,), 2)):
        updates = len(rates) * epochs * math.ceil(len(splits["fit"]) / EFFECTIVE_BATCH)
        updates += epochs * math.ceil(len(rows) / EFFECTIVE_BATCH)
        inferred = len(rates) * epochs * (len(splits["calibration"]) + len(splits["validation"]))
        inferred += 2 * (4275 + 17402)
        forecast = 1.5 * (updates * per_update + inferred * inference_seconds) + 600
        forecasts.append({"rates": list(rates), "epochs": epochs, "forecast_seconds": forecast})
        if chosen is None and forecast <= 7200:
            chosen = forecasts[-1]
    value = {**binding, "gpu": torch.cuda.get_device_name(), "microbatch": 8,
             "effective_batch": EFFECTIVE_BATCH, "initial_free_bytes": free, "peak_gpu_bytes": peak,
             "seconds_per_update": per_update, "inference_seconds_per_record": inference_seconds,
             "replay_max_absolute_error": error, "head_score_max_absolute_error": head_error,
             "backbone_projection_updated": changed, "trainable_parameters": trainable,
             "profile_losses": losses, "checkpoint_sha256": digest, "forecasts": forecasts,
             "chosen": chosen, "passed": chosen is not None and peak < 8 * 1024**3,
             "split_records": {key: len(value) for key, value in splits.items()}}
    write_json_atomic(REPORT / "profile.json", value)
    if not value["passed"]:
        raise RuntimeError("Fine-tuning did not pass the predeclared resource gate")
    return value


def prepare() -> dict[str, Any]:
    """Cache only admitted training inputs after a passing real-GPU profile."""
    profile_result = avance5.json_read(REPORT / "profile.json")
    if not profile_result["passed"]:
        raise ValueError("A passing profile is required")
    rows, _, _ = avance5.training_data()
    inputs = np.lib.format.open_memmap(OUT / "training_inputs.npy", mode="w+", dtype=np.float32,
                                       shape=(len(rows), 1000, 12))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(rows), 128):
            part = rows.iloc[start:start + 128]
            signals = list(pool.map(lambda item: preprocess_xecg(read_training_signal(ROOT, item[1])),
                                    part.iterrows()))
            inputs[start:start + len(part)] = np.stack(signals)
            if start % 1024 == 0:
                inputs.flush()
                print(f"Preparación {start + len(part):,}/{len(rows):,}", flush=True)
    inputs.flush()
    value = {"records": len(rows), "seconds": time.perf_counter() - started,
             "inputs_sha256": sha256_file(OUT / "training_inputs.npy"),
             "manifest_sha256": sha256_file(avance5.TRAIN / "training_rows.csv")}
    write_json_atomic(REPORT / "preparation.json", value)
    return value


def epoch(model: nn.Module, opt: torch.optim.Optimizer, inputs: np.ndarray, y: np.ndarray,
          indices: np.ndarray, microbatch: int, number: int) -> dict[str, float]:
    """Train one complete, seeded pass through the declared fitting records."""
    order = np.random.default_rng(42 + number).permutation(indices)
    started = time.perf_counter()
    losses = []
    for start in range(0, len(order), EFFECTIVE_BATCH):
        losses.append(training_update(model, opt, inputs, y,
                                      order[start:start + EFFECTIVE_BATCH], microbatch))
        if start % (100 * EFFECTIVE_BATCH) == 0:
            done = min(start + EFFECTIVE_BATCH, len(order))
            print(f"  época {number}, {done:,}/{len(order):,}", flush=True)
    torch.cuda.synchronize()
    return {"loss": float(np.mean(losses)), "seconds": time.perf_counter() - started}


def validation_metrics(model: nn.Module, inputs: np.ndarray, y: np.ndarray,
                       splits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Choose epochs on a training-only ranking endpoint and report a separately calibrated matrix."""
    calibration, validation = splits["calibration"], splits["validation"]
    cal_scores = inference(model, inputs, calibration)
    scores = inference(model, inputs, validation)
    threshold = budget_threshold(y[calibration], cal_scores)["threshold"]
    return {"partial_auroc": float(roc_auc_score(y[validation], scores, max_fpr=0.05)),
            "threshold": threshold, **point_metrics(y[validation], scores, threshold)}


def select_schedule(inputs: np.ndarray, rows: pd.DataFrame, vectors: np.ndarray,
                    profile_result: dict[str, Any]) -> dict[str, Any]:
    """Compare the prospectively cost-qualified schedules using training-only outcomes."""
    y = rows.target.to_numpy(dtype=int)
    splits = internal_split(rows)
    head = fit_head(vectors, y, splits["fit"])
    history = []
    for rate in profile_result["chosen"]["rates"]:
        model = initialized_model(head)
        opt = optimizer(model, rate)
        baseline = {"rate": rate, "epoch": 0, **validation_metrics(model, inputs, y, splits)}
        history.append(baseline)
        write_json_atomic(REPORT / "history.json", history)
        for number in range(1, profile_result["chosen"]["epochs"] + 1):
            timing = epoch(model, opt, inputs, y, splits["fit"], profile_result["microbatch"], number)
            metrics = validation_metrics(model, inputs, y, splits)
            history.append({"rate": rate, "epoch": number, **timing, **metrics})
            write_json_atomic(REPORT / "history.json", history)
            print(f"LR {rate:g}, época {number}: pAUROC={metrics['partial_auroc']:.4f}, "
                  f"sens={metrics['sensitivity']:.4f}, FP={metrics['fp']}", flush=True)
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    candidates = [row for row in history if row["epoch"] > 0]
    best = max(candidates, key=lambda row: (row["partial_auroc"], -row["rate"], -row["epoch"]))
    selected = {"rate": best["rate"], "epochs": best["epoch"], "training_validation": best,
                "frozen_training_validation": history[0], "source_sha256": source_binding()["source_sha256"]}
    write_json_atomic(REPORT / "selection.json", selected)
    return selected


def site_inputs(site: str) -> np.ndarray:
    """Read only fixed already explored hospital-development waveforms after model selection is frozen."""
    from .pr74_fresh_baseline import align_roles, read_signal

    rows = pd.read_csv(ROOT / "outputs/cohort25_recovery_20261028/raw_roles" / f"{site}_rows.csv",
                       dtype=str, keep_default_na=False)
    evidence_name = "ptb_administrative.csv" if site == "ptb" else "sph_all.csv"
    evidence = pd.read_csv(ROOT / "outputs/cohort25_recovery_20261028/raw_roles" / evidence_name,
                           dtype=str, keep_default_na=False)
    rows = align_roles(rows, evidence, site)
    expected = pd.read_csv(avance5.PRIOR / f"qualify/{site}_rows.csv", dtype={"ecg_id": str})
    if site == "ptb":
        expected = expected.loc[expected.strat_fold.ge(8)]
        rows = rows.loc[pd.to_numeric(rows.strat_fold).isin([8, 9])]
    if rows.ecg_id.astype(str).tolist() != expected.ecg_id.astype(str).tolist():
        raise ValueError("Raw evaluation role order differs from frozen scoring identities")
    arrays = []
    for _, row in rows.iterrows():
        arrays.append(preprocess_xecg(read_signal(ROOT, row)))
    return np.stack(arrays)


def evaluate(model: nn.Module, run: dict[str, Any]) -> dict[str, Any]:
    """Score the fixed final candidate and mixture at unchanged hospital normal-referral budgets."""
    records = []
    contrasts = {}
    for site in ("ptb", "sph"):
        raw_inputs = site_inputs(site)
        scores = inference(model, raw_inputs, np.arange(len(raw_inputs)))
        frame = run["sites"][site][0]
        y, held = frame.target.to_numpy(), frame.evaluation.to_numpy()
        predictions = {"xecg_finetuned": scores,
                       "blend_finetuned": (scores + run["predictions"][site]["hgb"]) / 2}
        for name, prediction in predictions.items():
            threshold = budget_threshold(y[~held], prediction[~held])["threshold"]
            point = point_metrics(y[held], prediction[held], threshold)
            control_id = "xecg" if name == "xecg_finetuned" else "blend"
            reference = run["table"].query("sitio == @site and id == @control_id").iloc[0]
            records.append({"site": site, "model": name, "threshold": threshold, **point})
            difference = sensitivity_difference(
                frame.patient_id.to_numpy()[held], y[held], prediction[held] > threshold,
                run["predictions"][site][control_id][held] > reference.umbral,
                draws=2000, seed=20261029)
            guard = (1 - point["specificity"]) - (1 - reference.specificity)
            contrasts[site + "/" + name] = {
                "sensitivity_difference": difference, "normal_referral_increase": float(guard),
                "practical_improvement": bool(difference["difference"] >= 0.02 and
                                              difference["ci_low"] > 0 and guard <= 0.005),
                "control": control_id, "control_metrics": reference.to_dict()}
        np.savez_compressed(OUT / f"{site}_predictions.npz", **predictions, target=y, evaluation=held)
    return {"records": records, "contrasts": contrasts}


def train() -> dict[str, Any]:
    """Run schedule selection, fresh all-cohort refitting and the single development comparison."""
    configure()
    profile_result = avance5.json_read(REPORT / "profile.json")
    preparation = avance5.json_read(REPORT / "preparation.json")
    digest = sha256_file(OUT / "training_inputs.npy")
    if not profile_result["passed"] or digest != preparation["inputs_sha256"]:
        raise ValueError("Profile or prepared inputs failed verification")
    for name, expected in profile_result["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Fine-tuning source changed after profiling")
    run = delivery_run()
    rows, _, vectors = avance5.training_data()
    inputs = np.load(OUT / "training_inputs.npy", allow_pickle=False, mmap_mode="r")
    y = rows.target.to_numpy(dtype=int)
    started = time.perf_counter()
    with gpu_lock("cuda", blocking=False):
        selection = select_schedule(inputs, rows, vectors, profile_result)
        all_indices = np.arange(len(rows))
        model = initialized_model(fit_head(vectors, y, all_indices))
        opt = optimizer(model, selection["rate"])
        refit = []
        for number in range(1, selection["epochs"] + 1):
            refit.append(epoch(model, opt, inputs, y, all_indices, profile_result["microbatch"], number))
            write_json_atomic(REPORT / "refit.json", refit)
        digest = saved_weights(model, OUT / "final.safetensors")
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        result = evaluate(model, run)
    result.update({**source_binding(), "selection": selection, "profile": profile_result,
                   "preparation": preparation, "refit": refit, "checkpoint_sha256": digest,
                   "seconds": time.perf_counter() - started, "complete": True})
    write_json_atomic(REPORT / "results.json", result)
    pd.DataFrame(result["records"]).to_csv(REPORT / "comparison.csv", index=False)
    return result
