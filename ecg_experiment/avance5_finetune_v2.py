"""One prospectively fixed BF16 full-backbone pass after a measured FP32 cost stop."""

from __future__ import annotations

import gc
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.special import expit
from torch import nn

from . import ROOT, avance5
from . import avance5_finetune as parent
from .avance5_cache import delivery_run
from .cpu_screening041 import sensitivity_difference
from .external_encoders import xecg_features
from .files import sha256_file, write_json_atomic
from .gpu import gpu_lock
from .paths import to_stored
from .pr74_screening_metrics import budget_threshold, point_metrics

OUT = ROOT / "outputs/avance5_finetune_v2"
REPORT = ROOT / "reports/avance5_finetune_v2"
PROTOCOL = ROOT / "docs/avance5-finetune-v2-protocol-2026-10-29.md"
RATE = 3e-6
MICROBATCH = 8
BATCH = 32


def configure() -> None:
    """Use the existing verified CUDA environment without changing any original output artifact."""
    parent.configure()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)


def binding() -> dict[str, Any]:
    """Bind successor and unchanged parent scientific sources before new scoring."""
    inherited = parent.source_binding()
    paths = [PROTOCOL, Path(__file__), ROOT / "scripts/experiments/run_avance5_finetune_v2.py"]
    sources = {**inherited["source_sha256"], **{to_stored(path): sha256_file(path) for path in paths}}
    return {"source_sha256": sources, "training_input_sha256": inherited["training_input_sha256"],
            "protocol": to_stored(PROTOCOL), "scope": "reused_development", "closed_test_scored": False}


def update(model: nn.Module, opt: torch.optim.Optimizer, inputs: np.ndarray,
           y: np.ndarray, indices: np.ndarray) -> float:
    """Accumulate a full effective batch with BF16 forwards and FP32 master parameters and BCE."""
    model.train()
    opt.zero_grad(set_to_none=True)
    losses = []
    for start in range(0, len(indices), MICROBATCH):
        part = indices[start:start + MICROBATCH]
        tensor = torch.from_numpy(np.array(inputs[part], dtype=np.float32)).to("cuda")
        target = torch.from_numpy(y[part].astype(np.float32)).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tensor)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), target, reduction="sum")
        loss = loss / len(indices)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite BF16 loss")
        loss.backward()
        losses.append(float(loss.detach()))
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    return sum(losses)


@torch.inference_mode()
def mixed_scores(model: nn.Module, inputs: np.ndarray) -> np.ndarray:
    """Quantify mixed-precision prediction drift before optimizer updates."""
    model.eval()
    logits = []
    for start in range(0, len(inputs), MICROBATCH):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(torch.from_numpy(inputs[start:start + MICROBATCH]).to("cuda"))
        logits.append(output.float().cpu().numpy())
    return expit(np.concatenate(logits).astype(float))


def profile() -> dict[str, Any]:
    """Verify BF16 arithmetic and actual optimizer performance before admitting a complete training pass."""
    configure()
    identity = binding()
    rows, _, vectors = avance5.training_data()
    y = rows.target.to_numpy(dtype=int)
    head = parent.fit_head(vectors, y, np.arange(len(rows)))
    inputs = parent.input_rows(rows.iloc[:64])
    with gpu_lock("cuda", blocking=False):
        free, _ = torch.cuda.mem_get_info()
        model = parent.initialized_model(head)
        model.eval()
        replay = xecg_features(model.backbone, inputs[:16])
        replay_error = float(np.max(np.abs(replay - vectors[:16])))
        if not np.allclose(replay, vectors[:16], atol=1e-4, rtol=1e-5):
            raise ValueError("FP32 frozen feature replay failed")
        exact = parent.inference(model, inputs, np.arange(len(inputs)))
        expected = head.predict_proba(vectors[:64])[:, 1]
        head_error = float(np.max(np.abs(exact - expected)))
        mixed_error = float(np.max(np.abs(exact - mixed_scores(model, inputs))))
        if head_error > 2e-4 or mixed_error > 0.02:
            raise ValueError("Head conversion or BF16 drift exceeds the frozen tolerance")
        torch.cuda.reset_peak_memory_stats()
        before = model.backbone.patch_embedding.conv.weight.detach().clone()
        opt = parent.optimizer(model, RATE)
        for number in range(2):
            update(model, opt, inputs, y[:64], np.arange(number * BATCH, (number + 1) * BATCH))
        torch.cuda.synchronize()
        times, losses = [], []
        for number in range(4):
            started = time.perf_counter()
            indices = np.arange((number % 2) * BATCH, ((number % 2) + 1) * BATCH)
            losses.append(update(model, opt, inputs, y[:64], indices))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        changed = not torch.equal(before, model.backbone.patch_embedding.conv.weight.detach())
        if not changed:
            raise ValueError("BF16 profile did not update backbone weights")
        started = time.perf_counter()
        parent.inference(model, inputs, np.arange(len(inputs)))
        torch.cuda.synchronize()
        inference_per_record = (time.perf_counter() - started) / len(inputs)
        peak = torch.cuda.max_memory_allocated()
        digest = parent.saved_weights(model, OUT / "profile.safetensors")
        saved = load_file(str(OUT / "profile.safetensors"))
        roundtrip = all(torch.equal(value.detach().cpu(), saved[name])
                        for name, value in model.state_dict().items())
        if not roundtrip:
            raise ValueError("BF16-profile FP32 tensor checkpoint did not round-trip")
        del saved, model, opt
        gc.collect()
        torch.cuda.empty_cache()
    forecast = 1.5 * (max(times) * math.ceil(len(rows) / BATCH) +
                      inference_per_record * (4275 + 17402)) + 600
    value = {**identity, "precision": "BF16 autocast; FP32 master weights and final inference",
             "gpu": torch.cuda.get_device_name(), "microbatch": MICROBATCH, "effective_batch": BATCH,
             "initial_free_bytes": free, "peak_gpu_bytes": peak, "update_seconds": times,
             "inference_seconds_per_record": inference_per_record, "forecast_seconds": forecast,
             "replay_max_absolute_error": replay_error, "head_score_max_error": head_error,
             "bf16_score_max_error": mixed_error, "backbone_updated": changed,
             "checkpoint_sha256": digest, "profile_losses": losses,
             "passed": bool(forecast <= 7200 and peak < free - 512 * 1024**2)}
    write_json_atomic(REPORT / "profile.json", value)
    if not value["passed"]:
        raise RuntimeError("Single-pass BF16 resource gate did not pass")
    return value


def prepare() -> dict[str, Any]:
    """Prepare a checksum-verified 100 Hz cache of training ECGs only after successful GPU profiling."""
    profile_result = avance5.json_read(REPORT / "profile.json")
    if not profile_result["passed"]:
        raise ValueError("A passing BF16 resource profile is required")
    rows, _, _ = avance5.training_data()
    inputs = np.lib.format.open_memmap(OUT / "training_inputs.npy", mode="w+", dtype=np.float32,
                                       shape=(len(rows), 1000, 12))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(rows), 128):
            stops = range(start, min(start + 128, len(rows)), 16)
            parts = [rows.iloc[i:min(i + 16, start + 128)] for i in stops]
            inputs[start:start + sum(len(part) for part in parts)] = np.concatenate(
                list(pool.map(parent.input_rows, parts)))
            if start % 1024 == 0:
                inputs.flush()
                print(f"Preparación {min(start + 128, len(rows)):,}/{len(rows):,}", flush=True)
    inputs.flush()
    result = {"records": len(rows), "seconds": time.perf_counter() - started,
              "input_sha256": sha256_file(OUT / "training_inputs.npy"),
              "manifest_sha256": sha256_file(avance5.TRAIN / "training_rows.csv")}
    write_json_atomic(REPORT / "preparation.json", result)
    return result


def evaluate(model: nn.Module, run: dict[str, Any]) -> dict[str, Any]:
    """Evaluate exactly the fixed component and equal blend, retaining all negative results."""
    records, contrasts = [], {}
    for site in ("ptb", "sph"):
        inputs = parent.site_inputs(site)
        scores = parent.inference(model, inputs, np.arange(len(inputs)))
        frame = run["sites"][site][0]
        held, y = frame.evaluation.to_numpy(), frame.target.to_numpy()
        predictions = {"xecg": scores, "blend": (scores + run["predictions"][site]["hgb"]) / 2}
        for name, predicted in predictions.items():
            threshold = budget_threshold(y[~held], predicted[~held])["threshold"]
            point = point_metrics(y[held], predicted[held], threshold)
            control = run["table"].query("sitio == @site and id == @name").iloc[0]
            difference = sensitivity_difference(frame.patient_id.to_numpy()[held], y[held],
                predicted[held] > threshold, run["predictions"][site][name][held] > control.umbral,
                draws=2000, seed=20261029)
            increase = float(control.specificity - point["specificity"])
            contrasts[site + "/" + name] = {"sensitivity_difference": difference,
                "normal_referral_increase": increase, "control_metrics": control.to_dict(),
                "practical_improvement": bool(difference["difference"] >= 0.02 and
                                              difference["ci_low"] > 0 and increase <= 0.005)}
            records.append({"site": site, "model": name, "threshold": threshold, **point})
        np.savez_compressed(OUT / f"{site}_predictions.npz", **predictions, target=y, evaluation=held)
        print(f"Evaluación {site} completa", flush=True)
    return {"records": records, "contrasts": contrasts}


def train() -> dict[str, Any]:
    """Take one fresh complete full-backbone pass, then freeze weights before hospital evaluation."""
    configure()
    profile_result = avance5.json_read(REPORT / "profile.json")
    prep = avance5.json_read(REPORT / "preparation.json")
    if not profile_result["passed"] or sha256_file(OUT / "training_inputs.npy") != prep["input_sha256"]:
        raise ValueError("Profile or training input verification failed")
    for name, expected in profile_result["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Successor scientific source changed after profiling")
    run = delivery_run()
    rows, _, vectors = avance5.training_data()
    y = rows.target.to_numpy(dtype=int)
    inputs = np.load(OUT / "training_inputs.npy", allow_pickle=False, mmap_mode="r")
    head = parent.fit_head(vectors, y, np.arange(len(rows)))
    order = np.random.default_rng(43).permutation(len(rows))
    started = time.perf_counter()
    with gpu_lock("cuda", blocking=False):
        model = parent.initialized_model(head)
        opt = parent.optimizer(model, RATE)
        losses = []
        for start in range(0, len(rows), BATCH):
            losses.append(update(model, opt, inputs, y, order[start:start + BATCH]))
            if start % (50 * BATCH) == 0:
                progress = {"trained_records": min(start + BATCH, len(rows)), "total_records": len(rows),
                            "seconds": time.perf_counter() - started, "mean_loss": float(np.mean(losses))}
                write_json_atomic(OUT / "progress.json", progress)
                print(progress, flush=True)
        digest = parent.saved_weights(model, OUT / "final.safetensors")
        training_seconds = time.perf_counter() - started
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        result = evaluate(model, run)
    result.update({**binding(), "complete": True, "epochs": 1, "trained_records": len(rows),
                   "training_seconds": training_seconds, "total_seconds": time.perf_counter() - started,
                   "mean_loss": float(np.mean(losses)), "checkpoint_sha256": digest,
                   "profile": profile_result, "preparation": prep})
    write_json_atomic(REPORT / "results.json", result)
    pd.DataFrame(result["records"]).to_csv(REPORT / "comparison.csv", index=False)
    return result
