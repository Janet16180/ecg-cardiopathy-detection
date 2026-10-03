"""Time pipeline v4 and its explanation per ECG on the CPU only, on PTB-XL training ECGs.

Steps per ECG: read and preprocess one record; xECG features; ECG-JEPA tokens; the three attention heads;
the readouts and finding heads; the 042 beat cutting and wave scores (``U_B``); and 048's explanation rule.
The 042 wave references are fitted once from the 5,872 fit-set ECGs in a child process, and that one-time
cost is reported separately. CPU outputs are checked against the saved GPU-path values where they exist.
Run with ``CUDA_VISIBLE_DEVICES=`` so that no GPU is visible.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from ecg_experiment.cpu_inference import (
    attention_scores,
    linear_logit,
    load_attention_heads,
    load_jepa_cpu,
    peak_rss_mib,
)
from ecg_experiment.explanation_rule import rule_marks
from ecg_experiment.external_encoders import (
    XECG_DROP_PATH,
    jepa_cache,
    jepa_input,
    read_ptb_float64,
    xecg_cache,
    xecg_input,
)
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.lead_wave_maps import (
    MINIMUM_BEATS,
    WAVES,
    beat_pieces,
    beat_unit_map,
    fit_wave_references,
    jepa_tokens,
    jepa_unit_map,
    wave_scores,
)
from ecg_experiment.pipeline_v4 import ensemble_logit
from ecg_experiment.provenance import git_head
from ecg_experiment.pvc_switch import pvc_z, switch_explain
from ecg_experiment.xecg import DEFAULT_CHECKPOINT_DIR, load_xecg
from scripts.experiments.run_lead_wave_maps042 import read_with, select_rows
from scripts.experiments.run_rhythm_findings032 import sph_inputs

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/inference_timing_cpu_v1"
HEADS = ROOT / "outputs/experiment044_pipeline_v3_v1/pipeline_v3_heads.npz"
ATTENTION = ROOT / "outputs/experiment046_pipeline_v4_v1/attention_jepa_weights.npz"
PREDICTIONS046 = ROOT / "outputs/experiment046_pipeline_v4_v1/predictions.npz"
RESULT042 = ROOT / "outputs/experiment042_lead_wave_maps_v1/result.json"
RESULT048 = ROOT / "outputs/experiment048_pvc_switch_v1/result.json"
TOKEN_CACHE = Path("/tmp/claude-218201143/-home-janetrivera-ecg-cardiopathy-detection/"
                   "64bd5d6e-e2f5-403b-8837-1c9210d4ec66/scratchpad/exp043_cache")
SAMPLE_SEED = 2026
FIT_THREADS = 4
FEATURE_TOLERANCE = 1e-4
LOGIT_TOLERANCE = 1e-5
FS = 500
STEPS = ("read_preprocess", "xecg", "jepa_tokens", "attention_heads", "readouts_findings", "beat_wave_map",
         "explanation")


def sample_rows(count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return a seeded sample of PTB-XL training ECGs in every feature cache, and the 042 fit set.

    Parameters
    ----------
    count : int
        Number of ECGs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Sampled training rows (``ecg_id``, ``filename_hr``, ``token_row``) and 042's fit rows.

    Raises
    ------
    ValueError
        If a sampled row is not a training row or is missing from the token cache.
    """
    rows = select_rows(False)
    pool = rows["pool"].sort_values("ecg_id")
    keys = pd.read_csv(TOKEN_CACHE / "rows.csv")["key"].tolist()
    token_row = {key: index for index, key in enumerate(keys)}
    pool = pool[[f"ptbxl:{i}" in token_row for i in pool["ecg_id"]]]
    chosen = np.sort(np.random.default_rng(SAMPLE_SEED).choice(len(pool), count, replace=False))
    sample = pool.iloc[chosen][["ecg_id", "filename_hr"]].copy()
    sample["token_row"] = [token_row[f"ptbxl:{i}"] for i in sample["ecg_id"]]
    if set(sample["ecg_id"]) & set(rows["scored"]["ecg_id"]):
        raise ValueError("A sampled ECG is a development row")
    return sample.reset_index(drop=True), rows["fit"]


def fit_references(stems: list[str], queue: Any) -> None:
    """Fit 042's 48 wave references from the fit-set records, in a child process, and report the cost."""
    with threadpool_limits(limits=FIT_THREADS):
        started = time.perf_counter()
        beats = read_with(lambda stem: beat_pieces(read_ptb_float64(stem), FS), stems)
        read_seconds = time.perf_counter() - started
        kept = [found for found in beats if len(found[0]) >= MINIMUM_BEATS]
        started = time.perf_counter()
        references = fit_wave_references({name: np.concatenate([found[1][name] for found in kept])
                                          for name in WAVES})
        fit_seconds = time.perf_counter() - started
    size = sum(ref.precision_.nbytes + ref.location_.nbytes for refs in references.values() for ref in refs)
    queue.put({"references": references, "read_cut_seconds": read_seconds, "fit_seconds": fit_seconds,
               "records": len(stems), "usable": len(kept), "fit_beats": int(sum(len(f[0]) for f in kept)),
               "reference_mib": size / 2**20, "peak_rss_mib": peak_rss_mib()})


def one_time_references(fit: pd.DataFrame) -> dict[str, Any]:
    """Run ``fit_references`` in a fresh process, so its memory does not count in the per-ECG peak."""
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    started = time.perf_counter()
    process = context.Process(target=fit_references, args=(fit["filename_hr"].tolist(), queue))
    process.start()
    found = queue.get()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"Reference fit failed with exit code {process.exitcode}")
    found["wall_seconds"] = time.perf_counter() - started
    return found


def load_models() -> tuple[dict[str, Any], dict[str, float]]:
    """Load every model and parameter file on the CPU, timing each load."""
    seconds, models = {}, {}
    started = time.perf_counter()
    models["xecg"] = load_xecg(DEFAULT_CHECKPOINT_DIR, device="cpu", backend="vanilla",
                               drop_path_prob=XECG_DROP_PATH).eval()
    models["xecg"].requires_grad_(False)
    seconds["xecg"] = time.perf_counter() - started
    started = time.perf_counter()
    models["jepa"] = load_jepa_cpu()
    seconds["jepa"] = time.perf_counter() - started
    started = time.perf_counter()
    models["attention"] = load_attention_heads(ATTENTION)
    seconds["attention_heads"] = time.perf_counter() - started
    started = time.perf_counter()
    with np.load(HEADS) as saved:
        models["heads"] = {name: saved[name] for name in saved.files}
    seconds["readout_and_finding_heads"] = time.perf_counter() - started
    return models, seconds


@torch.inference_mode()
def score_one(models: dict[str, Any], references: dict[str, Any], stem: str, constants: dict[str, float]
              ) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Run pipeline v4 and the explanation on one record, timing each step.

    Parameters
    ----------
    models : dict[str, Any]
        Output of ``load_models``.
    references : dict[str, Any]
        042's fitted wave references.
    stem : str
        PTB-XL ``filename_hr``.
    constants : dict[str, float]
        Explanation thresholds: ``switch`` (048's z_PVC switch) and each map's own red threshold.

    Returns
    -------
    tuple[dict[str, float], dict[str, Any]]
        Seconds per step, and the outputs (features, tokens, logits, explanation).
    """
    seconds, out = {}, {}
    started = time.perf_counter()
    signal = read_ptb_float64(stem)
    xecg_in = xecg_input(signal)
    jepa_in = jepa_input(signal.astype(np.float32))
    seconds["read_preprocess"] = time.perf_counter() - started

    started = time.perf_counter()
    pooled, _ = models["xecg"](torch.from_numpy(xecg_in[None]))
    out["xecg"] = pooled.float().numpy()[0]
    seconds["xecg"] = time.perf_counter() - started

    started = time.perf_counter()
    out["tokens"] = jepa_tokens(models["jepa"], jepa_in[None], batch=1)
    seconds["jepa_tokens"] = time.perf_counter() - started

    started = time.perf_counter()
    logits, contributions = attention_scores(models["attention"], out["tokens"])
    out["attention_logit"], out["contributions"] = float(logits[0]), contributions[0]
    seconds["attention_heads"] = time.perf_counter() - started

    started = time.perf_counter()
    heads = models["heads"]
    jepa = out["tokens"][0].mean(axis=0)
    out["r3_logit"] = float(linear_logit(heads, "v3", np.concatenate([out["xecg"], jepa])[None])[0])
    out["E"] = float(ensemble_logit(np.array([out["r3_logit"]]), np.array([out["attention_logit"]]))[0])
    out["z_pvc"] = float(pvc_z(heads, out["xecg"][None])[0])
    out["z_wpw"] = float(pvc_z(heads, out["xecg"][None], "wpw")[0])
    out["F"] = max(out["z_pvc"], out["z_wpw"])
    seconds["readouts_findings"] = time.perf_counter() - started

    started = time.perf_counter()
    r_times, pieces = beat_pieces(signal, FS)
    out["U_B"] = beat_unit_map(wave_scores(references, pieces), r_times)
    seconds["beat_wave_map"] = time.perf_counter() - started

    started = time.perf_counter()
    attention_map = jepa_unit_map(out["contributions"])
    found = switch_explain(out["U_B"], attention_map, out["z_pvc"] > constants["switch"], constants["U_B"],
                           constants["attention_jepa"])
    threshold = constants["U_B"] if found.layer == 1 else constants["attention_jepa"]
    out["explanation"] = {"layer": found.layer, "marks": rule_marks(found, threshold)}
    seconds["explanation"] = time.perf_counter() - started
    return seconds, out


def time_threads(models: dict[str, Any], references: dict[str, Any], sample: pd.DataFrame,
                 constants: dict[str, float], threads: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Time every sampled ECG with a given number of CPU threads, after one untimed warm-up ECG."""
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        score_one(models, references, sample["filename_hr"].iloc[0], constants)
        records, outputs = [], []
        for stem in sample["filename_hr"]:
            seconds, out = score_one(models, references, stem, constants)
            records.append(seconds)
            outputs.append(out)
    table = pd.DataFrame(records)
    table["total"] = table[list(STEPS)].sum(axis=1)
    summary = {name: {"median_ms": float(table[name].median() * 1000),
                      "p90_ms": float(table[name].quantile(0.9) * 1000)} for name in (*STEPS, "total")}
    return {"threads": threads, "steps": summary, "peak_rss_mib_so_far": peak_rss_mib()}, outputs


def check_outputs(sample: pd.DataFrame, outputs: list[dict[str, Any]], models: dict[str, Any]
                  ) -> dict[str, Any]:
    """
    Compare CPU outputs with the saved GPU-path values.

    Parameters
    ----------
    sample : pd.DataFrame
        Output of ``sample_rows``.
    outputs : list[dict[str, Any]]
        ``score_one`` outputs in sample order.
    models : dict[str, Any]
        Output of ``load_models``.

    Returns
    -------
    dict[str, Any]
        Largest differences and whether each is within its tolerance.
    """
    ids = sample["ecg_id"].to_numpy(np.int64)
    xecg, xecg_ids, _ = xecg_cache()
    jepa, jepa_ids, _ = jepa_cache()
    x_rows = [int(np.flatnonzero(xecg_ids == i)[0]) for i in ids]
    j_rows = [int(np.flatnonzero(jepa_ids == i)[0]) for i in ids]
    tokens = np.load(TOKEN_CACHE / "tokens.npy", mmap_mode="r")
    cached_tokens = np.stack([np.asarray(tokens[r], dtype=np.float32) for r in sample["token_row"]])
    mine_tokens = np.concatenate([out["tokens"] for out in outputs])
    cached_logits, _ = attention_scores(models["attention"], cached_tokens)
    found: dict[str, Any] = {
        "xecg_vs_cache": float(np.abs(np.stack([o["xecg"] for o in outputs]) - xecg[x_rows]).max()),
        "jepa_pooled_vs_cache": float(np.abs(mine_tokens.mean(axis=1) - jepa[j_rows]).max()),
        "tokens_vs_043_cache": float(np.abs(mine_tokens - cached_tokens).max()),
        "token_cache_float16_resolution": float(np.abs(cached_tokens).max() * 2.0**-11),
        "tokens_float16_equal_share": float(np.mean(mine_tokens.astype(np.float16).astype(np.float32)
                                                    == cached_tokens)),
        "attention_logit_cpu_tokens_vs_cached_tokens": float(np.abs(
            np.array([o["attention_logit"] for o in outputs]) - cached_logits).max()),
    }
    found.update(check_sph(models, tokens))
    found["within_tolerance"] = {
        "xecg": found["xecg_vs_cache"] <= FEATURE_TOLERANCE,
        "jepa_pooled": found["jepa_pooled_vs_cache"] <= FEATURE_TOLERANCE,
        "attention_sph": found["sph_attention_vs_046"] <= LOGIT_TOLERANCE,
        "r3_sph": found["sph_r3_vs_046"] <= LOGIT_TOLERANCE,
        "E_sph": found["sph_E_vs_046"] <= LOGIT_TOLERANCE,
    }
    return found


def check_sph(models: dict[str, Any], tokens: np.ndarray, count: int = 50) -> dict[str, Any]:
    """Score 50 cached SPH token rows and saved SPH features on the CPU against 046's saved logits."""
    keys = pd.read_csv(TOKEN_CACHE / "rows.csv")["key"].tolist()
    token_row = {key: index for index, key in enumerate(keys)}
    sph, sph_x, _ = sph_inputs()
    with np.load(PREDICTIONS046) as saved:
        saved_ids = saved["sph_ecg_ids"]
        if not np.array_equal(saved_ids, sph["ecg_id"].to_numpy(dtype=str)):
            raise ValueError("046's SPH rows differ from 032's")
        usable = np.flatnonzero(saved["sph_in_043_cache"])
        picked = np.random.default_rng(SAMPLE_SEED).choice(len(usable), count, replace=False)
        positions = usable[np.sort(picked)]
        attention, r3, ensemble = (saved[f"sph_{name}"][positions]
                                   for name in ("attention_jepa", "v3_logit", "v4_logit"))
    rows = [token_row[f"sph:{saved_ids[p]}"] for p in positions]
    logits, _ = attention_scores(models["attention"], np.stack([np.asarray(tokens[r]) for r in rows]))
    x = np.concatenate([sph_x["xecg"][positions], sph_x["jepa"][positions]], axis=1)
    mine_r3 = linear_logit(models["heads"], "v3", x)
    return {"sph_attention_vs_046": float(np.abs(logits - attention).max()),
            "sph_r3_vs_046": float(np.abs(mine_r3 - r3).max()),
            "sph_E_vs_046": float(np.abs(ensemble_logit(mine_r3, logits) - ensemble).max())}


def explanation_constants() -> dict[str, float]:
    """Return 048's z_PVC switch threshold and the two maps' own red thresholds, as 048 saved them."""
    result = json.loads(RESULT048.read_text())
    return {"switch": float(result["cases"]["logistic_concat_q0.975"]["switch_threshold"]),
            "U_B": float(result["map_thresholds"]["U_B"]),
            "attention_jepa": float(result["map_thresholds"]["attention_jepa"])}


def machine() -> dict[str, Any]:
    """Return the CPU model, core count and library versions."""
    names = [line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
             if line.startswith("model name")]
    return {"cpu": names[0] if names else platform.processor(), "logical_cpus": multiprocessing.cpu_count(),
            "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "cuda_available": torch.cuda.is_available()}


def main() -> None:
    """Parse arguments, run the timing and write the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    if torch.cuda.is_available():
        raise RuntimeError("A GPU is visible; run with CUDA_VISIBLE_DEVICES=")
    started = time.perf_counter()
    sample, fit = sample_rows(arguments.count)
    one_time = one_time_references(fit)
    references = one_time.pop("references")
    expected_beats = json.loads(RESULT042.read_text())["arm_b"]["fit_beats"]
    one_time["fit_beats_equal_042"] = one_time["fit_beats"] == expected_beats
    if not one_time["fit_beats_equal_042"]:
        raise ValueError(f"Fit beats {one_time['fit_beats']} differ from 042's {expected_beats}")
    rss_before_models = peak_rss_mib()
    models, load_seconds = load_models()
    rss_after_models = peak_rss_mib()
    constants = explanation_constants()
    timings, checks = [], None
    for threads in arguments.threads:
        found, outputs = time_threads(models, references, sample, constants, threads)
        timings.append(found)
        if checks is None:
            checks = check_outputs(sample, outputs, models)
    layers = [out["explanation"]["layer"] for out in outputs]
    arguments.output.mkdir(parents=True)
    write_json_atomic(arguments.output / "result.json", {
        "machine": machine(), "git_head": git_head(ROOT), "ecg_ids": sample["ecg_id"].tolist(),
        "sample_seed": SAMPLE_SEED, "one_time_references": one_time, "load_seconds": load_seconds,
        "peak_rss_mib": {"before_models": rss_before_models, "after_models": rss_after_models,
                         "after_timing": peak_rss_mib(), "reference_fit_child": one_time["peak_rss_mib"]},
        "timings": timings, "checks": checks, "explanation_constants": constants,
        "explained_by_U_B": int(sum(layer == 1 for layer in layers)),
        "inputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in (HEADS, ATTENTION)},
        "total_seconds": time.perf_counter() - started, "development_rows_read": False,
    })
    print(json.dumps({"timings": timings, "checks": checks, "load_seconds": load_seconds,
                      "one_time": one_time}, indent=1))


if __name__ == "__main__":
    main()
