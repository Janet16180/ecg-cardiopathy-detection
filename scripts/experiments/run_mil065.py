"""
Experiment 065: train finding-specific multiple-instance heads, run sanity checks and write LocBench maps.

Phases: ``train`` (one network per lock hold), ``maps`` (held-out predictions, sanity checks, maps for
ECGAD development and the evaluation partition). ``--smoke`` writes to a separate directory with tiny
settings. See ``docs/experiment-065-finding-mil.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment import mil065 as mil
from ecg_experiment.ecgad import load_split
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.locbench_cache import CACHE_DIR, open_cache
from ecg_experiment.locbench_data import PARTITION_PATH, REGION_TABLE, load_partition
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head, utc_now
from ecg_experiment.waveforms import LEADS

PRIMARY_SEEDS = (65065, 65066, 65067)
MODELS: dict[str, dict[str, Any]] = {
    "lse_65065": {"pooling": "lse", "cross_lead": False, "seed": 65065},
    "lse_65066": {"pooling": "lse", "cross_lead": False, "seed": 65066},
    "lse_65067": {"pooling": "lse", "cross_lead": False, "seed": 65067},
    "attention_65065": {"pooling": "attention", "cross_lead": False, "seed": 65065},
    "crosslead_65065": {"pooling": "lse", "cross_lead": True, "seed": 65065},
}
GROUPS = {
    "mil_lse": ["lse_65065", "lse_65066", "lse_65067"],
    "mil_attention": ["attention_65065"],
    "mil_crosslead": ["crosslead_65065"],
}
VARIANTS = {
    "mil_lse": ("mil_lse", "sum"),
    "mil_lse_max": ("mil_lse", "max"),
    "mil_attention": ("mil_attention", "sum"),
    "mil_crosslead": ("mil_crosslead", "sum"),
}
REINIT_SEED_OFFSET = 100
LEAD_CHECK_HEADS = ("IMI", "ASMI", "AMI", "ALMI", "LMI", "CRBBB", "LAFB")
LEAD_CHECK_HEADS += ("INJAS", "INJAL", "ISCAS", "ISCAL", "ISCIN", "ISCLA")
CHUNK = 128


def output_dir(smoke: bool) -> Path:
    """Output directory of the run."""
    return ROOT / ("outputs/experiment065_mil_smoke" if smoke else "outputs/experiment065_mil_v1")


def preprocess_rows(signals: np.ndarray, positions: np.ndarray, workers: int = 8) -> np.ndarray:
    """Preprocess cache rows in parallel chunks."""
    out = np.empty((len(positions), 12, mil.SAMPLES), np.float32)
    chunks = np.array_split(np.arange(len(positions)), max(1, len(positions) // 256))

    def run(chunk: np.ndarray) -> None:
        out[chunk] = mil.preprocess(signals[positions[chunk]])

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, chunks))
    return out


def fit_data(smoke: bool) -> dict[str, Any]:
    """
    Fit-partition records, labels, heads, held-out mask and preprocessed signals.

    Returns
    -------
    dict
        ``ecg_ids``, ``patient_ids``, ``positions`` (cache rows), ``signals``, ``y``, ``mask``, ``heads``,
        ``heldout`` and ``counts``.
    """
    raw, index = open_cache()
    partition = load_partition()
    fit = partition[partition.partition == "fit"].reset_index(drop=True)
    forbidden = set(load_split().ecg_id) | set(partition.ecg_id[partition.partition == "evaluation"])
    if forbidden & set(fit.ecg_id):
        raise ValueError("A fit record is an ECGAD or evaluation record")
    row_of = dict(zip(index.ecg_id, range(len(index)), strict=True))
    positions = np.array([row_of[e] for e in fit.ecg_id])
    if not np.array_equal(index.patient_id.to_numpy()[positions], fit.patient_id.to_numpy()):
        raise ValueError("Cache and partition patients differ")
    heldout = mil.heldout_mask(fit.patient_id.to_numpy())
    meta, statements = load_metadata(), load_statements()
    ecg_ids = fit.ecg_id.to_numpy()
    codes = mil.candidate_codes(meta, ecg_ids)
    y_all, mask_all = mil.statement_labels(meta, statements, ecg_ids, codes)
    heads = mil.select_heads(y_all[~heldout], mask_all[~heldout], codes, mil.MIN_TRAIN_POSITIVES)
    columns = [codes.index(h) for h in heads]
    y, mask = y_all[:, columns], mask_all[:, columns]
    if smoke:
        keep = np.concatenate([np.flatnonzero(~heldout)[:1500], np.flatnonzero(heldout)[:300]])
        ecg_ids, positions, heldout, y, mask = (
            ecg_ids[keep],
            positions[keep],
            heldout[keep],
            y[keep],
            mask[keep],
        )
        fit = fit.iloc[keep].reset_index(drop=True)
    cache = Path(os.environ.get("ECG065_CACHE", "/tmp/ecg065-cache")) / ("smoke" if smoke else "full")
    path = cache / "fit_preprocessed.npy"
    if path.exists() and np.array_equal(np.load(cache / "positions.npy"), positions):
        signals = np.load(path, mmap_mode="r")
        signals = np.ascontiguousarray(signals)
    else:
        start = time.time()
        signals = preprocess_rows(raw, positions)
        cache.mkdir(parents=True, exist_ok=True)
        np.save(path, signals)
        np.save(cache / "positions.npy", positions)
        print(f"preprocessed {len(positions)} fit records in {time.time() - start:.0f}s", flush=True)
    counts = {
        h: {
            "train_positive": int((y[~heldout, k] * mask[~heldout, k]).sum()),
            "train_defined": int(mask[~heldout, k].sum()),
            "heldout_positive": int((y[heldout, k] * mask[heldout, k]).sum()),
            "heldout_defined": int(mask[heldout, k].sum()),
        }
        for k, h in enumerate(heads)
    }
    return {
        "fit": fit,
        "ecg_ids": ecg_ids,
        "positions": positions,
        "signals": signals,
        "raw": raw,
        "y": y,
        "mask": mask,
        "heads": heads,
        "heldout": heldout,
        "counts": counts,
    }


def build(name: str, heads: int) -> mil.LeadMIL:
    """Fresh network of a named model, seeded."""
    spec = MODELS[name]
    torch.manual_seed(spec["seed"])
    return mil.LeadMIL(heads, pooling=spec["pooling"], cross_lead=spec["cross_lead"])


def device_setup() -> str:
    """Pick CUDA when visible and cap this process at 6 GB."""
    if not torch.cuda.is_available():
        return "cpu"
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6e9 / total))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return "cuda"


def train_phase(smoke: bool) -> None:
    """Train every model not yet saved, holding the GPU lock for one network at a time."""
    out = output_dir(smoke)
    out.mkdir(parents=True, exist_ok=True)
    data = fit_data(smoke)
    device = device_setup()
    train_rows = np.flatnonzero(~data["heldout"])
    valid_rows = np.flatnonzero(data["heldout"])
    recipe = mil.Recipe(max_epochs=2) if smoke else mil.Recipe()
    names = ["lse_65065"] if smoke else list(MODELS)
    for name in names:
        if (out / f"weights_{name}.pt").exists():
            print(f"{name} exists, skipping", flush=True)
            continue
        model = build(name, len(data["heads"])).to(device)
        start = time.time()
        with gpu_lock(device):
            fit = mil.train(
                model,
                data["signals"],
                data["y"],
                data["mask"],
                train_rows,
                valid_rows,
                recipe,
                device,
                MODELS[name]["seed"],
                max_steps=20 if smoke else None,
            )
            model.load_state_dict(fit.state)
            bag, _, _ = mil.predict(model, data["signals"], valid_rows, device)
        torch.save(fit.state, out / f"weights_{name}.pt")
        write_npz_atomic(out / f"heldout_bags_{name}.npz", ecg_id=data["ecg_ids"][valid_rows], bag=bag)
        write_json_atomic(
            out / f"history_{name}.json",
            {
                "best_epoch": fit.best_epoch,
                "best_heldout_macro_auroc": fit.best_score,
                "seconds": time.time() - start,
                "history": fit.history,
                "heads": data["heads"],
                "counts": data["counts"],
            },
        )
        print(
            f"{name}: best epoch {fit.best_epoch}, {fit.best_score:.4f}, {time.time() - start:.0f}s",
            flush=True,
        )


def load_models(out: Path, heads: int, device: str) -> dict[str, mil.LeadMIL]:
    """Trained networks from their saved weights."""
    models = {}
    for name in MODELS:
        path = out / f"weights_{name}.pt"
        if path.exists():
            model = build(name, heads)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            models[name] = model.to(device).eval()
    return models


def group_outputs(
    models: dict[str, mil.LeadMIL], names: list[str], signals: np.ndarray, rows: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Seed-averaged bag logits and per-finding evidence of a model group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Bag logits (n, heads) and evidence (n, heads, 12, steps).
    """
    bags, logits, weights = [], [], []
    for name in names:
        bag, z, w = mil.predict(models[name], signals, rows, device, keep_instances=True)
        bags.append(bag)
        logits.append(z)
        weights.append(w)
    bag = np.mean(bags, axis=0)
    z = np.mean(logits, axis=0)
    w = None if weights[0] is None else np.mean(weights, axis=0)
    return bag, mil.finding_evidence(z, w)


def heldout_auroc(y: np.ndarray, mask: np.ndarray, bag: np.ndarray, heads: list[str]) -> dict[str, Any]:
    """Per-head held-out AUROC (heads with at least 5 positives) and macro means."""
    macro10, _ = mil.macro_auroc(y, mask, bag, mil.MIN_HELDOUT_POSITIVES)
    _, per_head = mil.macro_auroc(y, mask, bag, 5)
    return {
        "macro_auroc_min10": macro10,
        "per_head": {h: (None if np.isnan(a) else float(a)) for h, a in zip(heads, per_head, strict=True)},
    }


def sanity_checks(
    data: dict[str, Any], models: dict[str, mil.LeadMIL], device: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Held-out detection AUROCs and the four declared sanity checks.

    Returns
    -------
    tuple[dict, dict]
        Detection readings and sanity readings.
    """
    heads = data["heads"]
    abnormal_heads = np.array([k for k, h in enumerate(heads) if h != mil.REFERENCE_HEAD])
    rows = np.flatnonzero(data["heldout"])
    y, mask = data["y"][rows], data["mask"][rows]
    detection = {}
    outputs = {}
    for group, names in GROUPS.items():
        if all(n in models for n in names):
            bag, evidence = group_outputs(models, names, data["signals"], rows, device)
            detection[group] = heldout_auroc(y, mask, bag, heads)
            outputs[group] = (bag, evidence)
    for name in models:
        bag, _, _ = mil.predict(models[name], data["signals"], rows, device)
        detection[name] = heldout_auroc(y, mask, bag, heads)
    bag, evidence = outputs["mil_lse"]
    combined = mil.combined_map(bag, evidence, abnormal_heads, "sum")
    abnormal = (y[:, abnormal_heads] * mask[:, abnormal_heads]).sum(axis=1) > 0
    primary_names = GROUPS["mil_lse"]

    random_models = {name: build(name, len(heads)).to(device).eval() for name in primary_names}
    reinit_models = {}
    for name in primary_names:
        model = build(name, len(heads))
        model.load_state_dict(models[name].state_dict())
        torch.manual_seed(MODELS[name]["seed"] + REINIT_SEED_OFFSET)
        model.instance.reset_parameters()
        reinit_models[name] = model.to(device).eval()
    random_bag, random_evidence = group_outputs(random_models, primary_names, data["signals"], rows, device)
    random_map = mil.combined_map(random_bag, random_evidence, abnormal_heads, "sum")
    reinit_bag, reinit_evidence = group_outputs(reinit_models, primary_names, data["signals"], rows, device)
    reinit_map = mil.combined_map(reinit_bag, reinit_evidence, abnormal_heads, "sum")
    steps = combined.shape[-1]
    amplitude = mil.native_pool(mil.amplitude_map(data["raw"][data["positions"][rows[abnormal]]]), steps)

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "n": int(len(values)),
        }

    rho_random = mil.record_spearman(combined[abnormal], random_map[abnormal])
    rho_reinit = mil.record_spearman(combined[abnormal], reinit_map[abnormal])
    rho_amplitude = mil.record_spearman(combined[abnormal], amplitude)
    rho_amplitude_random = mil.record_spearman(random_map[abnormal], amplitude)
    randomization = {
        "untrained_vs_trained_spearman": summary(rho_random),
        "reinit_instance_layer_vs_trained_spearman": summary(rho_reinit),
        "pass": bool(np.median(rho_random) < 0.5 and np.median(rho_reinit) < 0.5),
    }
    amplitude_check = {
        "trained_vs_amplitude_spearman": summary(rho_amplitude),
        "untrained_vs_amplitude_spearman": summary(rho_amplitude_random),
        "tracks_amplitude": bool(np.median(rho_amplitude) > 0.5),
    }
    lead_check = {}
    lead_index = {lead: i for i, lead in enumerate(LEADS)}
    for head in LEAD_CHECK_HEADS:
        if head not in heads:
            continue
        k = heads.index(head)
        expected = np.zeros(12, bool)
        expected[[lead_index[lead] for lead in REGION_TABLE[head][0]]] = True
        positives = np.flatnonzero((y[:, k] == 1) & (mask[:, k] == 1))
        values = [mil.lead_auroc(evidence[r, k], expected) for r in positives]
        lead_check[head] = {
            "expected_leads": list(REGION_TABLE[head][0]),
            "n": len(values),
            "mean_lead_auroc": float(np.mean(values)) if values else None,
            "share_above_half": float(np.mean(np.array(values) > 0.5)) if values else None,
        }
    eligible = [v for v in lead_check.values() if v["n"] >= 5]
    lead_pass = sum(v["mean_lead_auroc"] > 0.5 for v in eligible)
    noise_values = []
    fit_rows = data["fit"].iloc[rows]
    for position, text in enumerate(fit_rows.noisy_leads):
        named = [lead for lead in text.split(";") if lead]
        if 1 <= len(named) <= 11:
            expected = np.isin(np.array(LEADS), named)
            noise_values.append(mil.lead_auroc(combined[position], expected))
    sanity = {
        "rows": {"heldout": int(len(rows)), "abnormal": int(abnormal.sum())},
        "randomization": randomization,
        "amplitude": amplitude_check,
        "expected_leads": {
            "per_head": lead_check,
            "heads_with_5_positives": len(eligible),
            "heads_above_half": int(lead_pass),
            "pass": bool(lead_pass >= 9),
        },
        "noise": {
            "n": len(noise_values),
            "mean_noisy_lead_auroc": float(np.mean(noise_values)) if noise_values else None,
        },
    }
    return detection, sanity


def reload_check(
    out: Path, data: dict[str, Any], models: dict[str, mil.LeadMIL], device: str
) -> dict[str, float]:
    """Largest difference between reloaded networks' held-out bag logits and those saved after training."""
    rows = np.flatnonzero(data["heldout"])
    result = {}
    for name, model in models.items():
        with np.load(out / f"heldout_bags_{name}.npz") as saved:
            if not np.array_equal(saved["ecg_id"], data["ecg_ids"][rows]):
                raise ValueError(f"Held-out rows changed for {name}")
            bag, _, _ = mil.predict(model, data["signals"], rows, device)
            result[name] = float(np.abs(bag - saved["bag"]).max())
    if max(result.values()) > 1e-4:
        raise ValueError(f"Reloaded networks do not reproduce held-out logits: {result}")
    return result


def scored_sets(smoke: bool) -> dict[str, np.ndarray]:
    """ECG IDs of the ECGAD development records and the evaluation partition."""
    split = load_split()
    partition = load_partition()
    sets = {
        "ecgad_dev": split.ecg_id[split.partition == "development"].to_numpy(),
        "evaluation": partition.ecg_id[partition.partition == "evaluation"].to_numpy(),
    }
    if smoke:
        sets = {name: ids[:16] for name, ids in sets.items()}
    return sets


def write_maps(
    out: Path,
    data: dict[str, Any],
    models: dict[str, mil.LeadMIL],
    device: str,
    sets: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    Write combined maps of every variant (float16, (n, 12, 5000)), bag logits and per-finding maps.

    Returns
    -------
    dict
        File names, hashes and shapes.
    """
    raw, index = data["raw"], open_cache()[1]
    row_of = dict(zip(index.ecg_id, range(len(index)), strict=True))
    partition = load_partition().set_index("ecg_id")
    heads = data["heads"]
    abnormal_heads = np.array([k for k, h in enumerate(heads) if h != mil.REFERENCE_HEAD])
    groups = {g: names for g, names in GROUPS.items() if all(n in models for n in names)}
    variants = {v: spec for v, spec in VARIANTS.items() if spec[0] in groups}
    files: dict[str, Any] = {}
    for set_name, ecg_ids in sets.items():
        positions = np.array([row_of[e] for e in ecg_ids])
        dtype = np.float32 if set_name == "ecgad_dev" else np.float16
        maps = {
            v: np.lib.format.open_memmap(
                out / f"maps_{v}_{set_name}.npy.part",
                mode="w+",
                dtype=dtype,
                shape=(len(ecg_ids), 12, 5000),
            )
            for v in variants
        }
        if set_name == "ecgad_dev":
            finding_keep = np.ones(len(ecg_ids), bool)
        else:
            sub = partition.loc[ecg_ids]
            finding_keep = ((sub.region_statements != "") | (sub.noisy_leads != "")).to_numpy()
        bags = {g: [] for g in groups}
        finding_maps, finding_ids = [], []
        for start in range(0, len(ecg_ids), CHUNK):
            chunk = np.arange(start, min(start + CHUNK, len(ecg_ids)))
            signals = mil.preprocess(raw[positions[chunk]])
            local = np.arange(len(chunk))
            for group, names in groups.items():
                bag, evidence = group_outputs(models, names, signals, local, device)
                bags[group].append(bag)
                for variant, (vgroup, mode) in variants.items():
                    if vgroup == group:
                        native = mil.combined_map(bag, evidence, abnormal_heads, mode)
                        maps[variant][chunk] = mil.upsample(native).astype(dtype)
                if group == "mil_lse":
                    keep = finding_keep[chunk]
                    finding_maps.append(evidence[keep].astype(np.float16))
                    finding_ids.append(ecg_ids[chunk][keep])
            print(f"{set_name}: {chunk[-1] + 1}/{len(ecg_ids)}", flush=True)
        for variant, array in maps.items():
            array.flush()
            del array
            final = out / f"maps_{variant}_{set_name}.npy"
            os.replace(out / f"maps_{variant}_{set_name}.npy.part", final)
            check = np.load(final, mmap_mode="r")
            if not np.isfinite(check).all():
                raise ValueError(f"Non-finite map {final}")
            files[f"maps_{variant}_{set_name}"] = {
                "file": final.name,
                "shape": list(check.shape),
                "sha256": sha256_file(final),
            }
        bag_file = out / f"bags_{set_name}.npz"
        write_npz_atomic(
            bag_file, ecg_id=ecg_ids, heads=np.array(heads), **{g: np.concatenate(b) for g, b in bags.items()}
        )
        finding_file = out / f"finding_maps_{set_name}.npz"
        write_npz_atomic(
            finding_file,
            ecg_id=np.concatenate(finding_ids),
            heads=np.array(heads),
            evidence=np.concatenate(finding_maps),
        )
        np.save(out / f"maps_index_{set_name}.npy", ecg_ids)
        files[f"bags_{set_name}"] = {"file": bag_file.name, "sha256": sha256_file(bag_file)}
        files[f"finding_maps_{set_name}"] = {"file": finding_file.name, "sha256": sha256_file(finding_file)}
    return files


def maps_phase(smoke: bool) -> None:
    """Reload networks, run the sanity checks, write the maps and the pre-scoring result."""
    out = output_dir(smoke)
    start = time.time()
    data = fit_data(smoke)
    device = device_setup()
    with gpu_lock(device):
        models = load_models(out, len(data["heads"]), device)
        reload = reload_check(out, data, models, device)
        if smoke:
            GROUPS["mil_lse"] = ["lse_65065"]
        detection, sanity = sanity_checks(data, models, device)
        files = write_maps(out, data, models, device, scored_sets(smoke))
    histories = {name: json.loads((out / f"history_{name}.json").read_text()) for name in models}
    result = {
        "experiment": "065",
        "protocol": "docs/experiment-065-finding-mil.md",
        "protocol_sha256": sha256_file(ROOT / "docs/experiment-065-finding-mil.md"),
        "git_head": git_head(ROOT),
        "created": utc_now(),
        "smoke": smoke,
        "inputs": {
            "partition": to_stored(PARTITION_PATH),
            "partition_sha256": sha256_file(PARTITION_PATH),
            "cache_dir": str(CACHE_DIR),
            "cache_records": int(json.loads((CACHE_DIR / "complete.json").read_text())["records"]),
        },
        "rows": {
            "fit": int(len(data["ecg_ids"])),
            "train": int((~data["heldout"]).sum()),
            "heldout": int(data["heldout"].sum()),
            "heldout_patients": int(data["fit"].patient_id[data["heldout"]].nunique()),
        },
        "heads": data["heads"],
        "counts": data["counts"],
        "training": {
            name: {k: h[k] for k in ("best_epoch", "best_heldout_macro_auroc", "seconds")}
            | {"epochs": len(h["history"])}
            for name, h in histories.items()
        },
        "histories": {name: h["history"] for name, h in histories.items()},
        "reload_max_abs_difference": reload,
        "heldout_detection": detection,
        "sanity": sanity,
        "files": files,
        "maps_phase_seconds": time.time() - start,
    }
    write_json_atomic(out / "result.json", result)
    print(
        json.dumps(
            {"sanity": sanity, "macro": {g: d["macro_auroc_min10"] for g, d in detection.items()}}, indent=1
        )
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["train", "maps"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.phase == "train":
        train_phase(args.smoke)
    else:
        maps_phase(args.smoke)


if __name__ == "__main__":
    main()
