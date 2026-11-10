"""Training-reference composition, geometry and local waveform atlas for KEY-03."""

from __future__ import annotations

import subprocess
import time
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import ROOT
from .condition_key_inputs import CACHE, COHORT, json_read, positions, verify_inputs
from .condition_keys import BATCH, GROUPS, ranking_metrics, spans, unit
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .long_section_keys import windows
from .pr74_screening_io import verify_hashes
from .pr74_screening_features import read_training_signal
from .waveforms import LEADS

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = ROOT / "outputs/reference_ecg_audit_20261106"
PROTOCOL = "docs/reference-ecg-audit-protocol-2026-11-06.md"


def angle(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle in degrees between nonzero vectors."""
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def bind() -> dict:
    """Require committed sources and bind the two predecessor result artifacts."""
    own = [
        PROTOCOL,
        "ecg_experiment/reference_ecg_audit.py",
        "scripts/experiments/run_reference_ecg_audit.py",
    ]
    sources = {}
    for name in own:
        if subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError(f"Uncommitted source: {name}")
        sources[name] = sha256_file(ROOT / name)
    old = ROOT / "outputs/condition_keys_20261103_v2"
    long = ROOT / "outputs/long_section_keys_20261106"
    checked = 0
    for directory, flag_name in ((old, "query_flags"), (long, "flags")):
        result = json_read(directory / "result.json")
        verify_hashes(ROOT, result["source_sha256"])
        verify_hashes(ROOT, result["artifact_sha256"])
        sources.update(result["source_sha256"])
        arrays = np.load(directory / "artifacts.npz", allow_pickle=False)
        for arm in result["metrics"]:
            for normal, key in ((False, "metrics"), (True, "condition_vs_normal")):
                if ranking_metrics(arrays[flag_name], arrays["score_" + arm], normal) != result[key][arm]:
                    raise ValueError("Saved metric replay mismatch")
                checked += 1
    sources["ecg_experiment/pr74_screening_features.py"] = sha256_file(
        ROOT / "ecg_experiment/pr74_screening_features.py"
    )
    return {
        "source_sha256": sources,
        "metric_dictionaries_reproduced": checked,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def inputs() -> tuple:
    """Align training-only metadata and caches with the saved support selections."""
    frame = pd.read_csv(ROOT / COHORT, dtype=str, keep_default_na=False)
    if len(frame) != 25000 or not frame["split"].eq("train").all() or frame["source"].eq("sph").any():
        raise ValueError("Reference cohort differs")
    labels = pd.read_csv(ROOT / CACHE / "qualify/labels.csv").set_index("record_id")
    flags = labels.loc[frame["record_id"], list(GROUPS)].to_numpy(dtype=int)
    if not np.array_equal(flags.max(axis=1), frame["target"].to_numpy(dtype=int)):
        raise ValueError("Label union mismatch")
    rows = pd.read_csv(ROOT / CACHE / "extract_training/rows.csv")
    index = positions(rows["record_id"], frame["record_id"])
    token = np.load(ROOT / CACHE / "extract_training/tokens.npy", mmap_mode="r")
    pooled = np.load(ROOT / CACHE / "extract_training/pooled.npy", mmap_mode="r")[index]
    old = np.load(ROOT / "outputs/condition_keys_20261103_v2/artifacts.npz", allow_pickle=False)
    longer = np.load(ROOT / "outputs/long_section_keys_20261106/artifacts.npz", allow_pickle=False)
    if not np.array_equal(old["train_record_id"], frame["record_id"].to_numpy(dtype=str)):
        raise ValueError("Support selection order mismatch")
    scaler = SimpleNamespace(mean_=old["scaler_mean"], scale_=old["scaler_scale"])
    whole = unit((pooled.astype(float) - scaler.mean_) / scaler.scale_)
    arms = {
        "whole": (None, None, old["whole_positive"], old["whole_normal"]),
        "short": (
            spans(),
            old["support_window_selected_keys"],
            old["window_selected_keys_positive"],
            old["window_selected_keys_normal"],
        ),
        "fixed_1s": (
            spans(True),
            old["support_fixed_selected_keys"],
            old["fixed_selected_keys_positive"],
            old["fixed_selected_keys_normal"],
        ),
    }
    for length in (10, 20, 30):
        arms[f"{length / 4:g}s"] = (
            windows(length),
            longer[f"support_{length}"],
            longer[f"positive_{length}"],
            longer[f"normal_{length}"],
        )
    return frame, flags, index, token, whole, scaler, arms


def composition(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    """Count source, identity, validation and quality evidence for one support pool."""
    table = frame.loc[mask]
    known = table.loc[table["patient_identity_known"].eq("True")]
    hashes = table.loc[table["signal_sha256"].ne(""), "signal_sha256"]
    full = table.loc[table["full_signal_sha256"].ne(""), "full_signal_sha256"]
    return {
        "records": len(table),
        "sources": table["source"].value_counts().to_dict(),
        "families": table["family"].value_counts().to_dict(),
        "known_identity_records": len(known),
        "known_patients": known["patient_id"].nunique(),
        "unknown_identity_records": len(table) - len(known),
        "known_patient_extra_record_count": len(known) - known["patient_id"].nunique(),
        "duplicate_canonical_records": int(hashes.duplicated().sum()),
        "duplicate_full_signal_records": int(full.duplicated().sum()),
        "label_evidence": table["label_evidence"].value_counts().to_dict(),
        "individual_human_validation": table["validated_by_human"].value_counts().to_dict(),
        "review_flags": table["review_flags"].value_counts().to_dict(),
    }


def selected(raw: np.ndarray, choice: np.ndarray, candidates: np.ndarray, scaler: Any) -> np.ndarray:
    """Recover selected unit vectors directly from saved windows without new search."""
    prefix = np.concatenate([np.zeros_like(raw[:, :1]), np.cumsum(raw, axis=1)], axis=1)
    bounds = candidates[choice]
    row = np.arange(len(raw))[:, None]
    mean = prefix[row, bounds[:, :, 1]] - prefix[row, bounds[:, :, 0]]
    mean /= (bounds[:, :, 1] - bounds[:, :, 0])[:, :, None]
    return unit((mean - scaler.mean_) / scaler.scale_)


def recover_vectors(
    tokens: np.ndarray,
    index: np.ndarray,
    whole: np.ndarray,
    scaler: Any,
    candidates: Any,
    choices: Any,
    start: int,
    stop: int,
) -> np.ndarray:
    """Recover one batch of either whole or selected vectors, keeping group order."""
    if candidates is None:
        return np.repeat(whole[start:stop, None, :], 4, axis=1)
    raw = tokens[index[start:stop]].astype(float)
    return selected(raw, choices[start:stop], candidates, scaler)


def geometries(
    frame: pd.DataFrame,
    flags: np.ndarray,
    index: np.ndarray,
    tokens: np.ndarray,
    whole: np.ndarray,
    scaler: Any,
    arms: dict,
) -> tuple[dict, dict]:
    """Reconstruct each bank and summarize support coherence and source/patient effects."""
    masks = [flags[:, c] == 1 for c in range(4)] + [~flags.any(axis=1)] * 4
    patient = np.where(
        frame["patient_identity_known"].eq("True"),
        frame["source"] + ":" + frame["patient_id"],
        frame["record_id"],
    )
    families = frame["family"].to_numpy()
    weights = []
    for mask in masks:
        counts = pd.Series(patient[mask]).value_counts()
        weights.append(np.array([1 / counts.get(p, 1) for p in patient]))
    aggregate, private = {}, {}
    for arm, (candidates, choices, positive, normal) in arms.items():
        keys = np.concatenate([positive, normal])
        sums = np.zeros_like(keys)
        by_family = {family: np.zeros_like(keys) for family in np.unique(families)}
        patient_sum, exclusive_sum, mixed_sum = np.zeros_like(keys), np.zeros_like(keys), np.zeros_like(keys)
        cosines = np.full((len(frame), 8), np.nan)
        drift = np.full((len(frame), 4), np.nan)
        for start in range(0, len(frame), BATCH):
            stop = min(start + BATCH, len(frame))
            vectors = recover_vectors(tokens, index, whole, scaler, candidates, choices, start, stop)
            drift[start:stop] = np.sum(vectors * whole[start:stop, None, :], axis=-1)
            for k, mask in enumerate(masks):
                keep = mask[start:stop]
                z = vectors[:, k % 4]
                sums[k] += z[keep].sum(axis=0)
                cosines[start:stop, k][keep] = z[keep] @ keys[k]
                patient_sum[k] += (z[keep] * weights[k][start:stop][keep, None]).sum(axis=0)
                exclusive_sum[k] += z[keep & (flags[start:stop].sum(axis=1) == 1)].sum(axis=0)
                mixed_sum[k] += z[keep & (flags[start:stop].sum(axis=1) > 1)].sum(axis=0)
                for family in by_family:
                    by_family[family][k] += z[keep & (families[start:stop] == family)].sum(axis=0)
        mismatch = float(np.max(np.abs(unit(sums) - keys)))
        if mismatch > 1e-12:
            raise ValueError(f"{arm} key reconstruction mismatch: {mismatch}")
        details = {}
        for k, mask in enumerate(masks):
            group = ("positive_" if k < 4 else "normal_") + GROUPS[k % 4]
            nearest = np.flatnonzero(mask)[np.argsort(-cosines[mask, k], kind="stable")[:50]]
            info = {
                "records": int(mask.sum()),
                "resultant_length": float(np.linalg.norm(sums[k]) / mask.sum()),
                "cosine_quantiles": np.quantile(
                    cosines[mask, k], [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
                ).tolist(),
                "top50_sources": frame.iloc[nearest]["source"].value_counts().to_dict(),
                "equal_patient_rotation_degrees": angle(patient_sum[k], keys[k]),
                "whole_key_rotation_degrees": angle(keys[k], np.concatenate(arms["whole"][2:])[k]),
                "source_removed_rotation_degrees": {
                    str(f): angle(sums[k] - s[k], keys[k])
                    for f, s in by_family.items()
                    if np.linalg.norm(sums[k] - s[k]) > 1e-12
                },
            }
            if k < 4 and np.linalg.norm(exclusive_sum[k]) > 0 and np.linalg.norm(mixed_sum[k]) > 0:
                info["exclusive_vs_cooccurring_angle_degrees"] = angle(exclusive_sum[k], mixed_sum[k])
            if choices is not None:
                bounds = candidates[choices[mask, k % 4]]
                info["start_counts"] = np.bincount(bounds[:, 0], minlength=40).tolist()
                lengths, counts = np.unique(bounds[:, 1] - bounds[:, 0], return_counts=True)
                info["length_counts"] = dict(zip(map(str, lengths), counts.tolist(), strict=True))
                info["source_mean_start_seconds"] = {
                    str(f): float(candidates[choices[mask & (families == f), k % 4], 0].mean() / 4)
                    for f in by_family
                    if np.any(mask & (families == f))
                }
            details[group] = info
        aggregate[arm] = {"key_max_error": mismatch, "groups": details}
        private[f"{arm}_cosine"] = cosines
        private[f"{arm}_whole_alignment"] = drift
        print(f"Reconstructed and audited {arm} keys", flush=True)
    return aggregate, private


def examples(frame: pd.DataFrame, flags: np.ndarray, private: dict) -> list[dict]:
    """Select the frozen four waveform-review roles per annotation group and normal pool."""
    panels = []
    for k, group in enumerate((*GROUPS, "normal")):
        mask = flags[:, k] == 1 if k < 4 else ~flags.any(axis=1)
        cosine = private["whole_cosine"][:, k]
        family_counts = frame.loc[mask, "family"].value_counts()
        for family in family_counts.index[:2]:
            rows = np.flatnonzero(mask & frame["family"].eq(family).to_numpy())
            chosen = rows[np.argmax(cosine[rows])]
            panels.append({"group": group, "role": f"nearest in {family}", "position": int(chosen)})
        rows = np.flatnonzero(mask)
        panels.append(
            {
                "group": group,
                "role": "lowest whole-key similarity",
                "position": int(rows[np.argmin(cosine[rows])]),
            }
        )
        panels.append(
            {
                "group": group,
                "role": "largest short-selection change",
                "position": int(rows[np.argmin(private["short_whole_alignment"][rows, k % 4])]),
            }
        )
    for i, panel in enumerate(panels):
        panel["alias"] = f"Reference {i + 1:02d}"
    return panels


def atlas(frame: pd.DataFrame, flags: np.ndarray, arms: dict, panels: list[dict]) -> dict:
    """Read only checksum-verified training examples and create a private waveform atlas."""
    (OUT / "waveforms").mkdir()
    summaries = []
    loaded = {}
    with PdfPages(OUT / "ReferenceECGAtlas.pdf") as pdf:
        for i, panel in enumerate(panels):
            pos = panel["position"]
            if pos not in loaded:
                loaded[pos] = read_training_signal(ROOT, frame.iloc[pos])
            signal = loaded[pos]
            column = GROUPS.index(panel["group"]) if panel["group"] != "normal" else 0
            fig, axes = plt.subplots(13, 1, figsize=(14, 18), sharex=True)
            for y, name in enumerate(("short", "5s", "7.5s")):
                candidates, choice = arms[name][:2]
                a, b = candidates[choice[pos, column]] / 4
                axes[0].broken_barh([(a, b - a)], (y, 0.7), label=name)
            axes[0].set(yticks=[0.35, 1.35, 2.35], yticklabels=["short", "5s", "7.5s"])
            label = ", ".join(g for c, g in enumerate(GROUPS) if flags[pos, c]) or "normal proxy"
            axes[0].set_title(
                f"{panel['alias']} | {panel['group']} | {panel['role']}\n"
                f"Source: {frame.iloc[pos]['source']} | Labels: {label}\n"
                "Bars: contextual token selections, not clinical boundaries"
            )
            for c, name in enumerate(LEADS):
                axes[c + 1].plot(np.arange(5000) / 500, signal[c], linewidth=0.6, rasterized=True)
                axes[c + 1].set_ylabel(f"{name}\nmV", rotation=0, labelpad=25)
                axes[c + 1].grid(alpha=0.2)
            axes[-1].set(xlabel="Seconds", xlim=(0, 10))
            fig.tight_layout()
            filename = f"waveforms/reference-{i + 1:02d}.png"
            fig.savefig(OUT / filename, dpi=100)
            pdf.savefig(fig)
            plt.close(fig)
            summaries.append(
                {
                    **panel,
                    "file": filename,
                    "record_id": frame.iloc[pos]["record_id"],
                    "source": frame.iloc[pos]["source"],
                    "review_flags": frame.iloc[pos]["review_flags"],
                    "finite": bool(np.isfinite(signal).all()),
                    "flat_leads": int((np.ptp(signal, axis=1) == 0).sum()),
                    "max_abs_mv": float(np.abs(signal).max()),
                }
            )
    write_json_atomic(OUT / "waveform_manifest.json", summaries)
    return {
        "panels": len(panels),
        "unique_waveforms": len(loaded),
        "all_finite": all(s["finite"] for s in summaries),
        "panels_with_flat_leads": sum(s["flat_leads"] > 0 for s in summaries),
    }


def run() -> dict:
    """Execute the frozen descriptive audit and preserve errors separately."""
    binding = bind()
    OUT.mkdir(exist_ok=False)
    write_json_atomic(OUT / "started.json", binding)
    started = time.perf_counter()
    try:
        with threadpool_limits(2):
            hashes = verify_inputs()
            frame, flags, index, token, whole, scaler, arms = inputs()
            groups = {g: composition(frame, flags[:, c] == 1) for c, g in enumerate(GROUPS)}
            groups["normal"] = composition(frame, ~flags.any(axis=1))
            geometry, private = geometries(frame, flags, index, token, whole, scaler, arms)
            panels = examples(frame, flags, private)
            waveforms = atlas(frame, flags, arms, panels)
            write_npz_atomic(OUT / "reference_metrics.npz", **private)
            result = {
                "study": "KEY-03",
                "status": "completed",
                **binding,
                "input_sha256": hashes,
                "composition": groups,
                "overall": composition(frame, np.ones(len(frame), bool)),
                "cooccurrence": (flags.T @ flags).tolist(),
                "exclusive_counts": {
                    g: int(((flags[:, c] == 1) & (flags.sum(axis=1) == 1)).sum())
                    for c, g in enumerate(GROUPS)
                },
                "geometry": geometry,
                "waveform_review": waveforms,
                "wall_seconds": time.perf_counter() - started,
                "clinical_review_performed": False,
                "new_evaluation_scores": False,
            }
            write_json_atomic(OUT / "result.json", result)
            report = ROOT / "reports/reference-ecg-audit-2026-11-06/aggregate.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(report, result)
            return result
    except Exception as error:
        write_json_atomic(OUT / "failure.json", {"error": str(error)})
        raise
