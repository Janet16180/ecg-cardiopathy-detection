"""Reserved-safe same-QRS reference classification without provided evidence intervals."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.signal import butter, filtfilt

from ecg_experiment.clustering01 import train_patient
from ecg_experiment.eda.avance2_features import AFTER_S, BEFORE_S, median_beat, qrs_bounds, qrs_peaks
from ecg_experiment.eda.ptbxl import diagnostic_classes, project_label
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.files import sha256_file
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.visual_reference_ntt03 import DINO_REVISION, pool_region, token_features

FINDINGS = ("NORM", "CRBBB", "CLBBB")
SEEDS = (20261018, 20261019, 20261020)
BASELINES = ("raw", "raw_full_context", "duration", "amplitude", "energy", "combined_scalars")
METHODS = ("dino", *BASELINES, "late_reference", "shuffled_labels")
MANIFEST_SHA = "efaa51a62f52fc2d10881a58994d2da206ae5683cff08872c09cc5ec08ef253c"
POLICY_SHA = "028a5ae4e36c29303b37e9d1268ca365090cb600cabe440b9ce0cc672bf78c5d"
CHECKPOINT_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
QRS_ROI, LATE_ROI = (-0.15, 0.15), (0.25, 0.55)


def digest(text: str) -> str:
    """Return a fixed UTF-8 SHA256 ordering key."""
    return hashlib.sha256(text.encode()).hexdigest()


def allowed_identities(identities: pd.DataFrame) -> pd.DataFrame:
    """Keep PR67 train patients and quarantine whole patients touching source folds 9/10."""
    frame = identities.copy()
    frame["patient_id"] = frame.patient_id.astype(int)
    closed = set(frame.loc[frame.strat_fold > 8, "patient_id"])
    return frame.loc[
        frame.strat_fold.between(1, 8) & ~frame.patient_id.isin(closed) & frame.patient_id.map(train_patient)
    ]


def verify_reservations(metadata: Path, manifest_path: Path, policy_path: Path) -> tuple[pd.DataFrame, dict]:
    """Verify identities and policy bytes before any diagnostic payload is parsed."""
    if sha256_file(manifest_path) != MANIFEST_SHA or sha256_file(policy_path) != POLICY_SHA:
        raise ValueError("Shared reservation bytes changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["metadata_sha256"] != sha256_file(metadata):
        raise ValueError("Source metadata differs from the reservation manifest")
    ids = pd.read_csv(metadata, usecols=["ecg_id", "patient_id", "strat_fold", "filename_hr"])
    allowed = allowed_identities(ids)
    expected = {int(row["ecg_id"]) for row in manifest["records"] if row["role"] == "train"}
    if set(allowed.ecg_id) != expected or len(allowed) != 13938 or allowed.patient_id.nunique() != 12078:
        raise ValueError("Exact training-role identity reproduction failed")
    return allowed.sort_values("ecg_id").drop_duplicates("patient_id").reset_index(drop=True), {
        "all_train_ecg_identities_exact": True,
        "train_ecgs": 13938,
        "train_patients": 12078,
        "patient_first_record_rule": "lowest train ECG ID before label parsing",
    }


def finding(codes: dict, classes: dict[str, str]) -> tuple[str | None, str]:
    """Keep complete BBB confidence >=50 or the unchanged existing strict-normal rule."""
    if "CRBBB" in codes and "CLBBB" in codes:
        return None, "both_complete_bbb_codes"
    for code in FINDINGS[1:]:
        if float(codes.get(code, 0)) >= 50:
            return code, "eligible"
    if project_label(codes, classes) == 0:
        return "NORM", "eligible"
    return None, "outside_three_finding_task"


def labeled_cohort(first_records: pd.DataFrame, metadata: Path) -> tuple[list[dict], dict]:
    """Parse source annotations only for the previously verified first training records."""
    allowed = first_records.set_index("ecg_id")
    classes = diagnostic_classes(pd.read_csv(metadata.with_name("scp_statements.csv"), index_col=0))
    rows, counts = [], Counter()
    with metadata.open() as stream:
        for raw in csv.DictReader(stream):
            ecg = int(raw["ecg_id"])
            if ecg not in allowed.index:
                continue
            code_dict = ast.literal_eval(raw["scp_codes"])
            label, reason = finding(code_dict, classes)
            counts[reason] += 1
            if label is None:
                continue
            identity = allowed.loc[ecg]
            rows.append(
                {
                    "ecg_id": ecg,
                    "patient_id": int(identity.patient_id),
                    "filename_hr": str(identity.filename_hr),
                    "finding": label,
                    "diagnostic_superclasses": sorted({classes[c] for c in code_dict if c in classes}),
                }
            )
            counts[label] += 1
    return rows, dict(counts)


def support_pool(patient: int) -> bool:
    """Assign a patient to one frozen reference/target pool without using its label."""
    return int(digest(f"ntt04_pool_v1:20261018:{patient}"), 16) % 10000 < 5000


def select_patients(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Choose sixty targets and three prespecified independent-patient reference draws."""
    targets, manifests, inventory = [], [], []
    if len({row["patient_id"] for row in rows}) != len(rows):
        raise ValueError("Finding cohort must have one ECG per patient")
    for label in FINDINGS:
        members = [row for row in rows if row["finding"] == label]
        supports = [row for row in members if support_pool(row["patient_id"])]
        candidates = [row for row in members if not support_pool(row["patient_id"])]
        inventory.append({"finding": label, "support_pool": len(supports), "target_pool": len(candidates)})
        if len(supports) < 3 or len(candidates) < 20:
            raise ValueError(f"Insufficient independent support/target patients: {label}")
        targets.extend(
            sorted(candidates, key=lambda row: digest(f"ntt04_targets:20261018:{row['patient_id']}"))[:20]
        )
        for seed in SEEDS:
            chosen = sorted(
                supports, key=lambda row: digest(f"ntt04_support:{seed}:{label}:{row['patient_id']}")
            )[:3]
            manifests.append({"seed": seed, "finding": label, "supports": chosen})
    return targets, manifests, inventory


def physical_median(signal: np.ndarray, fs: int = 500) -> tuple[np.ndarray, dict]:
    """Estimate an automatically centered complete median beat with a fixed baseline."""
    if signal.shape != (10 * fs, 12) or not np.isfinite(signal).all():
        raise ValueError("Expected finite ten-second twelve-lead waveform")
    peaks = qrs_peaks(signal, fs)
    complete = peaks[(peaks >= int(BEFORE_S * fs)) & (peaks + int(AFTER_S * fs) <= len(signal))]
    if len(complete) < 3:
        raise ValueError("Fewer than three complete automatic beats")
    b, a = butter(2, [0.5, 40], btype="band", fs=fs)
    beat, consistency = median_beat(filtfilt(b, a, signal, axis=0), complete, fs)
    baseline = beat[round((BEFORE_S - 0.28) * fs) : round((BEFORE_S - 0.20) * fs)].mean(axis=0)
    beat = beat - baseline
    onset, offset = qrs_bounds(beat, fs)
    if not np.isfinite(beat).all() or not np.isfinite(consistency) or offset <= onset:
        raise ValueError("Invalid automatic median beat or QRS duration")
    return beat, {
        "detected_peaks": len(peaks),
        "complete_beats": len(complete),
        "beat_consistency": consistency,
        "qrs_duration_seconds": (offset - onset) / fs,
    }


def normalized(vector: np.ndarray) -> np.ndarray:
    """Normalize a nonzero finite reference/target vector without fitting a transform."""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Empty or nonfinite morphology vector")
    return vector / norm


def raw_descriptors(beat: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract V1/V6 QRS/full-beat morphology and fixed-unit scalar comparisons."""
    center = int(BEFORE_S * 500)
    qrs = beat[center - 75 : center + 76, [LEADS.index("V1"), LEADS.index("V6")]]
    amplitude = np.ptp(qrs, axis=0)
    energy = np.sqrt(np.mean(np.diff(qrs, axis=0) ** 2, axis=0))
    scalars = np.log(np.maximum([duration, *amplitude, *energy], 1e-8))
    full = beat[:, [LEADS.index("V1"), LEADS.index("V6")]]
    return normalized(qrs.T.ravel()), normalized(full.T.ravel()), scalars


def render_median(beat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Render the identical V1/V6 geometry and two equal-width programmed time masks."""
    side = 518
    times = np.arange(len(beat)) / 500 - BEFORE_S
    ink = Image.new("L", (side, side), 0)
    drawer = ImageDraw.Draw(ink)
    pair = beat[:, [LEADS.index("V1"), LEADS.index("V6")]]
    x = (times + BEFORE_S) / (BEFORE_S + AFTER_S) * (side - 1)
    for panel in range(2):
        y = (panel + 0.5) * side / 2 - np.clip(pair[:, panel], -3, 3) / 3 * 110
        drawer.line(list(zip(x, y, strict=True)), fill=255, width=2)
    mask = np.asarray(ink).copy() / 255
    columns = np.arange(side) / (side - 1) * (BEFORE_S + AFTER_S) - BEFORE_S
    qrs = mask * ((columns >= QRS_ROI[0]) & (columns <= QRS_ROI[1]))[None]
    late = mask * ((columns >= LATE_ROI[0]) & (columns <= LATE_ROI[1]))[None]
    rgb = np.repeat((255 * (1 - mask))[:, :, None], 3, axis=2).astype(np.uint8)
    return (
        rgb,
        qrs,
        late,
        {
            "start_seconds": -BEFORE_S,
            "duration_seconds": BEFORE_S + AFTER_S,
            "lead_panels": ["V1", "V6"],
            "qrs_roi": list(QRS_ROI),
            "late_roi": list(LATE_ROI),
            "clipped_fraction": float((abs(pair) > 3).mean()),
            "qrs_ink_pixels": float(qrs.sum()),
            "late_ink_pixels": float(late.sum()),
        },
    )


def load_dino(base: Path) -> Any:
    """Load the cached public small DINO checkpoint with no SAM, download or weight fit."""
    checkpoint = base / "checkpoints/dinov2_vits14_pretrain.pth"
    if sha256_file(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("DINO checkpoint bytes changed")
    model = torch.hub.load(
        str(base / f"sources/dinov2-{DINO_REVISION}"), "dinov2_vits14", source="local", pretrained=False
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


def image_descriptors(model: Any, beat: np.ndarray, deadline: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pool frozen patch tokens in the programmed QRS/late masks of one median-beat image."""
    rgb, qrs, late, coordinates = render_median(beat)
    tokens = token_features(model, rgb, deadline)
    first, second = pool_region(tokens, qrs), pool_region(tokens, late)
    if first is None or second is None:
        raise ValueError("Programmed time region contains no usable trace ink")
    return first, second, coordinates


def classify(feature_rows: dict[int, dict], targets: list[dict], manifests: list[dict]) -> list[dict]:
    """Evaluate fixed class prototypes and descriptors without updating model weights."""
    predictions = []
    for seed in SEEDS:
        banks = {
            m["finding"]: [feature_rows[row["ecg_id"]] for row in m["supports"]]
            for m in manifests
            if m["seed"] == seed
        }
        prototypes = {
            key: np.stack(
                [normalized(np.mean([row[key] for row in banks[label]], axis=0)) for label in FINDINGS]
            )
            for key in ("dino", "raw", "raw_full_context", "late")
        }
        scalar = np.stack([np.median([row["scalars"] for row in banks[label]], axis=0) for label in FINDINGS])
        for target in targets:
            feature = feature_rows[target["ecg_id"]]
            scores = {
                "dino": prototypes["dino"] @ feature["dino"],
                "raw": prototypes["raw"] @ feature["raw"],
                "raw_full_context": prototypes["raw_full_context"] @ feature["raw_full_context"],
                "late_reference": prototypes["late"] @ feature["dino"],
            }
            scores["shuffled_labels"] = np.roll(scores["dino"], 1)
            for method, columns in (
                ("duration", [0]),
                ("amplitude", [1, 2]),
                ("energy", [3, 4]),
                ("combined_scalars", list(range(5))),
            ):
                distance = np.linalg.norm(scalar[:, columns] - feature["scalars"][columns], axis=1)
                scores[method] = -distance
            chosen = {method: int(np.argmax(values)) for method, values in scores.items()}
            truth = FINDINGS.index(target["finding"])
            predictions.append(
                {
                    "patient_id": target["patient_id"],
                    "ecg_id": target["ecg_id"],
                    "finding": target["finding"],
                    "truth": truth,
                    "seed": seed,
                    "scores": {key: value.tolist() for key, value in scores.items()},
                    "chosen": chosen,
                    "correct": {key: float(value == truth) for key, value in chosen.items()},
                }
            )
    return predictions


def balanced(values: np.ndarray, truth: np.ndarray) -> float:
    """Average correctness within each of the three source-annotation findings."""
    return float(np.mean([values[truth == i].mean() for i in range(3)]))


def exact_gain(predictions: list[dict], first: str, second: str) -> Fraction:
    """Compare point balanced accuracies exactly, including the five-point boundary."""
    differences = []
    for label in FINDINGS:
        rows = [row for row in predictions if row["finding"] == label]
        difference = sum(int(row["correct"][first]) - int(row["correct"][second]) for row in rows)
        differences.append(Fraction(difference, len(rows)))
    return sum(differences) / 3


def summarize(predictions: list[dict]) -> dict:
    """Apply the frozen best-baseline paired patient bootstrap and admission rule."""
    patients = sorted({row["patient_id"] for row in predictions})
    truth = np.array([next(row["truth"] for row in predictions if row["patient_id"] == p) for p in patients])
    correct = {
        method: np.array(
            [np.mean([r["correct"][method] for r in predictions if r["patient_id"] == p]) for p in patients]
        )
        for method in METHODS
    }
    scores = {method: balanced(value, truth) for method, value in correct.items()}
    best = max(BASELINES, key=scores.get)
    groups, rng = patient_groups(np.array(patients)), np.random.default_rng(20261018)
    differences, skipped = [], 0
    for _ in range(2000):
        rows = patient_resample(groups, rng)
        if len(np.unique(truth[rows])) != 3:
            skipped += 1
            continue
        alternative = max(balanced(correct[method][rows], truth[rows]) for method in BASELINES)
        differences.append(balanced(correct["dino"][rows], truth[rows]) - alternative)
    if not differences:
        raise ValueError("No bootstrap draw retained all three findings")
    low, high = np.percentile(differences, [2.5, 97.5])
    exact_difference = exact_gain(predictions, "dino", best)
    difference = float(exact_difference)
    confusions = {}
    for method in METHODS:
        matrix = np.zeros((3, 3), dtype=int)
        for row in predictions:
            matrix[row["truth"], row["chosen"][method]] += 1
        confusions[method] = {
            "counts": matrix.tolist(),
            "row_normalized": (matrix / matrix.sum(axis=1, keepdims=True)).tolist(),
        }
    return {
        "balanced_accuracy": scores,
        "best_baseline": best,
        "primary": {
            "difference": difference,
            "ci_low": float(low),
            "ci_high": float(high),
            "draws": 2000,
            "skipped_draws": skipped,
            "seed": 20261018,
            "baseline_maximum_recomputed_each_draw": True,
        },
        "advance_gate_passed": exact_difference >= Fraction(1, 20) and low > 0,
        "point_gate_uses_exact_rational_accuracy": True,
        "confusions": confusions,
        "by_draw": {
            str(seed): {
                m: float(np.mean([r["correct"][m] for r in predictions if r["seed"] == seed]))
                for m in METHODS
            }
            for seed in SEEDS
        },
        "by_finding": {
            label: {
                m: float(np.mean([r["correct"][m] for r in predictions if r["finding"] == label]))
                for m in METHODS
            }
            for label in FINDINGS
        },
    }
