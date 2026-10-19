"""Compact, frozen visual reference matching and proposal checks for NTT-03."""

from __future__ import annotations

import hashlib
import re
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import wfdb
from PIL import Image, ImageDraw

from ecg_experiment import ROOT
from ecg_experiment.reference_transfer_ntt02 import inner_fold

DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SAM_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
LEAD_PATTERN = r"aVR|aVL|aVF|V[1-6]|III|II|I"
SIDE = 518


def linked_lead(row: dict[str, Any]) -> tuple[str | None, str]:
    """Resolve only explicit or unambiguous selected-lead question references.

    Parameters
    ----------
    row : dict
        An already permitted NTT-02 question.

    Returns
    -------
    tuple
        Lead or None, and the resolution provenance.
    """
    matches = list(re.finditer(rf"\blead ({LEAD_PATTERN})\b", row["question"]))
    if any(
        re.match(rf"\s*(?:/|and\s+|or\s+|,\s*)(?:{LEAD_PATTERN})\b", row["question"][match.end() :])
        for match in matches
    ):
        return None, "ambiguous_explicit"
    names = [match[1] for match in matches]
    if len(set(names)) == 1:
        return names[0], "explicit_question"
    if names:
        return None, "ambiguous_explicit"
    if "selected lead" not in row["question"]:
        return None, "unresolved"
    answers = row["step_lead_answers_unlinked"]
    if len(answers) == 1 and len(answers[0]) == 1:
        match = re.fullmatch(rf"Lead ({LEAD_PATTERN})", answers[0][0])
        if match:
            return match[1], "selected_lead_answer"
    return None, "ambiguous_selected"


def select_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the prospective class and independent-patient selection rule.

    Parameters
    ----------
    rows : list of dict
        Frozen NTT-02 permitted questions.

    Returns
    -------
    list of dict
        At most four classes with three supports and two targets each.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lead, provenance = linked_lead(row)
        if lead:
            groups[(row["diagnosis"], row["class_key"])].append(
                {**row, "lead": lead, "lead_link": provenance}
            )
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_, key), values in sorted(groups.items()):
        patients = sorted({r["patient_id"] for r in values})
        supports = [p for p in patients if inner_fold(p) != 0]
        targets = [p for p in patients if inner_fold(p) == 0]
        if len(patients) < 6 or len(supports) < 3 or len(targets) < 2:
            continue
        chosen_supports = sorted(supports, key=patient_order)[:3]
        chosen_targets = sorted(targets, key=patient_order)[:2]
        by_patient = {
            p: min((r for r in values if r["patient_id"] == p), key=lambda r: r["qid"])
            for p in [*chosen_supports, *chosen_targets]
        }
        eligible[values[0]["diagnosis"]].append(
            {
                "class_key": key,
                "diagnosis": values[0]["diagnosis"],
                "patients_available": len(patients),
                "supports": [by_patient[p] for p in chosen_supports],
                "targets": [by_patient[p] for p in chosen_targets],
            }
        )
    return [eligible[name][0] for name in sorted(eligible)[:4]]


def patient_order(patient: int) -> str:
    """Return the frozen deterministic sampling order without reading labels.

    Parameters
    ----------
    patient : int
        Already permitted source patient identifier.

    Returns
    -------
    str
        Stable SHA-256 ordering key.
    """
    return hashlib.sha256(f"ntt03:{patient}".encode()).hexdigest()


def load_models() -> tuple[Any, Any]:
    """Load local official checkpoints without any network or environment mutation.

    Returns
    -------
    tuple
        Frozen DINOv2-small and SAM2-Tiny automatic proposal generator.
    """
    base = ROOT / "third_party/ntt_vision"
    sys.path[:0] = [str(base / "runtime"), str(base / f"sources/sam2-{SAM_REVISION}")]
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    dino = torch.hub.load(
        str(base / f"sources/dinov2-{DINO_REVISION}"), "dinov2_vits14", source="local", pretrained=False
    )
    dino.load_state_dict(
        torch.load(base / "checkpoints/dinov2_vits14_pretrain.pth", map_location="cpu", weights_only=True)
    )
    dino.eval()
    sam = build_sam2(
        "configs/sam2/sam2_hiera_t.yaml",
        str(base / "checkpoints/sam2_hiera_tiny.pt"),
        device="cpu",
        apply_postprocessing=False,
    )
    generator = SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=32,
        points_per_batch=8,
        pred_iou_thresh=0.4,
        stability_score_thresh=0.95,
        box_nms_thresh=0.5,
        min_mask_region_area=0,
    )
    return dino, generator


def physical_lead(row: dict[str, Any]) -> np.ndarray:
    """Read one permitted physical signal and remove its full-record median.

    Parameters
    ----------
    row : dict
        Selected question carrying a resolved lead and source waveform path.

    Returns
    -------
    np.ndarray
        One thousand physical mV samples at 100 Hz.
    """
    record = wfdb.rdrecord(str(ROOT / "data/raw/ptb-xl/1.0.3" / row["waveform"]))
    index = [name.upper() for name in record.sig_name].index(row["lead"].upper())
    if record.fs != 100 or record.sig_len != 1000 or record.units[index] != "mV":
        raise ValueError("Unexpected physical ECG format")
    signal = record.p_signal[:, index]
    if not np.isfinite(signal).all():
        raise ValueError("Nonfinite physical ECG")
    return signal - np.median(signal)


def render_trace(signal: np.ndarray, interval: list[float], cropped: bool) -> dict[str, Any]:
    """Draw a physically scaled trace and retain its exact image coordinate mapping.

    Parameters
    ----------
    signal : np.ndarray
        Median-centered physical 100 Hz trace.
    interval : list of float
        Candidate start and stop seconds; used to center each candidate crop only.
    cropped : bool
        True for a one-second candidate crop, false for ten seconds.

    Returns
    -------
    dict
        RGB image, ink mask, start/duration and clipping fraction.
    """
    duration = 1.0 if cropped else 10.0
    start = float(np.clip(np.mean(interval) - 0.5, 0, 9)) if cropped else 0.0
    times = np.arange(len(signal)) / 100
    keep = (times >= start) & (times <= start + duration)
    x = (times[keep] - start) / duration * (SIDE - 1)
    y = (3 - np.clip(signal[keep], -3, 3)) / 6 * (SIDE - 1)
    ink = Image.new("L", (SIDE, SIDE), 0)
    ImageDraw.Draw(ink).line(list(zip(x, y, strict=True)), fill=255, width=2)
    mask = np.asarray(ink).copy() / 255
    rgb = np.repeat((255 * (1 - mask))[:, :, None], 3, axis=2).astype(np.uint8)
    return {
        "image": rgb,
        "ink": mask,
        "start": start,
        "duration": duration,
        "clipped_fraction": float((np.abs(signal[keep]) > 3).mean()),
    }


def token_features(model: Any, image: np.ndarray, deadline: float | None = None) -> torch.Tensor:
    """Extract frozen 37 by 37 DINO patch tokens with ImageNet normalization.

    Parameters
    ----------
    model : object
        Local DINOv2-small model.
    image : np.ndarray
        518-square RGB uint8 image.
    deadline : float or None
        Optional perf-counter deadline checked before each image.

    Returns
    -------
    torch.Tensor
        Patch-token matrix of shape (1369, 384).
    """
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("Visual image budget exhausted")
    batch = torch.from_numpy(image.copy()).permute(2, 0, 1).float()[None] / 255
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    with torch.inference_mode():
        return model.forward_features((batch - mean) / std)["x_norm_patchtokens"][0]


def pool_region(tokens: torch.Tensor, mask: np.ndarray) -> np.ndarray | None:
    """Pool patch tokens by soft mask occupancy, preserving sub-patch thin traces.

    Parameters
    ----------
    tokens : torch.Tensor
        (1369, feature) normalized patch tokens.
    mask : np.ndarray
        Nonnegative full-resolution reference/target occupancy mask.

    Returns
    -------
    np.ndarray or None
        Unit-normalized masked feature, or None for an empty mask.
    """
    weights = functional.interpolate(
        torch.as_tensor(mask.copy(), dtype=torch.float32)[None, None], size=(37, 37), mode="area"
    ).flatten()
    if weights.sum() == 0:
        return None
    vector = ((tokens * weights[:, None]).sum(0) / weights.sum()).numpy()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def candidate_embedding(
    model: Any, signal: np.ndarray, interval: list[float], cropped: bool, deadline: float | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed one candidate using its trace pixels and no target answer.

    Parameters
    ----------
    model : object
        Frozen visual encoder.
    signal : np.ndarray
        Permitted physical trace.
    interval : list of float
        Supplied candidate bounds.
    cropped : bool
        Rendering view.
    deadline : float or None
        Optional perf-counter deadline checked before each image.

    Returns
    -------
    tuple
        Unit feature and rendered coordinate metadata.
    """
    rendered = render_trace(signal, interval, cropped)
    tokens = token_features(model, rendered["image"], deadline)
    seconds = rendered["start"] + np.arange(SIDE) / (SIDE - 1) * rendered["duration"]
    mask = rendered["ink"] * ((seconds >= interval[0]) & (seconds <= interval[1]))[None]
    vector = pool_region(tokens, mask)
    if vector is None:
        raise ValueError("Candidate contains no visible trace pixels")
    return vector, rendered


def interval_iou(first: list[float], second: list[float]) -> float:
    """Compute continuous temporal overlap, independent of rendered mask area.

    Parameters
    ----------
    first, second : list of float
        Positive-width intervals in seconds.

    Returns
    -------
    float
        Intersection divided by union.
    """
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return float(intersection / union) if union else 0.0


def class_prototypes(
    model: Any, supports: list[dict[str, Any]], cropped: bool, deadline: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Average three independent-patient instances into real and wrong-window prototypes.

    Parameters
    ----------
    model : object
        Frozen DINO model.
    supports : list of dict
        Three patients from nonzero inner folds.
    cropped : bool
        Fixed rendering view.
    deadline : float or None
        Optional perf-counter deadline checked before each image.

    Returns
    -------
    tuple
        Correct-support and wrong-window unit class prototypes.
    """
    correct, wrong = [], []
    for row in supports:
        signal = physical_lead(row)
        alternative = next(i for i in range(len(row["intervals"])) if i != row["answer"])
        correct.append(
            candidate_embedding(model, signal, row["intervals"][row["answer"]], cropped, deadline)[0]
        )
        wrong.append(candidate_embedding(model, signal, row["intervals"][alternative], cropped, deadline)[0])
    real, null = np.mean(correct, axis=0), np.mean(wrong, axis=0)
    return real / np.linalg.norm(real), null / np.linalg.norm(null)


def match_candidates(
    model: Any,
    row: dict[str, Any],
    prototypes: tuple[np.ndarray, np.ndarray],
    cropped: bool,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Rank supplied candidate regions with reference features in a fixed view.

    Parameters
    ----------
    model : object
        Frozen visual encoder.
    row : dict
        Permitted target question; its answer is used only after ranking.
    prototypes : tuple
        Correct and wrong reference prototypes.
    cropped : bool
        Rendering view.
    deadline : float or None
        Optional perf-counter deadline checked before each image.

    Returns
    -------
    dict
        Candidate scores, predictions, exact agreement and rendering metadata.
    """
    signal = physical_lead(row)
    embeddings = [
        candidate_embedding(model, signal, interval, cropped, deadline) for interval in row["intervals"]
    ]
    vectors = np.stack([value[0] for value in embeddings])
    chosen = [int(np.argmax(vectors @ prototype)) for prototype in prototypes]
    return {
        "qid": row["qid"],
        "patient_id": row["patient_id"],
        "lead": row["lead"],
        "cropped": cropped,
        "answer": row["answer"],
        "real_choice": chosen[0],
        "wrong_choice": chosen[1],
        "real_accuracy": float(chosen[0] == row["answer"]),
        "wrong_accuracy": float(chosen[1] == row["answer"]),
        "real_scores": (vectors @ prototypes[0]).tolist(),
        "wrong_scores": (vectors @ prototypes[1]).tolist(),
        "candidate_clipping": [value[1]["clipped_fraction"] for value in embeddings],
        "candidate_coordinates": [
            {"start": value[1]["start"], "duration": value[1]["duration"]} for value in embeddings
        ],
    }


def proposal_trial(
    model: Any, generator: Any, row: dict[str, Any], prototype: np.ndarray, deadline: float | None = None
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    """Separate automatic-mask proposal coverage from reference-based mask choice.

    Parameters
    ----------
    model, generator : object
        Frozen visual feature and SAM proposal models.
    row : dict
        First permitted target for a selected class.
    prototype : np.ndarray
        Full-view correct-support class vector.
    deadline : float or None
        Optional perf-counter deadline checked before each image.

    Returns
    -------
    tuple
        Descriptive metrics, original rendered RGB and chosen mask if any.
    """
    rendered = render_trace(physical_lead(row), [0, 10], False)
    tokens = token_features(model, rendered["image"], deadline)
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("Visual image budget exhausted")
    masks = sorted(
        generator.generate(rendered["image"].copy()), key=lambda item: item["predicted_iou"], reverse=True
    )[:100]
    candidates = []
    for index, item in enumerate(masks):
        mask = item["segmentation"]
        columns = np.flatnonzero(mask.any(axis=0))
        if not len(columns):
            continue
        interval = [float(columns[0] / (SIDE - 1) * 10), float(columns[-1] / (SIDE - 1) * 10)]
        feature = pool_region(tokens, mask * rendered["ink"])
        candidates.append(
            {
                "mask_index": index,
                "interval": interval,
                "reference_cosine": float(feature @ prototype) if feature is not None else None,
                "proposal_quality": float(item["predicted_iou"]),
                "selected_interval_iou": interval_iou(interval, row["intervals"][row["answer"]]),
            }
        )
    valid = [item for item in candidates if item["reference_cosine"] is not None]
    chosen = max(valid, key=lambda item: item["reference_cosine"]) if valid else None
    result = {
        "qid": row["qid"],
        "lead": row["lead"],
        "proposal_count": len(masks),
        "oracle_iou": max((item["selected_interval_iou"] for item in candidates), default=0.0),
        "chosen_iou": chosen["selected_interval_iou"] if chosen else 0.0,
        "chosen_interval": chosen["interval"] if chosen else None,
        "candidates": candidates,
        "clipped_fraction": rendered["clipped_fraction"],
    }
    return result, rendered["image"], masks[chosen["mask_index"]]["segmentation"] if chosen else None
