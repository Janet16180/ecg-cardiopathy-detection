"""Gate a new canonical ECG before any optional trusted local model loading."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ecg_experiment.ecg_quality import assess
from ecg_experiment.waveforms import LEADS

from .artifacts import EVALUATION
from .config import CACHED, ArtifactMode

BUNDLE = EVALUATION / "model_bundle.joblib"


class Predictor(Protocol):
    """Verified corrective evaluator's local predictor interface."""

    def predict(self, signal: np.ndarray, model_id: str) -> dict[str, Any]:
        """Predict only after receipt verification and unchanged waveform quality admission."""
        ...


def load_predictor(root: Path, mode: ArtifactMode = CACHED) -> Predictor | None:
    """Load only the evaluator's fixed locally generated, receipt-verified bundle."""
    bundle = mode.evaluation_dir / "model_bundle.joblib"
    if not (root / bundle).is_file():
        return None
    try:
        runtime = importlib.import_module("ecg_experiment.cuda_runtime20261028")
        runtime.configure_cuda_runtime()
        module = importlib.import_module(mode.inference_module)
    except ModuleNotFoundError as error:
        if error.name in {mode.inference_module, "ecg_experiment.cuda_runtime20261028"}:
            return None
        raise
    return module.load_bundle(root, bundle)


def analyze(samples: list[list[float | None]], leads: list[str], sample_rate: int, units: str,
            model_id: str, root: Path, predictor: Predictor | None = None,
            enable_inference: bool = True, mode: ArtifactMode = CACHED) -> dict[str, Any]:
    """Validate physical units and quality before returning a development-model threshold result.

    Parameters
    ----------
    samples : list of list
        Canonical 12 by 5000 physical-mV samples. Missing values remain nonfinite.
    leads : list of str
        Explicit canonical lead names in their stored order.
    sample_rate : int
        Must be exactly 500 Hz; this endpoint performs no implicit resampling.
    units : str
        Must be ``mV``.
    model_id : str
        Allowlisted executed model identity, validated by the caller.
    root : Path
        Trusted local artifact root configured by the operator at startup.
    predictor : Predictor or None
        Optional already-verified runtime, principally useful in integration tests.
    enable_inference : bool
        False until the executed aggregate evaluation receipt also passes its gate.
    mode : ArtifactMode
        Fixed startup namespace shared by aggregates, cohorts and the verified model loader.

    Returns
    -------
    dict
        Quality abstention, unavailable-model status, or an uncalibrated score and fixed decision.
    """
    if leads != list(LEADS) or sample_rate != 500 or units != "mV":
        raise ValueError("Use canonical lead order, exactly 500 Hz and physical mV.")
    signal = np.asarray(samples, dtype=np.float64)
    if signal.shape != (12, 5000):
        raise ValueError("Exactly twelve leads and 5000 samples per lead are required.")
    reasons, flags = assess(signal)
    quality = {"reasons": reasons, "review_flags": flags}
    if reasons:
        return {"status": "abstained", "quality": quality, "score": None,
                "message": "Signal quality did not pass. No model score or referral result was produced.",
                "next_action": "repeat_acquisition_or_clinical_review"}
    runtime = None
    if enable_inference:
        runtime = predictor if predictor is not None else load_predictor(root, mode)
    if runtime is None:
        return {"status": "model_unavailable", "quality": quality, "score": None,
                "message": "The verified corrective model bundle has not been generated locally."}
    result = runtime.predict(signal.astype(np.float32), model_id)
    if result.get("abstained"):
        return {"status": "abstained", "quality": {"reasons": result.get("quality_reasons", []),
                                                     "review_flags": flags}, "score": None,
                "message": "The verified model runtime abstained. No referral result was produced.",
                "next_action": "repeat_acquisition_or_clinical_review"}
    if result.get("available") is not True:
        return {"status": "model_unavailable", "quality": quality, "score": None,
                "message": "This corrective model is unavailable in the current local runtime."}
    score, threshold = result["score"], result["threshold"]
    if (not isinstance(score, (int, float)) or not math.isfinite(score) or
            not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or
            result.get("comparator") != ">" or result.get("score_kind") != "uncalibrated_model_score"):
        raise ValueError("The verified runtime returned an incompatible score contract.")
    referred = score > threshold
    if result.get("referred") is not referred:
        raise ValueError("The verified runtime's threshold result is inconsistent.")
    return {"status": "scored", "model_id": model_id, "quality": quality,
            "score": score, "threshold": threshold, "comparison": ">", "flagged": referred,
            "score_kind": "uncalibrated_model_score", "scope": "reused_development",
            "message": "Flagged for follow-up" if referred else "Not flagged at the frozen threshold"}
