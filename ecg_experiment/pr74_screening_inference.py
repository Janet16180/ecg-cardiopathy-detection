"""Verify a trusted local corrective model bundle and abstain on unusable ECGs."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from . import ROOT
from .cpu_screening041 import RAW_FEATURES
from .ecg_quality import assess
from .full_development import predict
from .measurement_extractor042 import extract_array
from .pr74_screening_io import OUTPUT, completed, verify_hashes
from .xecg import load_xecg, preprocess_xecg

BUNDLE = OUTPUT / "evaluate/model_bundle.joblib"


class VerifiedPredictor:
    """Serve a receipt-verified artifact with fixed hospital-calibration operating points."""

    def __init__(self, input_root: Path, bundle: dict[str, Any]) -> None:
        """Retain only locally generated verified model objects and their fixed thresholds."""
        if bundle.get("schema_version") != 1 or bundle.get("features") != list(RAW_FEATURES):
            raise ValueError("Unsupported corrective model bundle schema")
        self.input_root = input_root
        self.bundle = bundle
        self.encoder = None

    def vector(self, signal: np.ndarray) -> np.ndarray:
        """Compute one frozen xECG representation on CPU without acquiring or changing GPU resources."""
        if self.encoder is None:
            self.encoder = load_xecg(
                self.input_root / "third_party/checkpoints/xecg", device="cpu", backend="vanilla",
                xlstm_dir=self.input_root / "third_party/xecg-deps").eval()
            self.encoder.requires_grad_(False)
        with torch.inference_mode(), threadpool_limits(limits=2):
            tensor = torch.from_numpy(preprocess_xecg(signal)[None])
            pooled, _ = self.encoder(tensor)
        vector = pooled.float().numpy()
        if vector.shape != (1, 1024) or not np.isfinite(vector).all():
            raise ValueError("Nonfinite encoder representation")
        return vector

    def predict(self, signal_12x5000_mv: np.ndarray, model_id: str) -> dict[str, Any]:
        """Return an uncalibrated score and strict frozen rule, or abstain before any prediction."""
        if model_id not in self.bundle["models"]:
            raise ValueError("Unknown local corrective model ID")
        signal = np.asarray(signal_12x5000_mv, dtype=np.float64)
        if signal.shape != (12, 5000) or not np.isfinite(signal).all():
            raise ValueError("Expected finite 12-by-5000 canonical mV waveform")
        exclusions, reviews = assess(signal.astype(np.float32))
        if exclusions:
            return {"available": True, "abstained": True, "quality_reasons": exclusions,
                    "review_flags": reviews}
        features = extract_array(signal.T.copy())
        if features["status"] != "ok":
            return {"available": True, "abstained": True, "quality_reasons": [features["status"]],
                    "review_flags": reviews}
        descriptor = self.bundle["models"][model_id]
        fitted = self.bundle["fitted"][descriptor["cohort"]]
        kind = descriptor["kind"]
        scores = {}
        with threadpool_limits(limits=2):
            if kind in ("hgb", "blend"):
                raw = pd.DataFrame([features]).reindex(columns=RAW_FEATURES)
                raw = pd.read_csv(io.StringIO(raw.to_csv(index=False, float_format="%.6g")))
                scores["hgb"] = float(fitted["hgb"].predict_proba(raw)[0, 1])
            if kind in ("xecg", "blend"):
                scores["xecg"] = float(predict(fitted["xecg"], self.vector(signal))[0])
        score = (0.5 * scores["hgb"] + 0.5 * scores["xecg"] if kind == "blend" else scores[kind])
        threshold = float(self.bundle["thresholds"][model_id]["sph"]["threshold"])
        return {"available": True, "abstained": False, "score": score, "threshold": threshold,
                "comparator": ">", "referred": bool(score > threshold),
                "score_kind": "uncalibrated_model_score", "quality_reasons": [], "review_flags": reviews,
                "calibration_scope": "SPH reused development normals; not an Italian student calibration"}


def load_bundle(input_root: Path, relative_bundle_path: Path = BUNDLE) -> VerifiedPredictor:
    """Load the fixed trusted local artifact after checking stage and scientific source hashes."""
    if relative_bundle_path != BUNDLE:
        raise ValueError("Only the fixed corrective bundle path is allowed")
    receipt, _ = completed(input_root, OUTPUT / "evaluate")
    verify_hashes(ROOT, receipt["source_sha256"])
    verify_hashes(input_root, receipt["input_sha256"])
    return VerifiedPredictor(input_root, joblib.load(input_root / BUNDLE))
