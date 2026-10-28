"""Load only the fixed receipt-verified fresh-reproduction bundle for the application."""

from pathlib import Path

import joblib

from .pr74_fresh_baseline import stage_inputs
from .pr74_raw_study import OUTPUT
from .pr74_screening_inference import VerifiedPredictor

BUNDLE = OUTPUT / "evaluate/model_bundle.joblib"


def load_bundle(input_root: Path, relative_bundle_path: Path = BUNDLE) -> VerifiedPredictor:
    """Verify honest fresh-stage provenance without copying or relabeling cached completions."""
    if relative_bundle_path != BUNDLE:
        raise ValueError("Only the fixed fresh corrective bundle path is allowed")
    stage_inputs(input_root, OUTPUT, "evaluate")
    return VerifiedPredictor(input_root, joblib.load(input_root / BUNDLE))
