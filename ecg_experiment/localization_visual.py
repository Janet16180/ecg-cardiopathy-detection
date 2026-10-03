"""Render real localization intermediates in a portable interactive explanation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.localization_visual_data import build_visual_payload


def _display_array(values: list) -> list:
    """Round drawing coordinates without changing saved scores or selected intervals."""
    return np.round(np.asarray(values, dtype=np.float64), 6).tolist()


def display_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the real arrays needed by the browser and preserve exact decision evidence."""
    cases = []
    for case in payload["cases"]:
        shown = dict(case)
        steps = dict(case["steps"])
        shown["signal"] = _display_array(case["signal"])
        if case["kind"] == "incart":
            for name in ("signal", "time", "difference_mv", "normalized_difference"):
                steps.pop(name)
            for name in ("original_beats", "predictions", "residual_energy", "field"):
                steps[name] = _display_array(steps[name])
            steps["windows"] = {"aligned_residual": steps["windows"]["aligned_residual"]}
        else:
            for name in ("filtered_signal", "derivative_envelope"):
                steps[name] = _display_array(steps[name])
        shown["steps"] = steps
        cases.append(shown)
    return {
        "cases": cases,
        "metrics": payload["metrics"],
        "provenance": payload["provenance"],
        "display_note": "Waveform and intermediate drawing arrays rounded to six decimal places; "
        "saved scores, window coordinates and expert annotations retained unchanged.",
    }


def build_visual_explanation(root: Path, destination: Path) -> Path:
    """Write an offline HTML page containing real data and all interaction code."""
    payload = display_payload(build_visual_payload(root))
    template = (root / "notebooks/resources/localization-explainer.html").read_text()
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("__REAL_LOCALIZATION_DATA__", encoded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html)
    return destination
