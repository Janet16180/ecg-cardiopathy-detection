"""Synthetic packet checks; no real patient selection or waveform access."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ecg_experiment.files import write_json_atomic
from ecg_experiment.localization.blinded_packet import (
    PAGE_MM,
    audit_reviewer_case,
    page_to_signal,
    png_to_signal,
    render_case,
    select_cases,
    signal_to_page,
    signal_to_png,
)
from ecg_experiment.localization.review_forms import pending_form, validate_pending_form


def test_selection_depends_only_on_identity_and_anchor_membership():
    ids = pd.DataFrame(
        {"patient_id": range(1000, 1500), "ecg_id": range(500), "arbitrary_payload": "ignored"}
    )
    anchors = np.arange(400)
    selected = select_cases(ids, anchors)
    assert len(selected) == len({row["patient_id"] for row in selected}) == 30
    assert all(row["patient_id"] < 1400 for row in selected)
    ids["arbitrary_payload"] = "different"
    changed = select_cases(ids, anchors)
    assert [row["patient_id"] for row in changed] == [row["patient_id"] for row in selected]
    reordered = ids.iloc[::-1].reset_index(drop=True)
    reselected = select_cases(reordered, np.flatnonzero(reordered.patient_id < 1400))
    assert [row["patient_id"] for row in reselected] == [row["patient_id"] for row in selected]


@pytest.mark.parametrize("lead", range(12))
def test_coordinate_round_trip_and_physical_scale(lead):
    for seconds, millivolts in [(0, -4), (10, 4), (3.141, -0.127)]:
        x, y = signal_to_page(lead, seconds, millivolts)
        assert page_to_signal(lead, x, y) == pytest.approx((seconds, millivolts), abs=1e-13)
    origin = signal_to_page(lead, 0, 0)
    shifted = signal_to_page(lead, 1, 1)
    assert np.subtract(shifted, origin) == pytest.approx((25, 10))


def test_synthetic_rendering_and_actual_artifact_privacy(tmp_path):
    times = np.arange(5000) / 500
    signal = np.tile((0.2 * np.sin(times * 2 * np.pi))[:, None], (1, 12))
    transform = render_case(signal, "case-001", tmp_path)
    write_json_atomic(tmp_path / "case-001.json", pending_form("case-001"))
    assert all(audit_reviewer_case(tmp_path, "case-001").values())
    assert transform["page_mm"] == list(PAGE_MM)
    with Image.open(tmp_path / "case-001.png") as image:
        assert image.width > 2500
        assert image.height > 2500
    metadata = subprocess.check_output(["pdfinfo", str(tmp_path / "case-001.pdf")], text=True)
    page_size = next(line for line in metadata.splitlines() if line.startswith("Page size:"))
    numbers = page_size.split()[2:5:2]
    assert [float(value) for value in numbers] == pytest.approx(np.array(PAGE_MM) * 72 / 25.4, abs=0.01)
    form = json.loads((tmp_path / "case-001.json").read_text())
    validate_pending_form(form)


def test_out_of_range_has_no_replacement_or_clipping(tmp_path):
    signal = np.zeros((5000, 12))
    signal[100, 4] = 4.001
    with pytest.raises(ValueError, match="range"):
        render_case(signal, "case-001", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("lead", range(12))
def test_png_round_trip_preserves_original_coordinates(lead):
    for seconds, millivolts in [(0, -4), (10, 4), (3.141, -0.127)]:
        x, y = signal_to_png(lead, seconds, millivolts, 2787)
        assert png_to_signal(lead, x, y, 2787) == pytest.approx((seconds, millivolts), abs=1e-13)
        rounded = png_to_signal(lead, round(x), round(y), 2787)
        assert abs(rounded[0] - seconds) <= 25.4 / 120 / 25 / 2
        assert abs(rounded[1] - millivolts) <= 25.4 / 120 / 10 / 2
