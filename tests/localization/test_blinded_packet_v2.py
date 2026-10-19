"""Synthetic tests of the separately frozen wider physical renderer."""

import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from ecg_experiment.files import write_json_atomic
from ecg_experiment.localization.blinded_packet_v2 import (
    PAGE_MM,
    audit_case,
    draw_page,
    page_to_signal,
    png_to_signal,
    render_case,
    signal_to_page,
    signal_to_png,
)
from ecg_experiment.localization.review_forms import pending_form


@pytest.mark.parametrize("lead", range(12))
def test_repair_physical_and_pixel_round_trip(lead):
    for seconds, millivolts in [(0, -15), (10, 15), (3.141, -0.127)]:
        page = signal_to_page(lead, seconds, millivolts)
        assert page_to_signal(lead, *page) == pytest.approx((seconds, millivolts), abs=1e-13)
        png = signal_to_png(lead, seconds, millivolts, 1795)
        assert png_to_signal(lead, *png, 1795) == pytest.approx((seconds, millivolts), abs=1e-13)
    first = signal_to_page(lead, 0, 0)
    second = signal_to_page(lead, 1, 1)
    assert np.subtract(second, first) == pytest.approx((25, 10))


def test_all_twelve_unclipped_leads_and_pending_form(tmp_path):
    times = np.arange(5000) / 500
    signal = np.tile((5 * np.sin(times * 2 * np.pi))[:, None], (1, 12))
    original = signal.copy()
    transforms = render_case(signal, "case-001", tmp_path)
    write_json_atomic(tmp_path / "case-001.json", pending_form("case-001"))
    assert all(audit_case(tmp_path, "case-001").values())
    assert transforms["page_mm"] == list(PAGE_MM)
    assert len(list(tmp_path.glob("*.png"))) == 6
    assert len(transforms["panels"]) == 12
    assert {panel["page"] for panel in transforms["panels"]} == set(range(1, 7))
    assert np.array_equal(signal, original)
    for path in tmp_path.glob("*.png"):
        with Image.open(path) as image:
            assert image.width > 2500
            assert image.height > 1700
    metadata = subprocess.check_output(
        ["pdfinfo", "-f", "1", "-l", "6", str(tmp_path / "case-001.pdf")], text=True
    )
    assert len([line for line in metadata.splitlines() if line.startswith("Page") and " size:" in line]) == 6
    assert json.loads((tmp_path / "case-001.json").read_text())["status"] == "pending"


def test_new_range_still_aborts_without_clipping_or_output(tmp_path):
    signal = np.zeros((5000, 12))
    signal[100, 4] = 15.001
    with pytest.raises(ValueError, match="range"):
        render_case(signal, "case-001", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("page", [0, 5])
def test_pixel_convention_matches_actual_artist_and_integer_canvas(page):
    figure = draw_page(np.zeros((5000, 12)), "case-001", page)
    figure.canvas.draw()
    buffer_height = np.asarray(figure.canvas.buffer_rgba()).shape[0]
    for local, lead in enumerate((page * 2, page * 2 + 1)):
        for seconds, millivolts in [(0, -15), (10, 15), (3.141, -0.127)]:
            x_display, y_display = figure.axes[local].transData.transform((seconds, millivolts))
            expected = (x_display, buffer_height - y_display)
            assert signal_to_png(lead, seconds, millivolts, buffer_height) == pytest.approx(
                expected, abs=1e-10
            )
    plt.close(figure)
