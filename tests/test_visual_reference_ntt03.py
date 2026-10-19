"""Coordinate and leakage controls for compact visual reference transfer."""

import numpy as np
import pytest
import torch

from ecg_experiment.visual_reference_ntt03 import (
    interval_iou,
    linked_lead,
    pool_region,
    render_trace,
    token_features,
)


def test_explicit_v1_overrides_unlinked_v5_answer():
    row = {"question": "Measure the S amplitude in lead V1?", "step_lead_answers_unlinked": [["Lead V5"]]}
    assert linked_lead(row) == ("V1", "explicit_question")


def test_selected_leads_need_one_unambiguous_answer():
    row = {"question": "Within the selected leads, locate QRS?", "step_lead_answers_unlinked": [["Lead V5"]]}
    assert linked_lead(row) == ("V5", "selected_lead_answer")
    row["step_lead_answers_unlinked"] = [["Lead V5", "Lead V6"]]
    assert linked_lead(row)[0] is None


def test_unspecified_and_slash_leads_remain_unresolved():
    for question in ["Locate the wave?", "Measure in lead V5/V6?", "Measure in lead I and aVL?"]:
        assert linked_lead({"question": question, "step_lead_answers_unlinked": []})[0] is None


def test_deadline_stops_before_visual_model_call():
    with pytest.raises(TimeoutError, match="budget"):
        token_features(None, np.zeros((518, 518, 3), dtype=np.uint8), deadline=0)


@pytest.mark.parametrize("interval", [[0.01, 0.11], [4.1, 4.3], [9.8, 9.99]])
def test_crop_coordinates_contain_interval_without_answer_based_input(interval):
    signal = np.sin(np.arange(1000) / 10)
    result = render_trace(signal, interval, True)
    assert result["start"] <= interval[0] < interval[1] <= result["start"] + result["duration"]
    assert result["image"].shape == (518, 518, 3)
    assert result["duration"] == 1


def test_soft_occupancy_preserves_single_pixel_support():
    mask = np.zeros((518, 518))
    mask[110, 111] = 1
    tokens = torch.ones((1369, 4))
    np.testing.assert_allclose(pool_region(tokens, mask), np.full(4, 0.5))
    assert pool_region(tokens, np.zeros_like(mask)) is None


def test_temporal_overlap_is_not_ink_mask_accuracy():
    assert interval_iou([1, 2], [1.5, 2.5]) == pytest.approx(1 / 3)
    assert interval_iou([1, 2], [3, 4]) == 0
