import numpy as np
import pandas as pd
import pytest
import torch

from ecg_experiment.ann_heads import AttentionHead
from ecg_experiment.finding_screen import split_thresholds
from ecg_experiment.pipeline_v4 import (
    candidate_reading,
    ensemble_logit,
    load_state,
    matched_rate_screen,
    retrain_agreement,
    state_arrays,
    summarize_screens,
)


def test_ensemble_logit_is_the_mean() -> None:
    assert np.array_equal(ensemble_logit(np.array([1.0, -3.0]), np.array([3.0, 1.0])), np.array([2.0, -1.0]))
    with pytest.raises(ValueError, match="same shape"):
        ensemble_logit(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        ensemble_logit(np.array([np.nan]), np.zeros(1))


def test_candidate_reading_has_three_outcomes() -> None:
    assert candidate_reading([0.001, 0.01], 0.005, "v4", "v3") == "adopt_v4"
    assert candidate_reading([0.0, 0.01], 0.005, "v4", "v3") == "no_worse_keep_v3"
    assert candidate_reading([-0.004, 0.01], 0.005, "v4", "v3") == "no_worse_keep_v3"
    assert candidate_reading([-0.005, 0.01], 0.005, "v4", "v3") == "keep_v3"


def test_retrain_agreement_reports_identity_and_auroc() -> None:
    rng = np.random.default_rng(0)
    y = (np.arange(200) % 2).astype(np.int64)
    old = y + rng.normal(size=200)
    same = retrain_agreement(old.copy(), old, y)
    assert same["identical"]
    assert same["r"] == pytest.approx(1.0)
    assert same["auroc_difference"] == 0.0
    moved = retrain_agreement(old + 0.01 * rng.normal(size=200), old, y)
    assert not moved["identical"]
    assert 0.99 < moved["r"] < 1.0
    assert moved["max_abs_difference"] > 0


def test_saved_states_reload_exactly() -> None:
    torch.manual_seed(1)
    model = AttentionHead(16)
    states = {7: {key: value.detach().numpy().copy() for key, value in model.state_dict().items()}}
    arrays = state_arrays(states)
    assert all(key.startswith("seed7.") for key in arrays)
    torch.manual_seed(2)
    fresh = load_state(AttentionHead(16), arrays, 7)
    tokens = torch.randn(3, 5, 16)
    model.eval()
    fresh.eval()
    assert torch.equal(model(tokens)[0], fresh(tokens)[0])


def test_matched_rate_screen_with_unit_counts_matches_a_direct_screen() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(300, 2))
    normal = np.arange(300) < 200
    masks = {"normal": normal, "other": ~normal}
    found = matched_rate_screen(matrix, np.ones(300), normal, masks, 50, (500,))
    referred = (matrix > split_thresholds(matrix[normal], 50, (500,))).any(axis=1)
    assert found == {"normal": referred[normal].mean(), "other": referred[~normal].mean()}
    doubled = matched_rate_screen(matrix, np.full(300, 2.0), normal, masks, 50, (500,))
    assert doubled["other"] == pytest.approx(found["other"])


def test_summarize_screens_pairs_pipelines_and_rules() -> None:
    keys = [(pipeline, rule, 10, 50) for pipeline in ("a", "b") for rule in ("binary", "combined")]
    draws = pd.DataFrame([{"pipeline": pipeline, "rule": rule, "m": 10, "budget": 0.05, "draw": draw,
                           "threshold_0": 0.5, "normal": 0.05, "composite": 0.6 + 0.1 * index,
                           "binary_positive": 0.5}
                          for index, (pipeline, rule, _, _) in enumerate(keys) for draw in range(2)])
    boot = {name: np.tile(draws.groupby(["pipeline", "rule"], sort=False)[name].mean().to_numpy(), (5, 1))
            for name in ("normal", "composite", "binary_positive")}
    summary, contrasts = summarize_screens(draws, boot, keys, (("b_minus_a", "b", "a"),), 0.05)
    assert len(summary) == 4
    assert summary[0]["per_1000_composite"]["caught"] == pytest.approx(30.0)
    labels = {(row["contrast"], row["rule"]) for row in contrasts}
    assert labels == {("b_minus_a", "binary"), ("b_minus_a", "combined"),
                      ("a_combined_minus_binary", "combined"), ("b_combined_minus_binary", "combined")}
    (row,) = [row for row in contrasts if row["contrast"] == "b_minus_a" and row["rule"] == "combined"
              and row["outcome"] == "composite"]
    assert row["difference"] == pytest.approx(0.2)
    assert row["ci"] == pytest.approx([0.2, 0.2])
