"""Checks of the Experiment 025 label draws, C selection and summaries."""

import numpy as np
import pytest

from ecg_experiment.full_development import fit_logistic, predict
from ecg_experiment.label_efficiency import (
    C_GRID,
    draw_subset,
    fit_logistic_c,
    paired_summary,
    reading,
    select_c,
    smallest_budget,
    summarize,
)


def pool(patients: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(patients), 2).astype(str)
    y = (rng.random(len(ids)) < 0.6).astype(int)
    return ids, y


def test_draw_uses_distinct_patients_at_the_pool_prevalence():
    patients, y = pool()
    positions = draw_subset(patients, y, 100, seed=25025)
    assert len(positions) == 100
    assert len(set(patients[positions])) == 100
    assert y[positions].sum() == round(100 * y.mean())


def test_draw_is_reproducible_and_depends_on_the_seed():
    patients, y = pool()
    first = draw_subset(patients, y, 50, seed=1)
    assert np.array_equal(first, draw_subset(patients, y, 50, seed=1))
    assert not np.array_equal(first, draw_subset(patients, y, 50, seed=2))


def test_draw_fails_when_the_pool_has_too_few_patients():
    patients, y = pool(patients=40)
    with pytest.raises(ValueError, match="cannot supply"):
        draw_subset(patients, y, 60, seed=0)


def test_fixed_c_matches_the_experiment_020_head():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] + rng.normal(0, 1, 300) > 0).astype(int)
    assert np.array_equal(predict(fit_logistic_c(x, y, 0.01), x), predict(fit_logistic(x, y), x))


def test_select_c_returns_a_grid_value_and_breaks_ties_toward_smaller_c():
    x = np.arange(200.0)[:, None]
    y = (x[:, 0] >= 100).astype(int)
    chosen, means = select_c(x, y, np.arange(200), seed=0)
    assert chosen == C_GRID[0]
    assert means == [1.0] * len(C_GRID)


def test_select_c_keeps_patients_in_one_fold():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 3))
    y = (x[:, 0] > 0).astype(int)
    chosen, means = select_c(x, y, np.repeat(np.arange(60), 2), seed=5)
    assert chosen in C_GRID
    assert len(means) == len(C_GRID)


def test_summarize_reports_mean_sd_and_percentiles():
    result = summarize(np.array([1.0, 2.0, 3.0]))
    assert result["mean"] == 2.0
    assert result["sd"] == 1.0
    assert result["p2_5"] == pytest.approx(1.05)
    assert result["p97_5"] == pytest.approx(2.95)


def test_paired_reading_needs_ninety_percent_of_draws():
    first = np.arange(20.0)
    second = first.copy()
    second[:18] -= 1
    summary = paired_summary(first, second)
    assert summary["fraction_above_zero"] == 0.9
    assert reading("a", "b", summary) == "a better"
    second[17] += 2
    assert reading("a", "b", paired_summary(first, second)) == "no clear difference"
    assert reading("b", "a", paired_summary(first - 1, first)) == "a better"


def test_smallest_budget_reaching_a_target():
    means = {100: 0.8, 250: 0.9, 500: 0.88, 1000: 0.95}
    assert smallest_budget(means, 0.89) == 250
    assert smallest_budget(means, 0.99) is None
