"""Protect HEADS-01 mixture heads, paired intervals, gates, parity and the synthetic pipeline."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ecg_experiment.full_development import fit_logistic
from ecg_experiment.heads01_features import pooled_parity
from ecg_experiment.heads01_heads import (
    SUPERCLASSES,
    decision,
    fit_superclass,
    gates,
    mixture_score,
    paired_sensitivity,
)
from ecg_experiment.heads01_study import synthetic_run


def data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    flags = (rng.random((n, 4)) < 0.25).astype(int)
    y = flags.max(axis=1)
    x = rng.normal(size=(n, 6)) + flags @ rng.normal(size=(4, 6))
    return x, y, flags


def test_superclass_flags_must_reproduce_binary_target():
    x, y, flags = data()
    flags[np.flatnonzero(y == 1)[0]] = 0
    with pytest.raises(ValueError, match="reproduce"):
        fit_superclass(x, y, flags, "logistic")


def test_each_head_sees_negatives_and_its_own_positives_only():
    x, y, flags = data()
    fitted = fit_superclass(x, y, flags, "logistic")
    assert fitted["negatives"] == int((y == 0).sum())
    assert fitted["counts"] == {name: int(flags[:, i].sum()) for i, name in enumerate(SUPERCLASSES)}
    scores = mixture_score(fitted, x)
    assert scores.shape == (len(y),)
    assert np.all((scores > 0) & (scores < 1))


def test_identical_heads_reduce_to_the_binary_posterior():
    x, y, _ = data()
    head = fit_logistic(x, y)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    fitted = {"heads": dict.fromkeys(SUPERCLASSES, head), "counts": dict.fromkeys(SUPERCLASSES, n1),
              "negatives": n0, "positives": n1}
    np.testing.assert_allclose(mixture_score(fitted, x), expit(decision(head, x)), rtol=1e-12)


def test_mixture_is_monotone_in_each_head():
    x, y, flags = data()
    fitted = fit_superclass(x, y, flags, "logistic")
    base = mixture_score(fitted, x)
    scaler, model = fitted["heads"]["CD"]
    model.intercept_ = model.intercept_ + 1.0
    assert np.all(mixture_score(fitted, x) > base)


def test_paired_sensitivity_levels_and_identical_decisions():
    y = np.array([1, 1, 0, 1, 0, 1] * 20)
    patients = np.arange(len(y)) // 2
    first = np.array([1, 0, 0, 1, 1, 0] * 20, dtype=bool)
    result = paired_sensitivity(patients, y, first, first, draws=100, seed=3)
    assert result["difference"] == 0
    assert result["intervals"] == {"0.95": [0.0, 0.0], "0.975": [0.0, 0.0]}
    better = paired_sensitivity(patients, y, y.astype(bool), first, draws=100, seed=3)
    assert better["difference"] == pytest.approx(0.5)
    wide, narrow = better["intervals"]["0.975"], better["intervals"]["0.95"]
    assert wide[0] <= narrow[0] <= narrow[1] <= wide[1]


def test_gates_require_gain_positive_bound_and_specificity_guard():
    passing = {"difference": 0.025, "intervals": {"0.975": [0.001, 0.05]}}
    assert gates(passing, -0.005)["established"]
    assert not gates(passing, -0.0051)["established"]
    small = {"difference": 0.01, "intervals": {"0.975": [0.001, 0.02]}}
    result = gates(small, 0.0)
    assert not result["established"]
    assert result["statistically_supported"]
    assert not gates({"difference": 0.03, "intervals": {"0.975": [0.0, 0.06]}}, 0.0)["established"]


def test_pooled_parity_tolerance():
    a = np.ones((3, 4), dtype=np.float32)
    assert pooled_parity(a, a.copy())["bit_identical_rows"] == 3
    b = a.copy()
    b[1, 2] += 2e-6
    assert pooled_parity(b, a)["bit_identical_rows"] == 2
    b[0, 0] += 1e-4
    with pytest.raises(ValueError, match="differs"):
        pooled_parity(b, a)


def test_synthetic_pipeline_runs_end_to_end():
    result = synthetic_run()
    assert result["primary"] == ["token_pool_blend", "superclass_blend"]
    assert result["superclasses"] == list(SUPERCLASSES)


def test_hgb_superclass_heads_accept_missing_measurements():
    rng = np.random.default_rng(0)
    flags = (rng.random((300, 4)) < 0.3).astype(int)
    y = flags.max(axis=1)
    raw = pd.DataFrame(rng.normal(size=(300, 5)))
    raw.iloc[::5, 1] = np.nan
    fitted = fit_superclass(raw, y, flags, "hgb")
    assert np.isfinite(mixture_score(fitted, raw)).all()
