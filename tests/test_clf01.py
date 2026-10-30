"""Protect CLF-01 preprocessing, standalone scores and the fixed decision rule."""

import numpy as np
import pandas as pd

from ecg_experiment.clf01_study import CHALLENGERS, LEVEL, challenger, contrast, standalone
from ecg_experiment.pr74_screening_metrics import budget_threshold, point_metrics


def data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    x = rng.normal(size=(n, 6)) + y[:, None]
    x[::4, 2] = np.nan
    return x, y


def test_challengers_impute_and_scale_inside_the_pipeline():
    x, y = data()
    for name in CHALLENGERS:
        model = challenger(name).fit(x, y)
        assert model[0].statistics_.shape == (6,)
        scores = standalone(model, x)
        assert np.isfinite(scores).all()
        assert np.all((model.predict_proba(x)[:, 1] >= 0) & (model.predict_proba(x)[:, 1] <= 1))
    svm = challenger("svm").fit(x, y)
    np.testing.assert_array_equal(standalone(svm, x), svm.decision_function(x))


def test_bonferroni_level_and_decision_rule():
    assert abs(LEVEL - (1 - 0.05 / 3)) < 1e-15
    rng = np.random.default_rng(1)
    n = 600
    y = (rng.random(n) < 0.4).astype(int)
    frame = pd.DataFrame({"target": y, "patient_id": [f"p{i}" for i in range(n)],
                          "evaluation": np.arange(n) % 3 != 0})
    good, bad = y + rng.normal(0, 0.3, n), rng.normal(size=n)
    scores = {"good": good, "bad": bad}
    held = frame["evaluation"].to_numpy()
    points = {}
    for name, values in scores.items():
        threshold = budget_threshold(y[~held], values[~held])["threshold"]
        points[name] = {"threshold": threshold, "metrics": point_metrics(y[held], values[held], threshold)}
    better = contrast(points, scores, frame, "good", "bad")
    assert better["decision"] == "challenger_better"
    assert str(LEVEL) in better["sensitivity"]["intervals"]
    assert contrast(points, scores, frame, "bad", "good")["decision"] == "reference_better"
    assert contrast(points, scores, frame, "good", "good")["decision"] == "no_difference_established"
