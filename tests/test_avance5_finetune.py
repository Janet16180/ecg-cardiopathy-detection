"""Verify algebraic transfer and split isolation without requiring a GPU."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecg_experiment.avance5_finetune import internal_split, unscaled_head


def test_converted_head_preserves_frozen_logits():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(120, 9)) * np.arange(1, 10) + 8
    labels = (features[:, 0] + features[:, 2] > 14).astype(int)
    fitted = make_pipeline(StandardScaler(), LogisticRegression(C=0.1)).fit(features, labels)
    weights, bias = unscaled_head(fitted)
    assert np.allclose(features @ weights + bias, fitted.decision_function(features), atol=1e-5)


def test_candidate_selection_cannot_share_known_training_patients():
    rows = pd.DataFrame({"source": ["ptbxl"] * 100,
                         "patient_id": np.repeat(np.arange(50).astype(str), 2),
                         "record_id": np.arange(100).astype(str), "target": np.tile([0, 1], 50)})
    split = internal_split(rows)
    sets = [set(rows.patient_id.iloc[index]) for index in split.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    assert len(np.unique(np.concatenate(list(split.values())))) == len(rows)
