"""Recording-edge fixed-P regression for Experiment 059 successor."""

import numpy as np

from ecg_experiment import hybrid_boundaries059_v2


def test_negative_recording_edge_p_preserved_and_joint_union_has_no_oracle(monkeypatch):
    peaks = np.array([10])
    intervals = np.array([[[[-1, -1], [5, 25], [30, 60]], [[-1, -1], [8, 28], [-1, -1]]]])
    monkeypatch.setattr(hybrid_boundaries059_v2, "adaptive_boundaries", lambda signal, fs: (peaks, intervals))
    _, predictions = hybrid_boundaries059_v2.joint_hybrid(np.zeros((2, 300)))
    assert predictions["fixed"][0, 0, 0] < 0
    assert np.array_equal(predictions["hybrid"][:, 0], predictions["fixed"][:, 0])
    assert predictions["hybrid"][0, 1].tolist() == [5, 28]
    assert predictions["hybrid"][0, 2].tolist() == [30, 60]
