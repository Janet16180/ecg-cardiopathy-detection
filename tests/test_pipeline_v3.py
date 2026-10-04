import numpy as np

from ecg_experiment.finding_screen import split_thresholds
from ecg_experiment.pipeline_v3 import local_normal_plans, pipeline_reading, screen_draws


def test_plans_follow_the_030_seed_convention() -> None:
    local_normal = np.arange(100, 160)
    plans = local_normal_plans(local_normal, (10, 20), 3, 30030)
    expected = local_normal[np.sort(np.random.default_rng([30030, 20, 2]).choice(60, 20, replace=False))]
    assert set(plans) == {10, 20}
    assert len(plans[10]) == 3
    assert np.array_equal(plans[20][2], expected)


def test_screen_draws_match_a_direct_computation() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(400, 2))
    evaluation = np.arange(400) % 2 == 0
    normal = np.flatnonzero(~evaluation)[:100]
    plans = local_normal_plans(normal, (50,), 4, 1)[50]
    positive = matrix[evaluation, 0] > 0.5
    found = screen_draws(matrix, plans, 100, (500,), evaluation, {"positive": positive},
                         {"all": np.ones(200, dtype=bool)})
    for draw, positions in enumerate(plans):
        thresholds = split_thresholds(matrix[positions], 100, (500,))
        referred = (matrix[evaluation] > thresholds).any(axis=1)
        assert np.array_equal(found["thresholds"][draw], thresholds)
        assert found["outcomes"]["positive"][draw] == referred[positive].mean()
        assert found["local"]["all"][draw] == (matrix[~evaluation] > thresholds).any(axis=1).mean()
    assert found["shares"].shape == (200,)


def test_pipeline_reading_has_three_outcomes() -> None:
    assert pipeline_reading([0.001, 0.01], 0.005) == "adopt_v3"
    assert pipeline_reading([0.0, 0.01], 0.005) == "no_worse_keep_v2"
    assert pipeline_reading([-0.004, 0.01], 0.005) == "no_worse_keep_v2"
    assert pipeline_reading([-0.005, 0.01], 0.005) == "keep_v2"
