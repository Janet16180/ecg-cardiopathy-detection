"""Checks of the cross-source condition vocabulary."""

import pandas as pd

from ecg_experiment.eda.cross import SNOMED_CODES, _any_listed


def test_challenge_equivalent_codes_count_as_the_same_condition():
    listed = pd.Series([["733534002"], ["164909002"], ["713427006"], ["426783006"]])
    flags = _any_listed(listed, SNOMED_CODES)
    assert flags["left bundle branch block"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert flags["right bundle branch block"].tolist() == [0.0, 0.0, 1.0, 0.0]
