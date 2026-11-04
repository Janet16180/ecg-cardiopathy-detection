"""Strict parsing checks for the KEY-01 input correction."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.condition_key_inputs_v2 import memberships
from ecg_experiment.condition_keys import GROUPS


def test_decimal_membership_strings_preserve_binary_values():
    table = pd.DataFrame({name: ["0.0", "1.0", "0", "1"] for name in GROUPS})
    np.testing.assert_array_equal(memberships(table), np.repeat([[0], [1], [0], [1]], 4, axis=1))


@pytest.mark.parametrize("invalid", ["0.2", "-1", "2", "nan", "inf", "bad"])
def test_nonbinary_and_malformed_values_stop(invalid):
    table = pd.DataFrame({name: [invalid] for name in GROUPS})
    with pytest.raises(ValueError, match="binary|convert"):
        memberships(table)
