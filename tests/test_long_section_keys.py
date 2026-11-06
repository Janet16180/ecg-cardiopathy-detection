"""Window support and boundary checks for larger ECG sections."""

import numpy as np
import pytest

from ecg_experiment.long_section_keys import windows


@pytest.mark.parametrize(("length", "count"), [(10, 31), (20, 21), (30, 11)])
def test_long_windows_cover_the_full_record_without_crossing_boundaries(length, count):
    spans = windows(length)
    assert spans.shape == (count, 2)
    np.testing.assert_array_equal(spans[:, 1] - spans[:, 0], length)
    assert spans[0, 0] == 0
    assert spans[-1, 1] == 40
    np.testing.assert_array_equal(np.diff(spans[:, 0]), 1)


def test_unregistered_window_length_is_rejected():
    with pytest.raises(ValueError, match="prespecified"):
        windows(40)
