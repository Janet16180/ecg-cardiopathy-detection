"""Verify padding never changes admitted row order or introduces extra scientific examples."""

import numpy as np
import pytest

from ecg_experiment.pr74_batch_shape_reproduction import padded_vectors


def test_partial_microbatch_uses_sixteen_inputs_and_discards_all_repeats(monkeypatch):
    from ecg_experiment import pr74_batch_shape_reproduction as batch

    inputs = np.repeat(np.arange(13, dtype=np.float32)[:, None, None], 1000 * 12, axis=1).reshape(
        13, 1000, 12
    )
    captured = []

    def encode(model, padded):
        captured.append(padded.copy())
        return np.repeat(padded[:, 0, 0, None], 1024, axis=1)

    monkeypatch.setattr(batch, "xecg_features", encode)
    vectors = padded_vectors(None, inputs)
    assert captured[0].shape == (16, 1000, 12)
    np.testing.assert_array_equal(captured[0][:13], inputs)
    np.testing.assert_array_equal(captured[0][13:], np.repeat(inputs[-1:], 3, axis=0))
    assert vectors.shape == (13, 1024)
    np.testing.assert_array_equal(vectors[:, 0], np.arange(13))


def test_full_or_empty_batches_are_not_silently_reshaped():
    for count in (0, 16, 17):
        with pytest.raises(ValueError, match="partial admitted PTB"):
            padded_vectors(None, np.zeros((count, 1000, 12), dtype=np.float32))
