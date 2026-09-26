"""Scientific controls for the CPC data-scaling continuation."""

import json

import numpy as np
import pytest

from ecg_experiment.cpc_scaling_cached import CachedCPCDataset
from ecg_experiment.files import sha256_file
from scripts.experiments.run_cpc_scaling_pilot_v2 import ordered_indices


def test_old_and_new_arms_have_matched_exposures() -> None:
    """The new arm sees every selected ECG; the old arm repeats within its pool."""
    old = ordered_indices("old", old_count=7, new_count=19)
    new = ordered_indices("new", old_count=7, new_count=19)
    assert len(old) == len(new) == 19
    assert set(old) <= set(range(7))
    assert new.tolist() == list(range(19))
    assert np.array_equal(old, ordered_indices("old", 7, 19))


def test_cached_loader_normalizes_without_changing_cache(tmp_path) -> None:
    """Cached signals use the fixed historical normalizer and reject tampering."""
    cache = tmp_path / "cache"
    cache.mkdir()
    signals = np.full((2, 12, 2500), 3, dtype=np.float32)
    np.save(cache / "signals.npy", signals)
    (cache / "complete.json").write_text(json.dumps({
        "signals_sha256": sha256_file(cache / "signals.npy"),
        "signals_shape": [2, 12, 2500],
    }))
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"mean": [1] * 12, "std": [2] * 12}))

    dataset = CachedCPCDataset(cache, normalization)
    assert len(dataset) == 2
    assert np.all(dataset[0][0].numpy() == 1)
    assert np.all(np.load(cache / "signals.npy") == 3)

    signals[0, 0, 0] = 9
    np.save(cache / "signals.npy", signals)
    with pytest.raises(ValueError, match="hash mismatch"):
        CachedCPCDataset(cache, normalization)
