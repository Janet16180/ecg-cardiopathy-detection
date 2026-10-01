import numpy as np
import pandas as pd
import pytest
import torch

from ecg_experiment.ann_heads import AttentionHead, Recipe, create_row_file, open_row_file, write_rows
from ecg_experiment.attention_findings import (
    finding_reading,
    located_batch,
    located_tokens,
    token_locator,
    top_unit,
    train_seeds,
    unlabeled_sph_table,
)
from ecg_experiment.lead_wave_maps import jepa_unit_map


def stores_of(tmp_path, arrays):
    stores = []
    for index, values in enumerate(arrays):
        store = create_row_file(tmp_path / f"c{index}.npy", values.shape, values.dtype)
        write_rows(store, 0, values)
        stores.append(open_row_file(tmp_path / f"c{index}.npy"))
    return stores


def test_located_tokens_read_each_row_from_its_cache(tmp_path) -> None:
    first = np.arange(4 * 3 * 2, dtype=np.float16).reshape(4, 3, 2)
    second = -np.arange(2 * 3 * 2, dtype=np.float16).reshape(2, 3, 2) - 1
    stores = stores_of(tmp_path, [first, second])
    parts = [(0, np.array([0, 2, 3]), np.array([3, 1, 0])), (1, np.array([4]), np.array([1]))]
    locator = token_locator(parts, 6)
    assert locator[1].tolist() == [-1, -1]
    found = located_tokens(stores, locator, np.array([4, 0, 3, 2]))
    assert np.array_equal(found, np.stack([second[1], first[3], first[0], first[1]]))
    batch = located_batch(stores, locator, "cpu")
    assert batch(np.array([2])).dtype == torch.float32
    with pytest.raises(ValueError, match="no tokens"):
        located_tokens(stores, locator, np.array([1]))
    with pytest.raises(ValueError, match="exactly one"):
        token_locator([(0, np.array([0]), np.array([0])), (1, np.array([0]), np.array([1]))], 2)


def test_train_seeds_is_reproducible_and_keeps_contributions() -> None:
    rng = np.random.default_rng(0)
    tokens = rng.normal(size=(120, 5, 8)).astype(np.float32)
    y = (tokens[:, 0, 0] > 0).astype(np.int64)
    tokens[y == 1, 2] += 1.0

    def batch(rows):
        return torch.from_numpy(tokens[np.asarray(rows)])

    fit = {"rows": np.arange(80), "y": y[:80], "weights": np.ones(80), "check_rows": np.arange(80, 100),
           "check_y": y[80:100]}
    recipe = Recipe(1e-2, 0.0, 16, 2, 2)
    runs = [train_seeds(lambda: AttentionHead(8), batch, fit, {"held": np.arange(100, 120)}, recipe, (1, 2),
                        "cpu", keep=("held",), log=lambda line: None) for _ in range(2)]
    assert np.array_equal(runs[0]["logits"]["held"], runs[1]["logits"]["held"])
    assert runs[0]["contributions"]["held"].shape == (20, 5)
    assert set(runs[0]["states"]) == {1, 2}
    mean = np.mean([runs[0]["seed_logits"][s]["held"] for s in (1, 2)], axis=0)
    assert np.allclose(runs[0]["logits"]["held"], mean)


def test_finding_reading_has_three_outcomes() -> None:
    assert finding_reading(0.0001) == "beats"
    assert finding_reading(0.0) == "matches"
    assert finding_reading(-0.0049) == "matches"
    assert finding_reading(-0.005) == "below"


def test_top_unit_names_lead_and_time() -> None:
    scores = np.zeros(400)
    scores[50 * 2 + 7] = 3.0
    found = top_unit(jepa_unit_map(scores))
    assert found["lead"] == "V1"
    assert found["start"] == pytest.approx(1.4)
    assert found["end"] == pytest.approx(1.6)


def test_unlabeled_sph_table_keeps_the_other_rows() -> None:
    sph = pd.DataFrame({"ecg_id": ["a", "b", "c"], "signal_sha256": ["x", "y", "z"]})
    table, positions = unlabeled_sph_table(sph, np.array([1]))
    assert positions.tolist() == [0, 2]
    assert table["key"].tolist() == ["sph:a", "sph:c"]
