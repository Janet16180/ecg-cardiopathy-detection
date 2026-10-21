import numpy as np
import pandas as pd
from scipy.io import savemat

from ecg_experiment.challenge_pool import (
    assign_splits, centred_window, duplicate_groups, external_labels, read_header, read_signal,
)
from ecg_experiment.ptbxl_simdino_memory import LEAD_NAMES


def test_external_labels_map_equivalent_codes_and_sinus_rhythm_alone():
    assert external_labels({"426783006"})["NORM"]
    assert not external_labels({"426783006", "426177001"})["NORM"]
    labels = external_labels({"17338001", "59118001"})
    assert labels["PVC"] and labels["CRBBB"] and not labels["CLBBB"]


def test_duplicates_share_a_group_and_splits_keep_them_together():
    keys = [f"ningbo/JS{i:05d}" for i in range(200)] + ["chapman_shaoxing/JS00001"]
    hashes = [f"h{i}" for i in range(200)] + ["h5"]
    groups = duplicate_groups(hashes, keys)
    assert groups["chapman_shaoxing/JS00001"] == groups["ningbo/JS00005"]
    manifest = pd.DataFrame({"key": keys, "group": groups.loc[keys].to_numpy(),
                             "source": [key.split("/")[0] for key in keys]})
    splits = assign_splits(manifest, seed=53)
    assert splits["chapman_shaoxing/JS00001"] == splits["ningbo/JS00005"]
    shares = splits[[key.startswith("ningbo") for key in splits.index]].value_counts(normalize=True)
    assert abs(shares["train"] - 0.6) < 0.02 and abs(shares["prueba"] - 0.2) < 0.02


def test_centred_window_and_reading_a_synthetic_record(tmp_path):
    signal = np.arange(12 * 6000, dtype=np.int16).reshape(12, 6000) % 1000
    savemat(tmp_path / "A0001.mat", {"val": signal}, format="4")
    lines = ["A0001 12 500 6000"] + [f"A0001.mat 16 1000.0(0)/mV 16 0 0 0 0 {name}" for name in reversed(LEAD_NAMES)]
    (tmp_path / "A0001.hea").write_text("\n".join(lines + ["# Age: 50", "# Sex: Female", "# Dx: 426783006"]) + "\n")
    header = read_header(tmp_path / "A0001.hea")
    assert header["codes"] == {"426783006"} and header["rate"] == 500
    values = read_signal(tmp_path / "A0001.mat", header)
    assert np.allclose(values[0], signal[11] / 1000)
    window = centred_window(values, 500)
    assert window.shape == (12, 5000) and np.allclose(window[:, 0], values[:, 500])
    assert centred_window(values[:, :4000], 500) is None


def test_pairs_and_paired_bootstrap_favour_the_better_scores():
    from ecg_experiment.challenge_pool import evaluable_pairs, pair_aurocs, paired_pair_mean_bootstrap
    rng = np.random.default_rng(0)
    sources = np.array(["a"] * 200 + ["b"] * 200)
    labels = np.column_stack([rng.random(400) < 0.3, np.r_[rng.random(200) < 0.3, np.zeros(200, bool)]])
    pairs = evaluable_pairs(sources, labels, ("x", "y"), minimum=20)
    assert pairs == [("a", 0), ("a", 1), ("b", 0)]
    good = labels + rng.normal(scale=0.5, size=labels.shape)
    poor = labels + rng.normal(scale=2.0, size=labels.shape)
    assert pair_aurocs(sources, labels, good, pairs).mean() > pair_aurocs(sources, labels, poor, pairs).mean()
    result = paired_pair_mean_bootstrap(sources, labels, good, poor, pairs, draws=200, seed=1)
    assert result["ci_low"] > 0 and result["skipped_draws"] == 0


def test_youden_threshold_and_balanced_bootstrap():
    from ecg_experiment.challenge_pool import paired_balanced_bootstrap, threshold_table, youden_threshold
    rng = np.random.default_rng(2)
    labels = rng.random(400) < 0.4
    scores = labels + rng.normal(scale=0.7, size=400)
    threshold = youden_threshold(labels, scores)
    assert 0.0 < threshold < 1.0
    sources = np.array(["a"] * 400)
    table = threshold_table(sources, labels[:, None], scores[:, None], [("a", 0)], {("a", 0): threshold})
    assert table.balanced.iloc[0] > 0.7
    result = paired_balanced_bootstrap(sources, labels[:, None], scores[:, None], [("a", 0)], {("a", 0): threshold},
                                       {("a", 0): 3.0}, draws=100, seed=3)
    assert result["ci_low"] > 0
