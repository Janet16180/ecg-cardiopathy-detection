"""Tests of KEY-06 on synthetic data: role guards, clusters, residualisation, enrichment and probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from ecg_experiment import key06_embedding_structure as key06
from ecg_experiment.mlc02_data import CONDITIONS, VALIDATION_SOURCES


def blobs(sizes: tuple[int, ...], dimensions: int = 10, spread: float = 12.0, seed: int = 0
          ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, spread, size=(len(sizes), dimensions))
    points = np.concatenate([rng.normal(centre, 1.0, size=(size, dimensions))
                             for centre, size in zip(centres, sizes, strict=True)])
    return points, np.repeat(np.arange(len(sizes)), sizes)


def write_table(path: Path) -> None:
    pd.DataFrame({
        "record_key": ["a", "b", "c", "d", "e"],
        "role": ["train", "test_external", "validation", "sph_calibration", "train"],
        "included": ["True", "True", "True", "True", "False"],
        "age": ["30", "SECRET-NOT-A-NUMBER", "40", "SECRET", "50"],
    }).to_csv(path, index=False)


@pytest.mark.parametrize("role", key06.CLOSED_ROLES)
def test_closed_roles_are_refused(role: str) -> None:
    with pytest.raises(PermissionError):
        key06.check_open_roles(("train", role))


def test_unknown_role_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown"):
        key06.check_open_roles(("training",))


def test_read_by_role_never_parses_test_rows(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    write_table(path)
    frame = key06.read_by_role(path, ("train", "validation"), ["record_key", "age"], "included")
    assert frame["record_key"].tolist() == ["a", "c"]
    assert frame["position"].tolist() == [0, 2]
    assert not frame["age"].str.contains("SECRET").any()
    assert key06.to_float(frame["age"]).tolist() == [30.0, 40.0]


def test_read_by_role_refuses_test_roles_before_reading(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        key06.read_by_role(tmp_path / "absent.csv", ("test_same_source",), ["age"], "included")


def test_loaders_refuse_test_roles(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        key06.load_records(("validation", "test_external"), manifest=tmp_path / "absent.csv",
                           features=tmp_path)
    records = pd.DataFrame({"role": ["test_same_source"], "cache_block": ["test_same_source__ptbxl"],
                            "cache_row": [0], "sha256_10s": ["0"], "record_key": ["ptbxl:1"]})
    with pytest.raises(PermissionError):
        key06.read_windows(records)


def test_planted_clusters_are_recovered_and_stable() -> None:
    points, truth = blobs((300, 250, 200))
    for method in ("kmeans_3", "hdbscan"):
        fitted = key06.fit_clusters(points, method, seed=1)
        labels, _ = key06.order_by_size(fitted["labels"])
        assert adjusted_rand_score(truth, labels) == pytest.approx(1.0)
        assert labels.max() == 2
        indices = key06.subsample_indices(len(points), subsamples=5, share=0.8, seed=3)
        for rows in indices:
            again = key06.fit_clusters(points[rows], method, seed=2)["labels"]
            assert np.all(key06.jaccard_stability(labels, rows, again) == pytest.approx(1.0))


def test_jaccard_on_a_known_case() -> None:
    full = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1])
    indices = np.array([0, 1, 2, 4, 5, 6, 8])
    subsample = np.array([0, 0, 1, 1, 1, 1, 1])
    jaccard = key06.jaccard_stability(full, indices, subsample)
    assert jaccard[0] == pytest.approx(2 / 3)
    assert jaccard[1] == pytest.approx(3 / 5)


def test_order_by_size_keeps_noise() -> None:
    labels, order = key06.order_by_size(np.array([2, 2, 2, 0, -1, 1, 1]))
    assert labels.tolist() == [0, 0, 0, 2, -1, 1, 1]
    assert order.tolist() == [2, 1, 0]


def test_assignment_of_validation_records() -> None:
    points, truth = blobs((200, 200), seed=4)
    fitted = key06.fit_clusters(points, "kmeans_2", seed=1)
    labels, order = key06.order_by_size(fitted["labels"])
    centres = fitted["centres"][order]
    assigned = key06.assign_validation(points, labels, centres, points[:10] + 0.01)
    assert assigned.tolist() == labels[:10].tolist()
    noise = labels.copy()
    noise[:5] = -1
    assert key06.assign_validation(points, noise, None, points[:6]).tolist() == [-1] * 5 + [noise[5]]


def source_blocks(sources: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    half = len(sources) // 2
    design = key06.indicators(sources, VALIDATION_SOURCES)
    empty = np.zeros((len(sources), 0))
    blocks = dict.fromkeys(key06.FACTOR_GROUPS, (empty[:half], empty[half:]))
    blocks["source"] = (design[:half], design[half:])
    return blocks


def test_residualisation_removes_a_planted_source_effect() -> None:
    rng = np.random.default_rng(5)
    count = 4000
    sources = rng.choice(np.array(VALIDATION_SOURCES), size=count)
    shift = rng.normal(0.0, 6.0, size=(len(VALIDATION_SOURCES), 8))
    finding = rng.integers(0, 2, size=count)
    embedding = (shift[key06.stratum_codes(sources)] + 8.0 * finding[:, None] * np.eye(8)[0]
                 + rng.normal(0.0, 1.0, size=(count, 8)))
    blocks = source_blocks(sources)
    half = count // 2
    train, validation, _ = key06.residual_space(blocks, embedding[:half], embedding[half:])
    validation_sources = sources[half:]
    for source in VALIDATION_SOURCES:
        assert np.abs(validation[validation_sources == source, 1:].mean(axis=0)).max() < 0.25
    before = key06.ols_predict(blocks["source"][1], key06.ols_fit(blocks["source"][0], embedding[:half]))
    after = key06.ols_predict(blocks["source"][1], key06.ols_fit(blocks["source"][0], train))
    assert key06.variance_r2(embedding[half:], before) > 0.8
    assert key06.variance_r2(validation, after) < 0.01
    raw = key06.fit_clusters(embedding[:half], "kmeans_2", seed=1)["labels"]
    residual = key06.fit_clusters(train, "kmeans_2", seed=1)["labels"]
    assert adjusted_rand_score(finding[:half], residual) > 0.9
    assert adjusted_rand_score(finding[:half], raw) < 0.5


def test_variance_accounting_names_the_planted_group() -> None:
    rng = np.random.default_rng(6)
    count = 3000
    sources = rng.choice(np.array(VALIDATION_SOURCES), size=count)
    shift = rng.normal(0.0, 3.0, size=(len(VALIDATION_SOURCES), 12))
    embedding = shift[key06.stratum_codes(sources)] + rng.normal(0.0, 1.0, size=(count, 12))
    blocks = source_blocks(sources)
    blocks["labels"] = tuple(rng.integers(0, 2, size=(count, 3)).astype(float)[part]
                             for part in (slice(0, count // 2), slice(count // 2, None)))
    train, validation = embedding[: count // 2], embedding[count // 2:]
    table, components = key06.variance_accounting(blocks, train, validation, key06.pca_fit(train), 3)
    table = table.set_index("group")
    assert table.loc["source", "r2_alone_validation"] > 0.7
    assert table.loc["source", "r2_unique_validation"] > 0.7
    assert table.loc["labels", "r2_alone_validation"] < 0.01
    source_alone = table.loc["source", "r2_alone_validation"]
    assert table.loc["all", "r2_alone_validation"] == pytest.approx(source_alone, abs=0.01)
    assert len(components) == 3


def two_strata() -> tuple[np.ndarray, ...]:
    a = np.array([10.0, 15.0])
    members = np.array([30.0, 20.0])
    positives = np.array([40.0, 25.0])
    total = np.array([100.0, 50.0])
    return a, members, positives, total


def test_mantel_haenszel_on_a_known_case() -> None:
    result = key06.mantel_haenszel(*two_strata())
    assert result["odds_ratio"] == pytest.approx((4.0 + 6.0) / (6.0 + 1.0))
    variance = 30 * 70 * 40 * 60 / (100 ** 2 * 99) + 20 * 30 * 25 * 25 / (50 ** 2 * 49)
    assert result["statistic"] == pytest.approx(9.0 / variance)
    statsmodels = pytest.importorskip("statsmodels.stats.contingency_tables")
    tables = np.array([[[10, 20], [30, 40]], [[15, 5], [10, 20]]], dtype=float).transpose(1, 2, 0)
    reference = statsmodels.StratifiedTable(tables)
    assert result["odds_ratio"] == pytest.approx(reference.oddsratio_pooled)
    assert result["p_value"] == pytest.approx(reference.test_null_odds(correction=False).pvalue)


def test_mantel_haenszel_ignores_uninformative_strata_and_corrects_empty_cells() -> None:
    a, members, positives, total = two_strata()
    padded = key06.mantel_haenszel(np.append(a, 0.0), np.append(members, 10.0), np.append(positives, 0.0),
                                   np.append(total, 30.0))
    plain = key06.mantel_haenszel(a, members, positives, total)
    assert padded["odds_ratio"] == pytest.approx(plain["odds_ratio"])
    assert padded["p_value"] == pytest.approx(plain["p_value"])
    assert padded["strata"] == 2
    empty = key06.mantel_haenszel(np.array([0.0]), np.array([10.0]), np.array([5.0]), np.array([100.0]))
    assert empty["odds_ratio"] == pytest.approx((0.5 * 85.5 / 102) / (10.5 * 5.5 / 102))
    none = key06.mantel_haenszel(np.array([0.0]), np.array([10.0]), np.array([0.0]), np.array([100.0]))
    assert np.isnan(none["p_value"])
    assert np.isnan(none["odds_ratio"])


def test_benjamini_hochberg_on_a_known_case() -> None:
    p_values = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    adjusted = key06.benjamini_hochberg(p_values)
    assert adjusted[:4] == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert np.isnan(adjusted[4])
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(7)
    sample = rng.uniform(size=200) ** 3
    reference = multitest.multipletests(sample, method="fdr_bh")[1]
    assert key06.benjamini_hochberg(sample) == pytest.approx(reference)


def synthetic_strata(seed: int, count: int = 3000) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    strata = rng.integers(0, len(VALIDATION_SOURCES), size=count)
    membership = rng.integers(-1, 3, size=count)
    feature = (rng.uniform(size=count) < 0.1 + 0.2 * (membership == 1) + 0.05 * strata).astype(float)
    return strata, membership, feature


def test_enrichment_tables_and_bootstrap_agree_with_direct_counts() -> None:
    strata, membership, feature = synthetic_strata(8)
    result = key06.enrichment_tables(membership, feature[:, None], strata)
    member = membership == 1
    counts = [np.array([np.sum(member & (strata == s) & (feature == 1)) for s in range(6)]),
              np.array([np.sum(member & (strata == s)) for s in range(6)]),
              np.array([np.sum((strata == s) & (feature == 1)) for s in range(6)]),
              np.array([np.sum(strata == s) for s in range(6)])]
    direct = key06.mantel_haenszel(*counts)
    assert result["odds_ratio"][1, 0] == pytest.approx(direct["odds_ratio"])
    assert result["p_value"][1, 0] == pytest.approx(direct["p_value"])
    assert result["odds_ratio"][1, 0] > 1.5
    once = key06.bootstrap_log_odds(np.ones((1, len(member))), member, feature.astype(bool), strata)
    assert once[0] == pytest.approx(np.log(direct["odds_ratio"]))
    weights = key06.bootstrap_weights(np.arange(len(member)), draws=50, seed=1)
    assert weights.shape == (50, len(member))
    assert (weights.sum(axis=1) == len(member)).all()
    check = key06.replication(weights, member, feature.astype(bool), strata, np.ones(len(member), bool), 0.7)
    assert check["replicated"]


def test_enrichment_respects_undefined_entries() -> None:
    strata, membership, feature = synthetic_strata(9)
    defined = np.random.default_rng(1).uniform(size=len(feature)) < 0.7
    masked = key06.enrichment_tables(membership, feature[:, None], strata, defined[:, None])
    subset = key06.enrichment_tables(membership[defined], feature[defined][:, None], strata[defined])
    assert masked["odds_ratio"] == pytest.approx(subset["odds_ratio"])
    assert masked["members"] == pytest.approx(subset["members"])


def test_auroc_and_within_source_auroc() -> None:
    rng = np.random.default_rng(10)
    y = rng.uniform(size=600) < 0.3
    score = y + rng.normal(0.0, 1.0, size=600)
    score[::7] = np.round(score[::7])
    assert key06.auroc(y, score) == pytest.approx(roc_auc_score(y, score))
    strata = np.repeat(np.array(["a", "b"]), 300)
    shifted = score + 10.0 * (strata == "b") * y
    pooled, by_stratum = key06.within_source_auroc(y, shifted, strata)
    weights = [y[strata == s].sum() * (~y[strata == s]).sum() for s in ("a", "b")]
    expected = np.average([roc_auc_score(y[strata == s], shifted[strata == s]) for s in ("a", "b")],
                          weights=weights)
    assert pooled == pytest.approx(expected)
    assert by_stratum["b"] == pytest.approx(1.0)
    interval = key06.within_source_auroc_interval(y, shifted, strata, np.arange(600), draws=40, seed=2)
    assert interval["ci_low"] <= pooled <= interval["ci_high"]


def test_within_source_r2_ignores_source_means() -> None:
    strata = np.repeat(np.array(["a", "b"]), 50)
    y = np.where(strata == "a", 0.0, 100.0) + np.tile(np.linspace(-1, 1, 50), 2)
    assert key06.within_source_r2(y, np.where(strata == "a", 0.0, 100.0), strata) == pytest.approx(0.0)
    assert key06.within_source_r2(y, y, strata) == pytest.approx(1.0)


def test_ridge_predictions_match_sklearn_and_handle_missing() -> None:
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(11)
    x = rng.normal(size=(400, 6))
    targets = x @ rng.normal(size=(6, 3)) + rng.normal(size=(400, 3))
    targets[:50, 2] = np.nan
    predictions = key06.ridge_predictions(x, targets, x[:20], alpha=2.0)
    reference = Ridge(alpha=2.0).fit(x, targets[:, 0]).predict(x[:20])
    assert predictions[:, 0] == pytest.approx(reference)
    partial = Ridge(alpha=2.0).fit(x[50:], targets[50:, 2]).predict(x[:20])
    assert predictions[:, 2] == pytest.approx(partial)


def test_statements_exclude_the_fourteen_conditions() -> None:
    records = pd.DataFrame({
        "source": ["ptbxl", "ptbxl", "ningbo", "georgia"],
        "raw_codes": ["{'NORM': 100.0, 'SR': 0.0}", "{'STACH': 0.0, 'LVOLT': 0.0}", "426783006,427084000",
                      "164909002,251146004"],
        "heart_axis": ["MID", "", "", ""],
    })
    matrix, names = key06.statement_matrix(records)
    assert "scp:STACH" not in names
    assert "snomed:427084000" not in names
    expected = {"scp:NORM", "scp:SR", "scp:LVOLT", "scp:axis:MID", "snomed:426783006", "snomed:164909002"}
    assert expected <= set(names)
    assert "scp:axis:missing" not in names
    assert matrix.toarray().sum(axis=1).tolist() == [3, 1, 1, 2]
    assert key06.statement_mapping_note("snomed:164909002") == "space C positive: complete_lbbb"


def test_family_artefact_and_label_rules() -> None:
    assert key06.family_rule(pd.Series(["a"] * 90 + ["b"] * 10))[0]
    assert key06.family_rule(pd.Series(["a"] * 950 + ["b"] * 50))[0]
    assert not key06.family_rule(pd.Series(["a"] * 950 + ["b"] * 49))[0]
    flags = pd.DataFrame(False, index=range(1000), columns=list(key06.RELIABLE_FLAGS))
    member = np.zeros(1000, dtype=bool)
    member[:100] = True
    flags.loc[0, "edge_hold"] = True
    rates = key06.flag_rates(member, flags)
    assert not key06.artefact_rule(rates)[0]
    assert key06.artefact_rule(rates, key06.FLAG_MIN_MEMBERS)[0]
    flags.loc[100:199, "baseline_wander"] = True
    flags.loc[0:9, "baseline_wander"] = True
    assert key06.artefact_rule(key06.flag_rates(member, flags), key06.FLAG_MIN_MEMBERS)[0]
    flags.loc[0:29, "baseline_wander"] = True
    assert not key06.artefact_rule(key06.flag_rates(member, flags), key06.FLAG_MIN_MEMBERS)[0]
    labels = np.zeros((1000, len(CONDITIONS)))
    labels[:51, 3] = 1.0
    passes, name, share = key06.label_rule(member, labels)
    assert not passes
    assert name == CONDITIONS[3]
    assert share == pytest.approx(0.51)


def test_verdict_lists_failed_rules() -> None:
    row = pd.Series({"rule_stable": True, "rule_source": False, "rule_replicated": True,
                     "rule_artefact": False, "rule_labelled": True})
    assert key06.verdict(row) == (False, "one source; artefact")
    assert key06.verdict(row.replace(False, True)) == (True, "")


@pytest.mark.skipif(not key06.MANIFEST.exists() or not (key06.FEATURES / "rows.csv").exists(),
                    reason="local MLC-01 manifest and MLC-02 features are absent")
def test_local_records_align_with_feature_rows() -> None:
    records = key06.load_records()
    assert records["role"].value_counts().to_dict() == {"train": 56250, "validation": 15138}
    assert set(records["role"]) == {"train", "validation"}
    assert records["position"].is_monotonic_increasing
