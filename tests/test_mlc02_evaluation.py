import json
import math

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from ecg_experiment import mlc02_bootstrap as mb
from ecg_experiment import mlc02_data as md
from ecg_experiment import mlc02_evaluation as me
from ecg_experiment.intervals import patient_groups, patient_resample

NAN = np.nan


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _labels_and_scores(records=600, conditions=3, seed=0, undefined=0.1, strength=1.0):
    rng = np.random.default_rng(seed)
    labels = (rng.random((records, conditions)) < 0.3).astype(float)
    scores = _sigmoid(strength * (2 * labels - 1) + rng.normal(0, 1, labels.shape))
    labels[rng.random(labels.shape) < undefined] = NAN
    return labels, scores


def _clustered(patients=150, seed=0, effect=1.5):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 5, patients)
    patient_y = (rng.random(patients) < 0.3).astype(float)
    patient_u = rng.normal(0, 1, patients)
    groups = np.repeat(np.arange(patients), sizes)
    y = patient_y[groups]
    score = effect * y + patient_u[groups] + rng.normal(0, 1, len(groups))
    return groups.astype(str), y, _sigmoid(score)


# Point metrics


def test_condition_metrics_hand_case_with_undefined_entries():
    y = np.array([1, 0, NAN, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.5, 0.4, 0.6, 0.1])
    row = me.condition_metrics(y, p, threshold=0.5)
    assert (row["positives"], row["negatives"], row["undefined"]) == (2, 3, 1)
    assert row["prevalence"] == pytest.approx(0.4)
    assert row["auroc"] == pytest.approx(5 / 6)
    assert row["average_precision"] == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
    assert row["brier"] == pytest.approx((0.01 + 0.04 + 0.36 + 0.36 + 0.01) / 5)
    assert row["sensitivity"] == pytest.approx(0.5)
    assert row["specificity"] == pytest.approx(2 / 3)
    assert not row["enters_macro"]
    assert row["imprecise"]


def test_calibration_hand_case_saturated_two_point_design():
    # Logit 0 with 1 of 4 positive, logit 1 with 3 of 4 positive: a = -ln 3, a + b = ln 3.
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0, NAN, NAN])
    p = np.array([0.5] * 4 + [_sigmoid(1.0)] * 4 + [0.99, 0.01])
    row = me.condition_metrics(y, p)
    assert row["calibration_intercept"] == pytest.approx(-math.log(3), abs=1e-9)
    assert row["calibration_slope"] == pytest.approx(2 * math.log(3), abs=1e-9)


def test_calibration_in_the_large_hand_case():
    y = np.array([1, 0, 0, 0, NAN])
    p = np.full(5, 0.5)
    assert me.condition_metrics(y, p)["calibration_in_the_large"] == pytest.approx(-math.log(3), abs=1e-9)
    shifted = np.full(4, _sigmoid(0.4))
    assert me.offset_intercept(y[:4], me.logit(shifted))[0] == pytest.approx(-math.log(3) - 0.4, abs=1e-9)


def test_logistic_fit_matches_sklearn_and_weights_equal_repeated_rows():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 2, 400)
    y = (rng.random(400) < _sigmoid(-0.5 + 0.7 * x)).astype(float)
    weights = rng.poisson(1, (3, 400)).astype(float)
    a, b = me.logistic_fit(y, x, weights)
    for row in range(3):
        rows = np.repeat(np.arange(400), weights[row].astype(int))
        model = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000).fit(x[rows, None], y[rows])
        assert a[row] == pytest.approx(model.intercept_[0], abs=1e-6)
        assert b[row] == pytest.approx(model.coef_[0, 0], abs=1e-6)


def test_auroc_and_average_precision_equal_sklearn_on_defined_entries():
    labels, scores = _labels_and_scores(undefined=0.2)
    table = me.metrics_table(labels, scores, conditions=("a", "b", "c"))
    for column, row in enumerate(table["conditions"]):
        defined = ~np.isnan(labels[:, column])
        assert row["auroc"] == roc_auc_score(labels[defined, column], scores[defined, column])
        y, p = labels[defined, column], scores[defined, column]
        assert row["average_precision"] == average_precision_score(y, p)
    assert np.allclose(me.auroc_by_condition(labels, scores), [row["auroc"] for row in table["conditions"]])


def test_perfect_and_random_scorers():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.2).astype(float)
    perfect = me.condition_metrics(y, np.where(y == 1, 0.9, 0.1))
    assert perfect["auroc"] == 1.0
    assert perfect["average_precision"] == 1.0
    assert math.isnan(perfect["calibration_slope"])
    random = me.condition_metrics(y, rng.random(len(y)))
    assert abs(random["auroc"] - 0.5) < 0.02
    assert abs(random["average_precision"] - 0.2) < 0.02
    assert abs(random["calibration_slope"]) < 0.2


def test_macro_entry_rule_at_the_boundaries():
    def column(positives, negatives):
        return np.r_[np.ones(positives), np.zeros(negatives), np.full(100 - positives - negatives, NAN)]

    labels = np.column_stack([column(19, 50), column(20, 50), column(50, 19), column(20, 20), column(30, 70)])
    assert me.enters_macro(labels).tolist() == [False, True, False, True, True]
    rng = np.random.default_rng(2)
    scores = rng.random(labels.shape)
    names = tuple("abcde")
    table = me.metrics_table(labels, scores, conditions=names)
    assert table["entering"] == ["b", "d", "e"]
    expected = np.mean([row["auroc"] for row in table["conditions"] if row["condition"] in "bde"])
    assert table["macro_auroc"] == pytest.approx(expected)
    assert me.macro_auroc(labels, scores) == pytest.approx(expected)
    assert me.condition_metrics(np.r_[np.ones(99), np.zeros(99)], np.linspace(0, 1, 198))["imprecise"]
    assert not me.condition_metrics(np.r_[np.ones(100), np.zeros(99)], np.linspace(0, 1, 199))["imprecise"]


def test_macro_is_nan_without_entering_condition():
    labels = np.column_stack([np.r_[np.ones(5), np.zeros(50)]])
    assert math.isnan(me.macro_auroc(labels, np.linspace(0, 1, 55)[:, None]))


def test_scores_must_be_aligned_probabilities():
    labels = np.zeros((4, 2))
    with pytest.raises(ValueError, match="shape"):
        me.metrics_table(labels, np.zeros((4, 3)), conditions=("a", "b"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        me.metrics_table(labels, np.full((4, 2), 1.5), conditions=("a", "b"))
    with pytest.raises(ValueError, match="values other"):
        me.metrics_table(np.full((4, 2), 2.0), np.zeros((4, 2)), conditions=("a", "b"))


# Selection metric


def test_selection_metric_averages_sources_and_leaves_out_a_source_without_entering_condition():
    rng = np.random.default_rng(4)
    first, _ = _labels_and_scores(records=300, conditions=2, seed=5, undefined=0.0)
    second, _ = _labels_and_scores(records=300, conditions=2, seed=6, undefined=0.0)
    second[:, 1] = NAN
    third = np.zeros((40, 2))
    third[:5, 0] = 1
    labels = np.vstack([first, second, third])
    scores = rng.random(labels.shape)
    sources = np.array(["ptbxl"] * 300 + ["ningbo"] * 300 + ["georgia"] * 40)
    result = me.selection_metric(labels, scores, sources, conditions=("a", "b"))
    ptbxl = me.macro_auroc(first, scores[:300])
    ningbo = roc_auc_score(second[:, 0], scores[300:600, 0])
    assert result["sources"] == ["ptbxl", "ningbo", "georgia"]
    assert result["excluded_sources"] == ["georgia"]
    assert result["macro_auroc_by_source"] == pytest.approx({"ptbxl": ptbxl, "ningbo": ningbo})
    assert result["selection_metric"] == pytest.approx((ptbxl + ningbo) / 2)
    assert result["per_condition_sources"] == {"a": ["ptbxl", "ningbo"], "b": ["ptbxl"]}
    assert result["per_condition"]["a"] == pytest.approx(
        (roc_auc_score(first[:, 0], scores[:300, 0]) + ningbo) / 2)
    assert result["per_condition"]["b"] == pytest.approx(roc_auc_score(first[:, 1], scores[:300, 1]))


def test_selection_metric_refuses_test_sources():
    labels, scores = _labels_and_scores(records=100, conditions=2)
    with pytest.raises(ValueError, match="validation sources only"):
        me.selection_metric(labels, scores, np.array(["sph"] * 100), conditions=("a", "b"))


# Operating points


def test_threshold_reaches_ninety_percent_specificity_at_highest_sensitivity():
    labels, scores = _labels_and_scores(records=2000, conditions=3, seed=7, undefined=0.2)
    scores = np.round(scores, 2)
    thresholds = me.specificity_thresholds(labels, scores)
    for column, threshold in enumerate(thresholds):
        defined = ~np.isnan(labels[:, column])
        y, p = labels[defined, column], scores[defined, column]
        assert np.mean(p[y == 0] < threshold) >= 0.90
        lower = p[p < threshold]
        if len(lower):
            assert np.mean(p[y == 0] < lower.max()) < 0.90
        fpr, tpr, cuts = roc_curve(y, p, drop_intermediate=False)
        assert threshold == pytest.approx(cuts[fpr <= 0.10].min())


def test_threshold_counts_specificity_without_float_rounding():
    y = np.r_[np.zeros(10), np.ones(3)]
    p = np.r_[np.arange(10) / 10, [0.95, 0.85, 0.05]]
    assert me.specificity_threshold(y, p) == pytest.approx(0.85)
    assert me.specificity_threshold(np.zeros(3), np.array([0.1, 0.2, 0.3])) == math.inf


def test_platt_recovers_a_known_miscalibration():
    rng = np.random.default_rng(8)
    reported = rng.random((200000, 2)) * 0.98 + 0.01
    truth = np.column_stack([-0.7 + 1.6 * me.logit(reported[:, 0]), 0.4 + 0.5 * me.logit(reported[:, 1])])
    labels = (rng.random(reported.shape) < _sigmoid(truth)).astype(float)
    labels[:1000, 0] = NAN
    params = me.fit_platt(labels, reported)
    assert params[0] == pytest.approx([-0.7, 1.6], abs=0.03)
    assert params[1] == pytest.approx([0.4, 0.5], abs=0.03)
    fresh = rng.random((200000, 2)) * 0.98 + 0.01
    fresh_truth = np.column_stack([-0.7 + 1.6 * me.logit(fresh[:, 0]), 0.4 + 0.5 * me.logit(fresh[:, 1])])
    fresh_labels = (rng.random(fresh.shape) < _sigmoid(fresh_truth)).astype(float)
    table = me.metrics_table(fresh_labels, me.apply_platt(fresh, params), conditions=("a", "b"))
    for row in table["conditions"]:
        assert row["calibration_slope"] == pytest.approx(1.0, abs=0.03)
        assert row["calibration_intercept"] == pytest.approx(0.0, abs=0.03)
        assert row["calibration_in_the_large"] == pytest.approx(0.0, abs=0.03)


def test_apply_thresholds_is_greater_or_equal():
    predicted = me.apply_thresholds(np.array([[0.2, 0.5], [0.3, 0.49]]), np.array([0.3, 0.5]))
    assert predicted.tolist() == [[False, True], [True, False]]


# Derived multiclass view


def test_top_k_accuracy_under_both_readings():
    labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, NAN, 0], [1, 1, 0], [0, 0, 0]])
    probability = np.array([[0.9, 0.1, 0.2], [0.8, 0.7, 0.1], [0.3, 0.2, 0.1],
                            [0.1, 0.9, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    names = ("a", "b", "c")
    strict = me.top_k_accuracy(labels, probability, 1, True, names)
    assert strict["records"] == 3
    assert strict["accuracy"] == pytest.approx(1 / 3)
    assert strict["records_by_condition"] == {"a": 1, "b": 1, "c": 1}
    assert me.top_k_accuracy(labels, probability, 2, True, names)["accuracy"] == pytest.approx(2 / 3)
    loose = me.top_k_accuracy(labels, probability, 1, False, names)
    assert loose["records"] == 4
    assert loose["accuracy"] == pytest.approx(1 / 4)
    assert me.top_k_accuracy(labels, probability, 2, False, names)["accuracy"] == pytest.approx(2 / 4)


def test_exact_set_accuracy_under_both_readings():
    labels = np.array([[1, 0], [0, 0], [1, NAN], [0, 1]])
    probability = np.array([[0.9, 0.1], [0.1, 0.1], [0.9, 0.9], [0.9, 0.9]])
    result = me.exact_set_accuracy(labels, probability, np.array([0.5, 0.5]))
    assert result["records"] == 4
    assert result["accuracy"] == pytest.approx(3 / 4)
    assert result["records_all_defined"] == 3
    assert result["accuracy_all_defined"] == pytest.approx(2 / 3)


# Audits


def test_cooccurrence_split_keeps_negatives_and_splits_positives():
    labels = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, NAN]])
    split = me.cooccurrence_labels(labels)
    assert np.array_equal(split["one_condition"], np.array([[1, 0], [NAN, NAN], [0, 1], [0, 0], [1, NAN]]),
                          equal_nan=True)
    assert np.array_equal(split["several_conditions"],
                          np.array([[NAN, 0], [1, 1], [0, NAN], [0, 0], [NAN, NAN]]), equal_nan=True)


def test_subgroups_apply_the_macro_rule_within_the_subgroup():
    labels, scores = _labels_and_scores(records=400, conditions=2, undefined=0.0)
    sex = np.array(["F", "M"] * 190 + ["unknown"] * 20)
    masks = me.sex_masks(sex)
    assert [int(mask.sum()) for mask in masks.values()] == [190, 190, 20]
    tables = me.subgroup_tables(labels, scores, masks, conditions=("a", "b"))
    assert tables["female"]["records"] == 190
    female = sex == "F"
    assert tables["female"]["macro_auroc"] == pytest.approx(me.macro_auroc(labels[female], scores[female]))
    assert tables["unknown"]["entering"] == []
    lead = me.lead_masks(np.r_[np.ones(10, bool), np.zeros(390, bool)])
    assert "flat_or_missing_lead" in me.subgroup_tables(labels, scores, lead, conditions=("a", "b"))


def test_source_identification_separates_shifted_families_and_not_noise():
    rng = np.random.default_rng(9)
    families = np.repeat(np.array(["ptbxl", "chapman_ningbo", "georgia", "cpsc"]), 150)
    groups = np.array([f"r{index}" for index in range(len(families))])
    shift = np.searchsorted(np.unique(families), families)
    shifted = rng.normal(0, 1, (len(families), 8)) + 3 * np.eye(8)[shift]
    noise = rng.normal(0, 1, (len(families), 8))
    separable = me.source_identification(shifted, families, groups)
    chance = me.source_identification(noise, families, groups)
    assert separable["balanced_accuracy"] > 0.9
    assert separable["macro_ovr_auroc"] > 0.95
    assert chance["balanced_accuracy"] < 0.4
    assert chance["chance_balanced_accuracy"] == 0.25
    assert np.sum(chance["confusion"]) == len(families)
    with pytest.raises(ValueError, match="finite"):
        me.source_identification(np.full((600, 2), NAN), families, groups)


# Selection record


def test_tie_rule_goes_to_the_simpler_model_within_the_margin():
    assert me.select_candidate({"P0": 0.9000, "H-X": 0.9008})["selected"] == "P0"
    assert me.select_candidate({"P0": 0.9000, "H-X": 0.9011})["selected"] == "H-X"
    assert me.select_candidate({"P0": 0.9000, "H-X": 0.9010})["selected"] == "P0"
    chain = me.select_candidate({"P0": 0.9000, "L-J": 0.9009, "S-R": 0.9018})
    assert (chain["selected"], chain["highest"], chain["tied"]) == ("L-J", "S-R", ["L-J", "S-R"])
    assert me.select_candidate({"E": 0.95, "F": 0.95})["selected"] == "F"
    with pytest.raises(ValueError, match="protocol table"):
        me.select_candidate({"P1": 0.9})


def test_selection_record_writer(tmp_path):
    labels, _ = _labels_and_scores(records=400, conditions=2, undefined=0.0)
    rng = np.random.default_rng(10)
    sources = np.array(["ptbxl", "ningbo"] * 200)
    good = _sigmoid(3 * labels + rng.normal(0, 1, labels.shape))
    tables = {"P0": me.selection_metric(labels, rng.random(labels.shape), sources, ("a", "b")),
              "H-All": me.selection_metric(labels, good, sources, ("a", "b"))}
    record = me.selection_record(tables, {"P0": {"C": "per condition"}, "H-All": {"lr": 0.001}})
    assert record["selected"] == "H-All"
    assert [row["id"] for row in record["candidates"]] == ["P0", "H-All"]
    paths = me.write_selection_record(tmp_path, record)
    assert json.loads(paths["json"].read_text())["selected"] == "H-All"
    text = paths["markdown"].read_text()
    assert "Selected: H-All." in text
    assert "| ptbxl | ningbo |" in text
    other = labels.copy()
    other[:, 1] = NAN
    tables["L-J"] = me.selection_metric(other, good, sources, ("a", "b"))
    with pytest.raises(ValueError, match="other validation labels"):
        me.selection_record(tables)


# Bootstrap


def test_resamples_are_deterministic_and_follow_the_intervals_convention():
    labels, _ = _labels_and_scores(records=300, conditions=2, undefined=0.0)
    groups = np.repeat(np.arange(100), 3).astype(str)
    first = mb.cluster_resamples(groups, labels, draws=50, seed=11)
    second = mb.cluster_resamples(groups, labels, draws=50, seed=11)
    other = mb.cluster_resamples(groups, labels, draws=50, seed=12)
    assert np.array_equal(first["weights"], second["weights"])
    assert first["fingerprint"] == second["fingerprint"] != other["fingerprint"]
    assert first["redraws"] == 0
    rng = np.random.default_rng(11)
    members = patient_groups(groups)
    for row in range(50):
        expected = np.bincount(patient_resample(members, rng), minlength=300)
        assert np.array_equal(first["weights"][row], expected)


def test_resamples_move_whole_patients():
    groups, y, _ = _clustered(seed=12)
    labels = y[:, None]
    resamples = mb.cluster_resamples(groups, labels, draws=200, seed=13)
    weights = resamples["weights"]
    codes = np.unique(groups, return_inverse=True)[1]
    first_record = np.array([np.flatnonzero(codes == code)[0] for code in range(codes.max() + 1)])
    assert np.array_equal(weights, weights[:, first_record[codes]])
    assert (weights[:, first_record].sum(axis=1) == resamples["groups"]).all()


def test_redraw_rule_keeps_both_classes_of_every_entering_condition():
    entering = np.r_[np.ones(20), np.zeros(380)]
    rare = np.r_[np.zeros(20), np.ones(3), np.zeros(377)]
    labels = np.column_stack([entering, rare])
    groups = np.r_[np.zeros(20, int), np.arange(1, 381)].astype(str)
    resamples = mb.cluster_resamples(groups, labels, draws=300, seed=14)
    assert resamples["entering"].tolist() == [True, False]
    assert resamples["redraws"] > 50
    weights = resamples["weights"].astype(float)
    assert ((weights @ (labels[:, 0] == 1)) > 0).all()
    assert ((weights @ (labels[:, 0] == 0)) > 0).all()
    assert ((weights @ (labels[:, 1] == 1)) == 0).any()


def test_bootstrap_metrics_equal_sklearn_on_concatenated_resamples():
    labels, scores = _labels_and_scores(records=400, conditions=2, undefined=0.15)
    scores = np.round(scores, 2)
    groups = np.repeat(np.arange(200), 2).astype(str)
    resamples = mb.cluster_resamples(groups, labels, draws=5, seed=15)
    thresholds = np.array([0.5, 0.6])
    boot = mb.bootstrap_metrics(labels, scores, resamples, thresholds)
    for row, weights in enumerate(resamples["weights"]):
        rows = np.repeat(np.arange(len(labels)), weights)
        for column in range(2):
            y, p = labels[rows, column], scores[rows, column]
            defined = ~np.isnan(y)
            expected = me.condition_metrics(y[defined], p[defined], thresholds[column])
            for name in mb.METRICS:
                assert boot["samples"][name][row, column] == pytest.approx(expected[name], abs=1e-9), name
        assert boot["macro_auroc"][row] == pytest.approx(me.macro_auroc(labels[rows], scores[rows]))


def test_bootstrap_rejects_resamples_of_another_set():
    labels, scores = _labels_and_scores(records=300, conditions=2, undefined=0.0)
    resamples = mb.cluster_resamples(np.arange(300).astype(str), labels, draws=10)
    with pytest.raises(ValueError, match="another set"):
        mb.bootstrap_metrics(labels[:200], scores[:200], resamples)


def test_interval_coverage_of_a_known_auroc():
    effect = 1.5
    truth = norm.cdf(effect / math.sqrt(2 * 2))
    covered = []
    for dataset in range(200):
        groups, y, score = _clustered(patients=150, seed=1000 + dataset, effect=effect)
        labels = y[:, None]
        resamples = mb.cluster_resamples(groups, labels, draws=2000, seed=dataset)
        report, _ = mb.interval_report(labels, score[:, None], resamples, calibration=False,
                                       conditions=("a",))
        row = report["conditions"][0]
        covered.append(row["auroc_ci_low"] <= truth <= row["auroc_ci_high"])
    assert 0.90 <= np.mean(covered) <= 0.98


def test_interval_report_fields_and_usable_rule():
    labels, scores = _labels_and_scores(records=1500, conditions=2, undefined=0.0, strength=2.0)
    groups = np.arange(1500).astype(str)
    resamples = mb.cluster_resamples(groups, labels, draws=200, seed=16)
    report, boot = mb.interval_report(labels, scores, resamples, np.array([0.5, 0.5]), conditions=("a", "b"))
    for row in report["conditions"]:
        assert row["auroc_ci_low"] < row["auroc"] < row["auroc_ci_high"]
        assert row["calibration_slope_ci_low"] < row["calibration_slope"] < row["calibration_slope_ci_high"]
        assert row["usable"] == (row["auroc_ci_low"] >= 0.80 and row["positives"] >= 100)
    assert report["macro_auroc_ci_low"] < report["macro_auroc"] < report["macro_auroc_ci_high"]
    assert report["fingerprint"] == boot["fingerprint"] == resamples["fingerprint"]


def test_paired_difference_detects_a_better_model_and_not_a_copy():
    rng = np.random.default_rng(17)
    labels = (rng.random((3000, 3)) < 0.2).astype(float)
    latent = 1.0 * labels + rng.normal(0, 1, labels.shape)
    strong = _sigmoid(2.5 * labels + rng.normal(0, 1, labels.shape))
    weak = _sigmoid(latent)
    twin_a = _sigmoid(latent + rng.normal(0, 0.3, labels.shape))
    twin_b = _sigmoid(latent + rng.normal(0, 0.3, labels.shape))
    groups = np.repeat(np.arange(1500), 2).astype(str)
    resamples = mb.cluster_resamples(groups, labels, draws=1000, seed=18)
    boots = mb.bootstrap_models(labels, {"strong": strong, "weak": weak, "a": twin_a, "b": twin_b}, resamples,
                                calibration=False, workers=2)
    better = mb.paired_comparison(boots["strong"], boots["weak"], ("x", "y", "z"))
    assert better["macro"]["ci_low"] > 0
    assert better["macro"]["better"]
    assert all(row["p_holm"] < 0.01 for row in better["conditions"])
    same = mb.paired_comparison(boots["a"], boots["b"], ("x", "y", "z"))
    assert same["macro"]["ci_low"] < 0 < same["macro"]["ci_high"]
    assert not same["macro"]["better"]
    assert same["holm_family"] == 3
    other = mb.cluster_resamples(groups, labels, draws=1000, seed=19)
    with pytest.raises(ValueError, match="same resamples"):
        mb.paired_comparison(boots["a"], mb.bootstrap_metrics(labels, twin_b, other, calibration=False))


def test_holm_adjustment_hand_example():
    adjusted = mb.holm_adjust(np.array([0.01, 0.04, 0.03, 0.005, NAN]))
    assert np.allclose(adjusted[:4], [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(adjusted[4])
    assert np.allclose(mb.holm_adjust(np.array([0.5, 0.9])), [1.0, 1.0])
    assert np.allclose(mb.holm_adjust(np.array([0.9, 0.02])), [0.9, 0.04])


def test_bootstrap_p_value_hand_cases():
    assert mb.bootstrap_p_value(np.full(99, 0.1)) == pytest.approx(2 / 100)
    assert mb.bootstrap_p_value(np.zeros(99)) == 1.0
    assert mb.bootstrap_p_value(np.r_[np.full(9, -1.0), np.full(90, 1.0)]) == pytest.approx(0.2)


def test_percentile_interval_ignores_nan_draws():
    values = np.column_stack([np.r_[np.arange(100.0), np.full(10, NAN)], np.full(110, NAN)])
    low, high, missing = mb.percentile_interval(values)
    assert low[0] == pytest.approx(np.percentile(np.arange(100.0), 2.5))
    assert high[0] == pytest.approx(np.percentile(np.arange(100.0), 97.5))
    assert missing.tolist() == [10, 110]
    assert np.isnan(low[1])


# Prediction files


def test_prediction_file_round_trip_and_alignment(tmp_path):
    keys = np.array(["sph:A1", "sph:A2", "sph:A3"])
    probability = np.array([[0.1] * 14, [0.5] * 14, [0.9] * 14])
    path = md.prediction_path(tmp_path, "H-All", "test_external", (20261110, 20261111, 20261112))
    assert path.name == "test_external__seeds-20261110-20261111-20261112.npz"
    md.save_predictions(path, keys, probability, "H-All", (20261110, 20261111, 20261112))
    loaded = md.load_predictions(path, keys)
    assert np.array_equal(loaded["probability"], probability)
    assert (loaded["model_id"], loaded["seeds"]) == ("H-All", (20261110, 20261111, 20261112))
    with pytest.raises(ValueError, match="not aligned"):
        md.load_predictions(path, keys[[1, 0, 2]])
    with pytest.raises(ValueError, match="not aligned"):
        md.load_predictions(path, keys[:2])
    with pytest.raises(ValueError, match="conditions differ"):
        md.load_predictions(path, keys, md.GROUPS_A)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        md.save_predictions(path, keys, probability + 0.5, "H-All", (1,))
    with pytest.raises(ValueError, match="repeat"):
        md.save_predictions(path, keys[[0, 0, 1]], probability, "H-All", (1,))
    with pytest.raises(ValueError, match="model id"):
        md.prediction_path(tmp_path, "../P0", "test_external", (1,))


# Real manifest (labels only)


needs_manifest = pytest.mark.skipif(not md.MANIFEST.exists(), reason="the MLC-01 v4 manifest is local only")


@pytest.fixture(scope="module")
def manifest():
    return md.load_manifest()


@needs_manifest
def test_manifest_matches_the_mlc01_results_report(manifest):
    roles = {role: int((manifest["role"] == role).sum()) for role in md.ROLES}
    assert roles == {"train": 56250, "validation": 15138, "test_same_source": 2198, "test_external": 12698,
                     "sph_calibration": 12883}
    assert manifest["labels"].shape == (99167, 14)
    assert manifest["labels_a"].shape == (99167, 4)
    sph = md.role_data(manifest, "test_external")
    counts = me.support(sph["labels"])
    column = md.CONDITIONS.index("t_wave_abnormality")
    assert (counts["positives"][column], counts["negatives"][column], counts["undefined"][column]) == (
        1633, 11023, 42)
    assert len(np.unique(sph["group"])) == 12246
    fold10 = md.role_data(manifest, "test_same_source")
    column = md.CONDITIONS.index("left_axis_deviation")
    assert tuple(value[column] for value in me.support(fold10["labels"]).values()) == (134, 704, 1360)
    lik = me.support(fold10["labels_lik50"])["positives"]
    assert lik[md.CONDITIONS.index("mi_any")] == 391
    assert lik[md.CONDITIONS.index("lvh")] == 177
    assert me.enters_macro(fold10["labels"]).all()
    assert me.enters_macro(sph["labels"]).all()
    validation = md.role_data(manifest, "validation")
    entering = {source: int(me.enters_macro(validation["labels"][validation["source"] == source]).sum())
                for source in md.VALIDATION_SOURCES}
    assert entering == {"ptbxl": 14, "ningbo": 14, "chapman_shaoxing": 9, "georgia": 12, "cpsc_2018": 5,
                        "cpsc_2018_extra": 7}


@needs_manifest
def test_random_scores_on_the_real_labels_give_chance_macro_auroc(manifest):
    rng = np.random.default_rng(20261110)
    scores = rng.random(manifest["labels"].shape)
    sets = {f"validation:{source}": (manifest["role"] == "validation") & (manifest["source"] == source)
            for source in md.VALIDATION_SOURCES}
    sets |= {role: manifest["role"] == role
             for role in ("test_same_source", "test_external", "sph_calibration")}
    for name, mask in sets.items():
        labels = manifest["labels"][mask]
        counts = me.support(labels)
        entering = me.enters_macro(labels)
        positives, negatives = counts["positives"][entering], counts["negatives"][entering]
        variance = np.sum((positives + negatives + 1) / (12 * positives * negatives))
        null_sd = math.sqrt(variance) / entering.sum()
        assert abs(me.macro_auroc(labels, scores[mask]) - 0.5) < 4 * null_sd, name
