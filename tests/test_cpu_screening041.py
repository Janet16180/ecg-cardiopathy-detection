"""Scientific isolation, calibration, integrity and serialization checks on artificial data."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment import cpu_screening041 as experiment


def selected_table():
    return pd.DataFrame(
        {
            "ecg_id": np.arange(1, 13),
            "patient_id": np.arange(101, 113),
            "strat_fold": [1] * 4 + [8] * 4 + [9] * 4,
            "target": [0, 1] * 6,
            "NORM": [1, 0] * 6,
            "MI": [0, 1] * 6,
            "STTC": [0] * 12,
            "CD": [0] * 12,
            "HYP": [0] * 12,
        }
    )


def test_role_masks_and_identity_isolation():
    table = selected_table()
    roles = experiment.role_masks(table)
    assert np.all(np.sum(list(roles.values()), axis=0) == 1)
    assert np.array_equal(table.loc[roles["fit"], "strat_fold"], [1] * 4)
    table.loc[4, "patient_id"] = table.loc[0, "patient_id"]
    with pytest.raises(ValueError, match="patients cross folds"):
        experiment.role_masks(table)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "unique"),
        ("test", "folds must be 1-9"),
        ("single_class", "both binary classes"),
        ("missing", "missing identity"),
    ],
)
def test_reject_bad_roles(mutation, match):
    table = selected_table()
    if mutation == "duplicate":
        table.loc[1, "ecg_id"] = table.loc[0, "ecg_id"]
    if mutation == "test":
        table.loc[0, "strat_fold"] = 10
    if mutation == "single_class":
        table.loc[table.strat_fold == 8, "target"] = 0
    if mutation == "missing":
        table.loc[0, "patient_id"] = np.nan
    with pytest.raises(ValueError, match=match):
        experiment.role_masks(table)


def test_labels_preserve_norm_abnormal_cooccurrence():
    table = selected_table()
    table.loc[1, "NORM"] = 1
    metadata = table.set_index("ecg_id")[["patient_id", "strat_fold"]].copy()
    metadata["scp_codes"] = [{"N": 100} if y == 0 else {"I": 100} for y in table.target]
    metadata.at[2, "scp_codes"] = {"N": 100, "I": 100, "SR": 100}
    experiment.validate_labels(table, metadata, {"N": "NORM", "I": "MI"})
    table.loc[1, "target"] = 0
    with pytest.raises(ValueError, match="existing superclass label"):
        experiment.validate_labels(table, metadata, {"N": "NORM", "I": "MI"})


def test_threshold_exact_order_ties_and_small_sample():
    scores = np.linspace(0, 1, 40)
    calibration = experiment.calibrate_threshold(np.zeros(40), scores)
    assert calibration["threshold"] == scores[-3]
    assert calibration["referred_normal_count"] == 2
    assert calibration["normal_referral_rate"] == 0.05
    tied = experiment.calibrate_threshold(np.zeros(40), np.ones(40))
    assert tied["referred_normal_count"] == 0
    small = experiment.calibrate_threshold(np.zeros(19), np.arange(19))
    assert small["threshold"] == 18
    assert small["referred_normal_count"] == 0


@pytest.mark.parametrize(
    ("delta", "low", "sensitivity", "normal", "advance"),
    [
        (0.01, 1e-12, -0.02, 0.08, True),
        (0.009, 0.001, 0, 0.05, False),
        (0.02, 0, 0, 0.05, False),
        (0.02, 0.001, -0.02001, 0.05, False),
        (0.02, 0.001, 0, 0.08001, False),
    ],
)
def test_every_decision_gate(delta, low, sensitivity, normal, advance):
    result = experiment.decision({"difference": delta, "ci_low": low}, sensitivity, normal)
    assert (result["outcome"] == "advance_development") == advance


def test_raw_exact_schema_excludes_forbidden_and_preserves_nans(tmp_path):
    table = selected_table()
    raw = pd.DataFrame(np.ones((len(table), 130)), columns=experiment.RAW_FEATURES)
    for name, value in reversed(
        [("ecg_id", table.ecg_id), ("status", "ok"), ("implausible", ""), ("flat_leads", "")]
    ):
        raw.insert(0, name, value)
    raw.loc[0, "hr_bpm"] = np.inf
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    features = experiment.raw_measurements(table, path)
    assert len(features.columns) == 130
    assert np.isnan(features.loc[0, "hr_bpm"])
    assert not {"age", "sex", "meta_age_group", "target", "flat_leads"} & set(features.columns)
    raw["age"] = 20
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="frozen 130 measurements"):
        experiment.raw_measurements(table, path)


def baseline_artifacts(tmp_path):
    table = selected_table()
    table["signal"] = [-1, 1, -2, 2] * 3
    fit = table.strat_fold.le(8)
    dev = table.strat_fold.eq(9)
    model = LogisticRegression(C=0.1, max_iter=5000).fit(table.loc[fit, ["signal"]], table.loc[fit, "target"])
    probability = model.predict_proba(table.loc[dev, ["signal"]])[:, 1]
    model_path, metrics_path = tmp_path / "baseline.joblib", tmp_path / "metrics.json"
    joblib.dump({"model": model, "features": ["signal"], "C": 0.1}, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "caracteristicas": ["signal"],
                "validacion": {
                    "AUROC": roc_auc_score(table.loc[dev, "target"], probability),
                    "AUPRC": average_precision_score(table.loc[dev, "target"], probability),
                },
            }
        )
    )
    return table, model_path, metrics_path


def test_baseline_predictions_and_historical_metrics_both_required(tmp_path):
    table, model_path, metrics_path = baseline_artifacts(tmp_path)
    assert experiment.baseline_integrity(table, model_path, metrics_path)["status"] == "passed"
    saved = joblib.load(model_path)
    saved["model"].intercept_ += 0.001
    joblib.dump(saved, model_path)
    with pytest.raises(ValueError, match="saved/refitted predictions"):
        experiment.baseline_integrity(table, model_path, metrics_path)
    table, model_path, metrics_path = baseline_artifacts(tmp_path)
    receipt = json.loads(metrics_path.read_text())
    receipt["validacion"]["AUROC"] -= 0.01
    metrics_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="historical tolerance"):
        experiment.baseline_integrity(table, model_path, metrics_path)


def test_strict_threshold_metrics_and_paired_sensitivity():
    table = selected_table().iloc[:4]
    probability = np.array([0.5, 0.5, 0.4, 0.9])
    summary = experiment.arm_summary(table, probability, 0.5, draws=30)
    assert summary["metrics"]["sensitivity"]["value"] == 0.5
    assert summary["metrics"]["specificity"]["value"] == 1
    assert summary["exploratory_superclass_misses"]["MI"]["missed_count"] == 1
    delta = experiment.sensitivity_difference(
        table.patient_id.to_numpy(), table.target.to_numpy(), probability > 0.5, probability >= 0.5, draws=30
    )
    assert delta["difference"] == -0.5


def test_fit_isolation_and_saved_predictions(tmp_path):
    table = selected_table()
    roles = experiment.role_masks(table)
    rng = np.random.default_rng(7)
    features = pd.DataFrame(rng.normal(size=(len(table), 130)), columns=experiment.RAW_FEATURES)
    features.iloc[0, 0] = np.nan
    features.iloc[:3, 1] = np.nan
    features.iloc[4:, 2] = 100000
    y = table.target.to_numpy()
    with threadpool_limits(limits=1):
        models, _ = experiment.fit_arms(features, y, roles["fit"])
        scaler = models["logistic"].named_steps["standardscaler"]
        assert scaler.mean_[2] == pytest.approx(features.iloc[:4, 2].mean())
        assert models["logistic"].named_steps["simpleimputer"].statistics_[1] == features.iloc[3, 1]
        path = tmp_path / "models.joblib"
        joblib.dump(models, path)
        reloaded = joblib.load(path)
        for name, model in models.items():
            expected = model.predict_proba(features)[:, 1]
            assert np.array_equal(expected, reloaded[name].predict_proba(features)[:, 1])


def test_empty_column_fails_clearly_and_logistic_retains_it():
    table = selected_table()
    roles = experiment.role_masks(table)
    features = pd.DataFrame(np.ones((12, 130)), columns=experiment.RAW_FEATURES)
    with threadpool_limits(limits=1):
        models, _ = experiment.fit_arms(features, table.target.to_numpy(), roles["fit"])
        features.iloc[:4, 0] = np.nan
        with pytest.raises(ValueError, match="entirely missing column"):
            experiment.fit_arms(features, table.target.to_numpy(), roles["fit"])
        logistic = models["logistic"].fit(features.loc[roles["fit"]], table.target.to_numpy()[roles["fit"]])
        imputer = logistic.named_steps["simpleimputer"]
        assert imputer.statistics_[0] == 0
        assert imputer.transform(features).shape[1] == 131


def test_integrity_failure_prevents_primary_fit_and_writes_failure(tmp_path, monkeypatch):
    table = selected_table()
    monkeypatch.setattr(experiment, "read_selected", lambda: table)
    monkeypatch.setattr(experiment, "load_metadata", lambda: None)
    monkeypatch.setattr(experiment, "load_statements", lambda: None)
    monkeypatch.setattr(experiment, "diagnostic_classes", lambda _: {})
    monkeypatch.setattr(experiment, "validate_labels", lambda *_: None)
    monkeypatch.setattr(experiment, "provenance", lambda _: {})

    def fail_integrity(_):
        raise ValueError("artificial integrity failure")

    def forbidden_fit(*_):
        pytest.fail("primary fit reached after failed integrity")

    monkeypatch.setattr(experiment, "baseline_integrity", fail_integrity)
    monkeypatch.setattr(experiment, "fit_arms", forbidden_fit)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="artificial integrity failure"):
        experiment.run_experiment(output, tmp_path / "protocol", threads=1)
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        experiment.run_experiment(output, tmp_path / "protocol", threads=1)
