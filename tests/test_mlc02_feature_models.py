from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ecg_experiment import mlc02_data as md
from ecg_experiment import mlc02_evaluation as me
from ecg_experiment import mlc02_feature_data as fd
from ecg_experiment import mlc02_feature_models as fm
from ecg_experiment.key_classifier_models import fit_model

NAN = np.nan
MISSING = Path("/nonexistent/mlc02")


# Role guard


@pytest.mark.parametrize("role", fd.CLOSED_ROLES)
def test_closed_roles_are_refused_before_any_file_is_read(role):
    with pytest.raises(PermissionError, match="closed"):
        fd.load_split((role,), ("xecg",), directory=MISSING, manifest_path=MISSING / "manifest.csv")
    with pytest.raises(PermissionError, match="closed"):
        fd.load_split(("train", role), ("xecg",), directory=MISSING, manifest_path=MISSING / "manifest.csv")


def test_only_train_and_validation_are_open():
    assert fd.check_open_roles(["train", "validation"]) == ("train", "validation")
    assert set(fd.OPEN_ROLES) | set(fd.CLOSED_ROLES) == set(md.ROLES)
    for roles in ([], ["sph"], ["training"]):
        with pytest.raises(ValueError, match="reads roles"):
            fd.check_open_roles(roles)


def _toy_tables(tmp_path):
    manifest = pd.DataFrame(
        {
            "record_key": ["a:1", "a:2", "b:1", "b:2"],
            "role": ["train", "validation", "test_external", "train"],
            "source": ["ptbxl", "ptbxl", "sph", "georgia"],
            "sha256_10s": ["h1", "h2", "h3", "h4"],
            "included": [True, True, True, False],
        }
    )
    rows = manifest.rename(columns={"included": "included_v4"}).assign(manifest_row=range(4))
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    return manifest, rows


def test_aligned_rows_checks_key_hash_and_inclusion(tmp_path):
    _, rows = _toy_tables(tmp_path)
    rows.to_csv(tmp_path / "rows.csv", index=False)
    assert (
        fd.aligned_rows(tmp_path / "rows.csv", tmp_path / "manifest.csv")["record_key"].tolist()[0] == "a:1"
    )
    for column, value in (("sha256_10s", "other"), ("record_key", "a:9"), ("included_v4", False)):
        changed = rows.copy()
        changed.loc[1, column] = value
        changed.to_csv(tmp_path / "rows.csv", index=False)
        with pytest.raises(ValueError, match="rows.csv"):
            fd.aligned_rows(tmp_path / "rows.csv", tmp_path / "manifest.csv")


def test_feature_positions_refuse_excluded_and_misassigned_records(tmp_path):
    _, rows = _toy_tables(tmp_path)
    rows = rows.drop(columns="manifest_row")
    assert fd.feature_positions(
        np.array(["a:2", "a:1"]), np.array(["validation", "train"]), rows
    ).tolist() == [1, 0]
    with pytest.raises(ValueError, match="no included"):
        fd.feature_positions(np.array(["b:2"]), np.array(["train"]), rows)
    with pytest.raises(ValueError, match="other roles"):
        fd.feature_positions(np.array(["a:1"]), np.array(["validation"]), rows)


def test_imputation_uses_training_medians_and_adds_flags():
    train = np.array([[1.0, NAN], [3.0, 10.0], [NAN, 20.0]])
    medians = fd.fit_imputation(train)
    np.testing.assert_array_equal(medians, [2.0, 15.0])
    imputed = fd.impute(np.array([[NAN, 1.0], [5.0, NAN]]), medians)
    np.testing.assert_array_equal(imputed, [[2.0, 1.0, 1.0, 0.0], [5.0, 15.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="no training value"):
        fd.fit_imputation(np.array([[1.0, NAN], [2.0, NAN]]))


# Masking


def _unit_rows(records, width, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(records, width))
    return x / np.linalg.norm(x, axis=1, keepdims=True), rng


def test_logistic_fit_uses_only_defined_records():
    z, rng = _unit_rows(300, 6, 0)
    y = (z[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(float)
    y[:60] = NAN
    fit = fm.fit_logistic(z, y, 1.0)
    reference = fit_model(z[60:], y[60:], 1.0)
    np.testing.assert_array_equal(fit["coef"], reference.coef_[0])
    assert fit["intercept"] == reference.intercept_[0]
    assert fit["records"] == 240
    changed = z.copy()
    changed[:60] = -changed[:60]
    np.testing.assert_array_equal(fm.fit_logistic(changed, y, 1.0)["coef"], fit["coef"])
    with pytest.raises(ValueError, match="positives and negatives"):
        fm.fit_logistic(z, np.where(np.isnan(y), NAN, 1.0), 1.0)


def test_boosting_fit_uses_only_defined_records():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 3))
    y = (x[:, 0] > 0).astype(float)
    y[:100] = NAN
    model = fm.fit_boosting(x, y, 0.1, 15, 5)
    reference = fm.fit_boosting(x[100:], y[100:], 0.1, 15, 5)
    np.testing.assert_array_equal(model.predict_proba(x)[:, 1], reference.predict_proba(x)[:, 1])


def test_masked_loss_ignores_undefined_entries():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.3, -1.2, 2.0], [1.5, 0.1, -0.4]], requires_grad=True)
    labels = torch.tensor([[1.0, NAN, 0.0], [NAN, 1.0, NAN]])
    loss = fm.masked_bce(logits, labels)
    defined = ~torch.isnan(labels)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[defined], labels[defined])
    assert float(loss) == pytest.approx(float(expected), abs=1e-7)
    loss.backward()
    assert torch.all(logits.grad[~defined] == 0)
    assert torch.all(logits.grad[defined] != 0)


def test_network_training_is_deterministic_and_keeps_the_best_epoch():
    pytest.importorskip("torch")
    z, rng = _unit_rows(600, 8, 2)
    y = (z[:, :1] @ np.ones((1, 14)) + 0.5 * rng.normal(size=(600, 14)) > 0).astype(float)
    y[rng.random(y.shape) < 0.2] = NAN
    sources = np.where(np.arange(200) < 100, "ptbxl", "ningbo")
    score = lambda p: me.selection_metric(y[:200], p, sources)["selection_metric"]  # noqa: E731
    runs = [
        fm.train_network(
            z[200:].astype(np.float32), y[200:], 1e-3, 1e-4, 7, 4, "cpu", z[:200].astype(np.float32), score
        )
        for _ in range(2)
    ]
    for key in runs[0]["weights"]:
        np.testing.assert_array_equal(runs[0]["weights"][key], runs[1]["weights"][key])
    best = max(runs[0]["history"], key=lambda row: row["validation_metric"])
    assert runs[0]["epoch"] == best["epoch"]
    np.testing.assert_allclose(
        fm.network_probability(z[:200], runs[0]["weights"]), runs[0]["probability"], atol=1e-5
    )


# Per-condition selection


def _two_source_labels(seed=3, records=400):
    rng = np.random.default_rng(seed)
    labels = (rng.random((records, 2)) < 0.4).astype(float)
    sources = np.where(np.arange(records) % 2 == 0, "ptbxl", "georgia")
    return labels, sources, rng


def test_per_condition_choice_takes_each_conditions_best_value():
    labels, sources, rng = _two_source_labels()
    good = expit(3 * (2 * labels - 1) + rng.normal(size=labels.shape))
    noise = rng.random(labels.shape)
    first = np.column_stack([good[:, 0], noise[:, 1]])
    second = np.column_stack([noise[:, 0], good[:, 1]])
    choice = fm.choose_per_condition(labels, sources, [first, second], ("a", "b"))
    assert choice["chosen"].tolist() == [0, 1]
    np.testing.assert_array_equal(choice["probability"], good)
    assert choice["ties"] == []
    for column in range(2):
        mean = np.mean(
            [
                me.roc_auc_score(labels[sources == s, column], good[sources == s, column])
                for s in ("ptbxl", "georgia")
            ]
        )
        assert choice["scores"][column, column] == pytest.approx(mean)


def test_per_condition_choice_breaks_ties_towards_the_first_value():
    labels, sources, rng = _two_source_labels(4)
    same = rng.random(labels.shape)
    choice = fm.choose_per_condition(labels, sources, [same, same.copy(), same], ("a", "b"))
    assert choice["chosen"].tolist() == [0, 0]
    assert choice["ties"] == ["a", "b"]


def test_per_condition_choice_ignores_undefined_entries():
    labels, sources, rng = _two_source_labels(5)
    good = expit(3 * (2 * labels - 1) + rng.normal(size=labels.shape))
    hidden = labels.copy()
    hidden[:100, 0] = NAN
    wrong = good.copy()
    wrong[:100, 0] = 1 - good[:100, 0]
    choice = fm.choose_per_condition(hidden, sources, [wrong, good], ("a", "b"))
    assert choice["scores"][0, 0] == choice["scores"][1, 0]


def test_staged_scores_equal_the_selection_metric_of_each_iteration():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(600, 4))
    y = (x[:, 0] + rng.normal(size=600) > 0).astype(float)
    y[rng.random(600) < 0.1] = NAN
    sources = np.where(np.arange(200) < 120, "ningbo", "cpsc_2018")
    model = fm.fit_boosting(x[200:], y[200:], 0.1, 15, 6)
    curve, staged = fm.staged_condition_scores(model, x[:200], y[:200], sources)
    assert staged.shape == (200, 6)
    for iteration in (0, 5):
        expected = me.selection_metric(y[:200, None], staged[:, [iteration]], sources, ("c",))[
            "per_condition"
        ]["c"]
        assert curve[iteration] == expected
    np.testing.assert_array_equal(staged[:, -1], model.predict_proba(x[:200])[:, 1])


# Artefacts and predict


def _toy_features(records=160, seed=7):
    rng = np.random.default_rng(seed)
    features = {
        name: rng.normal(size=(records, width)).astype(np.float32)
        for name, width in fd.FEATURE_WIDTHS.items()
    }
    measurements = features["measurements"].astype(np.float64)
    measurements[rng.random(measurements.shape) < 0.05] = NAN
    features["measurements"] = measurements
    labels = (rng.random((records, 14)) < 0.4).astype(float)
    labels[rng.random(labels.shape) < 0.1] = NAN
    return features, labels


def test_predict_reproduces_a_saved_logistic_candidate(tmp_path):
    features, labels = _toy_features()
    scaler = fm.fit_scaler(features["cpc"])
    z = fm.logistic_inputs(features["cpc"], scaler)
    fits = [fm.fit_logistic(z, labels[:, c], 1.0) for c in range(14)]
    arrays = {
        "coef": np.stack([f["coef"] for f in fits]),
        "intercept": np.array([f["intercept"] for f in fits]),
        "c": np.ones(14),
        "iterations": np.array([f["iterations"] for f in fits]),
    }
    fm.save_logistic(tmp_path / "L-C", "L-C", scaler, arrays, {"c": 1.0})
    probability = fm.predict("L-C", features, directory=tmp_path)
    np.testing.assert_array_equal(
        probability, fm.logistic_probability(z, arrays["coef"], arrays["intercept"])
    )
    with pytest.raises(ValueError, match="width"):
        fm.predict("L-C", {"cpc": features["cpc"][:, :10]}, directory=tmp_path)
    np.savez(tmp_path / "L-C" / "scaler.npz", mean=scaler["mean"] + 1, scale=scaler["scale"])
    with pytest.raises(ValueError, match="differ"):
        fm.predict("L-C", features, directory=tmp_path)


def test_predict_reproduces_a_saved_boosting_candidate(tmp_path):
    features, labels = _toy_features()
    models = [fm.fit_boosting(features["measurements"], labels[:, c], 0.1, 15, 3) for c in range(14)]
    fm.save_boosting(tmp_path / "G-M", models, {"toy": True})
    probability = fm.predict("G-M", {"measurements": features["measurements"]}, directory=tmp_path)
    expected = np.column_stack([m.predict_proba(features["measurements"])[:, 1] for m in models])
    np.testing.assert_array_equal(probability, expected)


def test_predict_reproduces_a_saved_network_candidate(tmp_path):
    features, _ = _toy_features()
    rng = np.random.default_rng(8)
    medians = fd.fit_imputation(features["measurements"])
    raw = fm.network_inputs("H-All", features, medians)
    assert raw.shape == (160, 1024 + 768 + 512 + 2 * 130)
    scaler = fm.fit_scaler(raw)
    weights = {
        seed: {
            "w1": rng.normal(size=(fm.HIDDEN, raw.shape[1])).astype(np.float32) * 0.02,
            "b1": rng.normal(size=fm.HIDDEN).astype(np.float32),
            "w2": rng.normal(size=(14, fm.HIDDEN)).astype(np.float32) * 0.05,
            "b2": rng.normal(size=14).astype(np.float32),
        }
        for seed in fm.SEEDS
    }
    fm.save_network(tmp_path / "H-All", "H-All", scaler, medians, weights, {"toy": True})
    z = fm.standardize(raw, scaler)
    each = {seed: fm.network_probability(z, w) for seed, w in weights.items()}
    np.testing.assert_array_equal(
        fm.predict("H-All", features, directory=tmp_path, seeds=(fm.SEEDS[1],)), each[fm.SEEDS[1]]
    )
    np.testing.assert_allclose(
        fm.predict("H-All", features, directory=tmp_path),
        np.mean(list(each.values()), axis=0),
        rtol=0,
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="seeds"):
        fm.predict("H-All", features, directory=tmp_path, seeds=(1,))


SAVED = [name for name in fm.INPUTS if (fm.CANDIDATES / name / "model.json").exists()]


@pytest.mark.skipif(
    not SAVED or not (fd.LOCAL / "rows.csv").exists(), reason="no fitted candidate on this machine"
)
def test_predict_reproduces_the_saved_validation_predictions():
    names = tuple(sorted({name for candidate in SAVED for name in fm.INPUTS[candidate]}))
    data, features = fd.load_split(("validation",), names)
    assert set(data["role"]) == {"validation"}
    for candidate in SAVED:
        model = fm.load_candidate(candidate)
        runs = [tuple(model["seeds"])] + (
            [(seed,) for seed in model["seeds"]] if len(model["seeds"]) > 1 else []
        )
        for seeds in runs:
            saved = md.load_predictions(
                md.prediction_path(fm.CANDIDATES, candidate, "validation", seeds), data["record_key"]
            )
            probability = fm.predict_loaded(model, features, seeds)
            tolerance = 1e-6 if model["kind"] == "network" else 0.0
            np.testing.assert_allclose(probability, saved["probability"], rtol=0, atol=tolerance)
