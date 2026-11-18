"""Tests of MLC-02 Stage 3: E's family rule, ties, operating points, calibration fits and the test guards."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ecg_experiment import ROOT
from ecg_experiment import mlc02_stage3 as s3
from ecg_experiment.files import write_json_atomic
from ecg_experiment.mlc02_calibration import calibration_report, damped_logistic_fit, fit_platt_damped
from ecg_experiment.mlc02_data import (
    CONDITIONS,
    MANIFEST,
    VALIDATION_SOURCES,
    prediction_path,
    save_predictions,
)
from ecg_experiment.mlc02_evaluation import (
    logistic_fit,
    logit,
    select_candidate,
    selection_metric,
    specificity_threshold,
)

SEEDS3 = [20261110, 20261111, 20261112]
STRENGTH = {
    "P0": 1.2,
    "L-J": 1.0,
    "L-C": 0.6,
    "G-M": 1.1,
    "H-X": 1.3,
    "H-J": 1.25,
    "H-C": 0.8,
    "H-All": 1.6,
    "S-R": 1.4,
}
CLOSED_WORDS = ("test_same_source", "test_external", "sph_calibration")
OPENED: list[str] | None = None


def _audit(event: str, args: tuple) -> None:
    if OPENED is not None and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENED.append(os.fsdecode(args[0]))


sys.addaudithook(_audit)


def _validation(per_source: int = 160, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sources = np.repeat(np.array(VALIDATION_SOURCES), per_source)
    labels = (rng.random((len(sources), 14)) < 0.35).astype(float)
    labels[rng.random(labels.shape) < 0.05] = np.nan
    keys = np.array([f"{source}:{i}" for i, source in enumerate(sources)])
    return {
        "record_key": keys,
        "source": sources,
        "family": sources,
        "role": np.full(len(keys), "validation"),
        "group": keys,
        "sex": rng.choice(["F", "M", "unknown"], len(keys)),
        "lead_flag": rng.random(len(keys)) < 0.1,
        "labels": labels,
        "labels_lik50": labels.copy(),
        "labels_a": labels[:, :4].copy(),
    }


def _probability(labels: np.ndarray, strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = np.where(np.isnan(labels), 0.0, 2 * np.nan_to_num(labels) - 1)
    return expit(strength * signal + rng.normal(0, 1.0, labels.shape) - 0.5)


def _write_candidates(directory: Path, data: dict[str, np.ndarray], names: tuple[str, ...]) -> None:
    for index, name in enumerate(names):
        probability = _probability(data["labels"], STRENGTH[name], index)
        metric = selection_metric(data["labels"], probability, data["source"])["selection_metric"]
        folder = directory / name
        if name == "S-R":
            seeds = SEEDS3
            write_json_atomic(
                folder / "settings.json",
                {"seeds": [{"seed": s, "checkpoint": f"seed-{s}/m.pt", "sha256": "x"} for s in seeds]},
            )
            write_json_atomic(
                folder / "results.json",
                {
                    "selection": {"mean": {"selection_metric": metric}},
                    "chosen_epochs": {str(s): 10 for s in seeds},
                },
            )
        else:
            seeds = SEEDS3 if name.startswith("H-") else [42]
            meta = {
                "validation": {"selection_metric": metric},
                "chosen_c": dict.fromkeys(CONDITIONS, 1.0),
                "chosen": {"learning_rate": 3e-4},
                "epoch_by_seed": {str(s): 5 for s in seeds},
            }
            write_json_atomic(folder / "model.json", {"candidate": name, "seeds": seeds})
            write_json_atomic(folder / "candidate.json", meta)
        save_predictions(
            prediction_path(directory, name, "validation", tuple(seeds)),
            data["record_key"],
            probability,
            name,
            tuple(seeds),
        )


@pytest.fixture
def stage3(tmp_path: Path) -> dict[str, object]:
    if not MANIFEST.exists():
        pytest.skip("the MLC-01 v4 manifest is local only")
    data = _validation()
    candidates = tmp_path / "candidates"
    _write_candidates(candidates, data, tuple(STRENGTH))
    return {
        "data": data,
        "candidates": candidates,
        "output": tmp_path / "stage3",
        "doc": tmp_path / "record.md",
        "roots": (tmp_path,),
    }


def _select(setup: dict[str, object], **kwargs: object) -> dict[str, object]:
    return s3.run_select(
        setup["output"],
        setup["doc"],
        setup["candidates"],
        data=setup["data"],
        allow_missing=("F",),
        roots=setup["roots"],
        require_committed=False,
        **kwargs,
    )


def test_ensemble_takes_the_best_member_of_the_three_best_families():
    metrics = {
        "P0": 0.90,
        "H-X": 0.92,
        "L-J": 0.91,
        "H-J": 0.915,
        "L-C": 0.85,
        "H-C": 0.86,
        "G-M": 0.93,
        "H-All": 0.95,
        "S-R": 0.94,
    }
    result = s3.ensemble_members(metrics)
    assert result["members"] == ["G-M", "H-All", "S-R"]
    assert result["family_best"]["xecg"] == "H-X"
    assert result["family_best"]["jepa"] == "H-J"
    assert [row["member"] for row in result["family_ranking"]][:4] == ["H-All", "S-R", "G-M", "H-X"]
    with_f = s3.ensemble_members(metrics | {"F": 0.96})
    assert with_f["members"] == ["H-All", "S-R", "F"]
    assert with_f["family_best"]["xecg"] == "F"
    variants = s3.ensemble_members({"P0": 0.99, "H-X": 0.98, "F": 0.97, "L-J": 0.5, "G-M": 0.6})
    assert variants["members"] == ["P0", "L-J", "G-M"]
    tied = s3.ensemble_members(metrics | {"P0": 0.92})
    assert tied["family_best"]["xecg"] == "P0"
    with pytest.raises(ValueError, match="base candidates"):
        s3.ensemble_members(metrics | {"E": 0.99})
    with pytest.raises(ValueError, match="families"):
        s3.ensemble_members({"P0": 0.9, "H-X": 0.91, "L-J": 0.9})


def test_ties_within_the_margin_go_to_the_simpler_candidate():
    assert select_candidate({"H-All": 0.9700, "S-R": 0.9709, "E": 0.9710})["selected"] == "H-All"
    assert select_candidate({"H-All": 0.9700, "S-R": 0.9705, "E": 0.9711})["selected"] == "S-R"
    assert select_candidate({"H-All": 0.9700, "E": 0.9711})["selected"] == "E"
    assert select_candidate({"P0": 0.96, "F": 0.9605, "E": 0.961})["selected"] == "P0"


def test_select_writes_a_record_that_follows_the_rules(stage3):
    record = _select(stage3)
    assert record["provisional"]
    assert record["missing_candidates"] == ["F"]
    metrics = {row["id"]: row["selection_metric"] for row in record["candidates"]}
    assert list(metrics) == ["P0", "L-J", "L-C", "G-M", "H-X", "H-J", "H-C", "H-All", "S-R", "E"]
    assert record["selected"] == select_candidate(metrics)["selected"]
    members = s3.ensemble_members({k: v for k, v in metrics.items() if k != "E"})["members"]
    assert record["ensemble"]["members"] == members
    ensemble = next(row for row in record["candidates"] if row["id"] == "E")
    assert ensemble["seeds"] == sorted(
        {s for row in record["candidates"] if row["id"] in members for s in row["seeds"]}
    )
    saved = s3.validation_probabilities(record, stage3["data"])
    np.testing.assert_allclose(saved["E"], np.mean([saved[m] for m in members], axis=0), rtol=0, atol=0)
    assert all(row["stored_difference"] == 0 for row in record["candidates"] if row["id"] != "E")
    text = stage3["doc"].read_text()
    assert "PROVISIONAL" in text
    assert f"Selected: {record['selected']}" in text
    with pytest.raises(ValueError, match="absent"):
        s3.run_select(
            stage3["output"],
            stage3["doc"],
            stage3["candidates"],
            data=stage3["data"],
            roots=stage3["roots"],
            overwrite_before_tests=True,
            require_committed=False,
        )


def test_select_refuses_a_stored_metric_it_cannot_reproduce(stage3):
    meta_path = stage3["candidates"] / "P0" / "candidate.json"
    meta = pd.read_json(meta_path, typ="series").to_dict()
    meta["validation"] = {"selection_metric": meta["validation"]["selection_metric"] + 1e-6}
    write_json_atomic(meta_path, meta)
    with pytest.raises(ValueError, match="differs from the stored"):
        _select(stage3)


def test_candidate_info_marks_absent_unfinished_and_smoke_runs(tmp_path):
    assert s3.candidate_info("F", tmp_path) is None
    write_json_atomic(tmp_path / "F" / "model.json", {"candidate": "F", "seeds": SEEDS3})
    with pytest.raises(ValueError, match="not finished"):
        s3.candidate_info("F", tmp_path)
    write_json_atomic(
        tmp_path / "F" / "candidate.json",
        {"validation": {"selection_metric": 0.9}, "chosen": {"top_rate": 1e-5}, "smoke": True},
    )
    with pytest.raises(ValueError, match="smoke"):
        s3.candidate_info("F", tmp_path)
    write_json_atomic(
        tmp_path / "F" / "candidate.json",
        {"validation": {"selection_metric": 0.9}, "chosen": {"top_rate": 1e-5}},
    )
    with pytest.raises(ValueError, match="missing"):
        s3.candidate_info("F", tmp_path)


def test_select_and_fix_refuse_to_overwrite_and_close_after_test_predictions(stage3):
    _select(stage3)
    with pytest.raises(FileExistsError):
        _select(stage3)
    _select(stage3, overwrite_before_tests=True)
    s3.run_fix(stage3["output"], data=stage3["data"], roots=stage3["roots"], require_committed=False)
    with pytest.raises(FileExistsError):
        s3.run_fix(stage3["output"], data=stage3["data"], roots=stage3["roots"], require_committed=False)
    fake = prediction_path(stage3["output"] / "predictions", "P0", "test_external", (42,))
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"")
    with pytest.raises(PermissionError, match="test predictions exist"):
        _select(stage3, overwrite_before_tests=True)
    with pytest.raises(PermissionError, match="test predictions exist"):
        s3.run_fix(
            stage3["output"],
            data=stage3["data"],
            roots=stage3["roots"],
            overwrite_before_tests=True,
            require_committed=False,
        )


def test_fix_masks_undefined_labels_and_fits_platt(stage3):
    _select(stage3)
    points = s3.run_fix(stage3["output"], data=stage3["data"], roots=stage3["roots"], require_committed=False)
    record, record_sha = s3.read_record(stage3["output"])
    assert points["selection_record_sha256"] == record_sha
    labels = stage3["data"]["labels"]
    probability = s3.validation_probabilities(record, stage3["data"])["P0"]
    for column, name in enumerate(CONDITIONS):
        defined = ~np.isnan(labels[:, column])
        expected = specificity_threshold(labels[defined, column], probability[defined, column])
        assert points["candidates"]["P0"]["thresholds"][name] == expected
        assert points["candidates"]["P0"]["validation_at_threshold"][name]["specificity"] >= 0.9
    loaded, _ = s3.read_operating_points(stage3["output"], record_sha)
    assert loaded["E"]["platt"].shape == (14, 2)
    assert np.isfinite(loaded["E"]["platt"]).all()
    with pytest.raises(ValueError, match="another selection record"):
        s3.read_operating_points(stage3["output"], "0" * 64)


def test_threshold_ignores_scores_of_undefined_entries():
    rng = np.random.default_rng(3)
    labels = (rng.random((500, 14)) < 0.3).astype(float)
    probability = rng.random((500, 14))
    undefined = rng.random((500, 14)) < 0.2
    labels[undefined] = np.nan
    first = s3.operating_points(labels, probability)["thresholds"]
    moved = probability.copy()
    moved[undefined] = np.where(rng.random(undefined.sum()) < 0.5, 0.0, 1.0)
    assert s3.operating_points(labels, moved)["thresholds"] == first


def test_damped_fit_matches_the_library_and_recovers_known_platt_parameters():
    rng = np.random.default_rng(11)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.random(20000) < expit(-0.7 + 1.6 * logit(p))).astype(float)
    a, b = damped_logistic_fit(y, logit(p))
    assert abs(a[0] + 0.7) < 0.06
    assert abs(b[0] - 1.6) < 0.06
    a0, b0 = logistic_fit(y, logit(p))
    assert abs(a[0] - a0[0]) < 1e-8
    assert abs(b[0] - b0[0]) < 1e-8
    labels = np.column_stack([y] * 14)
    params = fit_platt_damped(labels, np.column_stack([p] * 14))
    np.testing.assert_allclose(params, np.tile([a[0], b[0]], (14, 1)), rtol=0, atol=1e-12)
    separated = np.r_[np.zeros(50), np.ones(50)]
    assert np.isnan(damped_logistic_fit(separated, np.r_[np.full(50, -2.0), np.full(50, 2.0)])[0][0])


def _well_separated(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive_mode = rng.random(15000) < 0.02
    x = np.where(positive_mode, rng.normal(2.0, 1.5, 15000), rng.normal(-12.0, 1.5, 15000))
    return (rng.random(15000) < expit(x)).astype(float), logit(expit(x))


def test_damped_fit_converges_on_well_separated_scores_where_the_library_does_not():
    from sklearn.linear_model import LogisticRegression

    from ecg_experiment.mlc02_bootstrap import cluster_resamples
    from ecg_experiment.mlc02_evaluation import fit_platt

    y, x = _well_separated(0)
    assert 200 < y.sum() < 300
    assert np.isnan(logistic_fit(y, x)[0][0])
    a, b = damped_logistic_fit(y, x)
    reference = LogisticRegression(C=np.inf, max_iter=100000, tol=1e-14).fit(x[:, None], y)
    assert abs(a[0] - reference.intercept_[0]) < 1e-5
    assert abs(b[0] - reference.coef_[0, 0]) < 1e-5
    assert abs(a[0]) < 0.4
    assert abs(b[0] - 1.0) < 0.25
    labels, probability = np.column_stack([y] * 14), np.column_stack([expit(x)] * 14)
    with pytest.raises(ValueError, match="did not converge"):
        fit_platt(labels, probability)
    np.testing.assert_allclose(
        fit_platt_damped(labels, probability), np.tile([a[0], b[0]], (14, 1)), atol=1e-12
    )
    resamples = cluster_resamples(np.arange(15000).astype(str), labels, draws=50)
    report, _ = calibration_report(labels, probability, resamples)
    for row in report["conditions"]:
        assert row["calibration_slope_nan_draws"] == 0
        assert row["calibration_slope_ci_low"] <= row["calibration_slope"] <= row["calibration_slope_ci_high"]


def test_calibration_report_gives_intervals_around_the_point_estimate():
    from ecg_experiment.mlc02_bootstrap import cluster_resamples

    rng = np.random.default_rng(5)
    labels = (rng.random((800, 14)) < 0.3).astype(float)
    probability = _probability(labels, 1.0, 2)
    resamples = cluster_resamples(np.arange(800).astype(str), labels, draws=100)
    report, samples = calibration_report(labels, probability, resamples)
    assert samples.shape == (100, 14, 2)
    for row in report["conditions"]:
        assert row["calibration_slope_ci_low"] <= row["calibration_slope"] <= row["calibration_slope_ci_high"]
        assert row["calibration_slope_nan_draws"] == 0


@pytest.mark.skipif(
    not (s3.CANDIDATES_DIR / "G-M" / "model.json").exists(), reason="needs the Stage 2 outputs"
)
def test_damped_fit_converges_where_the_library_does_not_on_real_validation_scores():
    from sklearn.linear_model import LogisticRegression

    from ecg_experiment.mlc02_data import load_predictions

    data = s3.validation_data()
    probability = load_predictions(
        prediction_path(s3.CANDIDATES_DIR, "G-M", "validation", (20261110,)), data["record_key"]
    )["probability"]
    column = CONDITIONS.index("left_axis_deviation")
    defined = ~np.isnan(data["labels"][:, column])
    y, x = data["labels"][defined, column], logit(probability[defined, column])
    assert np.isnan(logistic_fit(y, x)[0][0])
    a, b = damped_logistic_fit(y, x)
    reference = LogisticRegression(C=np.inf, max_iter=100000, tol=1e-14).fit(x[:, None], y)
    assert abs(a[0] - reference.intercept_[0]) < 1e-5
    assert abs(b[0] - reference.coef_[0, 0]) < 1e-5


def test_test_predict_refuses_without_flag_record_or_commit(stage3):
    from ecg_experiment.mlc02_stage3_predict import run_test_predict

    arguments = {
        "output": stage3["output"],
        "doc": stage3["doc"],
        "candidates_dir": stage3["candidates"],
        "roots": stage3["roots"],
    }
    with pytest.raises(PermissionError, match="i-am-the-final-run"):
        run_test_predict(False, "cpu", **arguments)
    with pytest.raises(FileNotFoundError, match="run select first"):
        run_test_predict(True, "cpu", **arguments)
    _select(stage3)
    with pytest.raises(PermissionError, match="inside the repository"):
        run_test_predict(True, "cpu", **arguments)
    tracked = ROOT / "docs/mlc02-condition-models-protocol-2026-11-10.md"
    with pytest.raises(PermissionError, match="not the document generated"):
        run_test_predict(True, "cpu", **(arguments | {"doc": tracked}))


def test_test_predict_runs_once(stage3):
    from ecg_experiment.mlc02_stage3_predict import run_test_predict

    _select(stage3)
    fake = prediction_path(stage3["output"] / "predictions", "E", "sph_calibration", tuple(SEEDS3))
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"")
    with pytest.raises(PermissionError, match="test predictions exist"):
        run_test_predict(
            True, "cpu", stage3["output"], stage3["doc"], stage3["candidates"], roots=stage3["roots"]
        )


def test_dry_run_and_block_reader_refuse_test_roles(tmp_path):
    from ecg_experiment.mlc02_stage3_predict import open_role_blocks, run_dry_run

    with pytest.raises(PermissionError, match="validation role only"):
        run_dry_run(tmp_path, "cpu", role="test_external", output=tmp_path)
    rows = pd.DataFrame(
        {
            "record_key": ["sph:1"],
            "role": ["validation"],
            "block": ["test_external__sph"],
            "row": [0],
            "sha256_10s": ["x"],
        }
    )
    global OPENED
    OPENED = []
    try:
        with pytest.raises(PermissionError):
            open_role_blocks(rows, "validation", tmp_path)
        with pytest.raises(PermissionError):
            open_role_blocks(rows.assign(role="test_external"), "validation", tmp_path)
    finally:
        opened, OPENED = OPENED, None
    assert not [path for path in opened if str(tmp_path) in path]


def _closed_paths(opened: list[str]) -> list[str]:
    return [
        path
        for path in opened
        if any(word in Path(path).name for word in CLOSED_WORDS)
        or "/features/" in path
        or "/waveforms/" in path
    ]


def test_select_and_fix_never_open_a_test_role_file(stage3):
    for name in STRENGTH:
        for role in CLOSED_WORDS:
            seeds = (42,) if name in ("P0", "L-J", "L-C", "G-M") else tuple(SEEDS3)
            poison = prediction_path(stage3["candidates"], name, role, seeds)
            poison.write_bytes(b"not a prediction file")
    roots = (stage3["output"],)
    global OPENED
    OPENED = []
    try:
        s3.run_select(
            stage3["output"],
            stage3["doc"],
            stage3["candidates"],
            data=stage3["data"],
            allow_missing=("F",),
            roots=roots,
            require_committed=False,
        )
        s3.run_fix(stage3["output"], data=stage3["data"], roots=roots, require_committed=False)
    finally:
        opened, OPENED = OPENED, None
    assert opened
    assert not _closed_paths(opened)


@pytest.mark.skipif(
    not (s3.CANDIDATES_DIR / "S-R" / "settings.json").exists(), reason="needs the Stage 2 outputs"
)
def test_select_and_fix_on_real_outputs_open_no_test_role(tmp_path):
    global OPENED
    OPENED = []
    try:
        record = s3.run_select(
            tmp_path, tmp_path / "record.md", allow_missing=("F",), roots=(tmp_path,), require_committed=False
        )
        s3.run_fix(tmp_path, roots=(tmp_path,), require_committed=False)
    finally:
        opened, OPENED = OPENED, None
    assert not _closed_paths(opened)
    assert any(path.endswith("manifest.csv") for path in opened)
    assert record["test_roles_opened"] is False


def test_evaluate_sets_runs_on_synthetic_sets():
    from ecg_experiment.mlc02_stage3_evaluate import evaluate_sets

    sets = {
        role: _validation(per_source=60, seed=seed)
        for seed, role in enumerate(("test_same_source", "test_external", "sph_calibration"))
    }
    names = ("P0", "H-All", "E")
    predictions = {
        name: {role: _probability(data["labels"], STRENGTH.get(name, 1.8), i) for role, data in sets.items()}
        for i, name in enumerate(names)
    }
    operating = {}
    for name in names:
        fitted = s3.operating_points(sets["sph_calibration"]["labels"], predictions[name]["sph_calibration"])
        operating[name] = {
            "thresholds": np.array([fitted["thresholds"][c] for c in CONDITIONS]),
            "platt": np.array(
                [[fitted["platt"][c]["intercept"], fitted["platt"][c]["slope"]] for c in CONDITIONS]
            ),
        }
    result = evaluate_sets(sets, predictions, operating, "E", draws=40, workers=2)
    assert set(result["comparisons"]["test_external"]) == {"H-All", "E"}
    assert result["confirmatory"]["selected"] == "E"
    assert result["confirmatory"]["better"] == result["confirmatory"]["macro"]["better"]
    assert set(result["candidates"]["test_same_source"]["P0"]) == {
        "raw",
        "platt",
        "raw_calibration_damped",
        "platt_calibration_damped",
    }
    assert set(result["secondary_sph_calibration"]["candidates"]) == set(names)
    assert result["likelihood_50"]["entries_made_undefined"] == 0
    assert "top_1" in result["audits"]["test_external"]["E"]["multiclass"]
    same = evaluate_sets(sets, predictions, operating, "P0", draws=40, workers=2)
    assert same["confirmatory"]["selected_is_reference"]
    assert not same["confirmatory"]["better"]
