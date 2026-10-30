import json

import numpy as np
import pytest

from ecg_experiment import normative_map062 as nm
from ecg_experiment import replacement070 as rp
from ecg_experiment.localization import nbr01_counterpart as nc
from ecg_experiment.localization import nbr01_export as ne
from ecg_experiment.localization import nbr02_analysis as na2
from ecg_experiment.localization import nbr02_export as ex
from ecg_experiment.localization import nbr02_templates as nt


def synthetic_model(rank: int) -> nm.PopulationModel:
    rng = np.random.default_rng(3)
    covariates = nm.design(rng.uniform(50, 100, 60), rng.uniform(18, 80, 60), rng.integers(0, 2, 60))
    medians = rng.normal(scale=0.2, size=(60, 12 * nm.BINS)) + covariates[:, 1:2] * 0.05
    return nm.fit_population(medians, covariates, rank)


def synthetic_features() -> nm.RecordFeatures:
    rng = np.random.default_rng(4)
    empty = np.zeros(0)
    median = rng.normal(scale=0.2, size=(12, nm.BINS))
    return nm.RecordFeatures(times=empty, knots=empty, beats=empty, median=median, dispersion=empty,
                             dominant=empty, focal=empty,
                             intervals=np.array([72.0, 0.09, 0.4, 0.16]), prematurity=empty, qrs_change=empty,
                             noise=empty)


def test_rank0_template_matches_070():
    model = synthetic_model(0)
    template, coordinates = nt.record_template(synthetic_features(), 35.0, 1.0, model)
    expected = rp.template_beat(nc.normative(model), 72.0, 35.0, 1.0)
    assert len(coordinates) == 0
    assert np.array_equal(template, expected)


def test_rank6_template_is_median_minus_population_residual():
    model = synthetic_model(nm.SUBSPACE)
    features = synthetic_features()
    template, coordinates = nt.record_template(features, 35.0, 1.0, model)
    covariates = nm.design(features.intervals[:1], np.array([35.0]), np.array([1.0]))[0]
    residual = nm.population_residual(model, features.median.ravel(), covariates)
    assert coordinates.shape == (nm.SUBSPACE,)
    assert np.allclose(template.ravel(), features.median.ravel() - residual, atol=1e-12)
    assert nt.clipped(coordinates, model) <= nm.SUBSPACE


def test_arm_arrays_replace_only_counterpart_variants():
    base = np.arange(2 * 105 * 2 * 3, dtype=float).reshape(2, 105, 2, 3)
    energy = np.arange(2 * 105, dtype=float).reshape(2, 105)
    scores, replaced = na2.arm_arrays(base, energy, -np.ones((2, 52, 2, 3)), -np.ones((2, 52)))
    assert np.array_equal(scores[:, 0], base[:, 0])
    assert np.array_equal(scores[:, 53:], base[:, 53:])
    assert (scores[:, 1:53] == -1).all()
    assert (replaced[:, 1:53] == -1).all()
    assert np.array_equal(replaced[:, 53:], energy[:, 53:])


def interval(point: float, low: float, high: float) -> dict:
    return {"point": point, "low": low, "high": high, "valid_draws": 2000}


@pytest.mark.parametrize(("selectivity_low", "territory_low", "drop_low", "expected"),
                         [(0.01, -0.04, 0.3, True), (-0.01, -0.04, 0.3, False), (0.01, -0.06, 0.3, False),
                          (0.01, -0.04, -0.01, False)])
def test_decision_requires_all_three_parts(selectivity_low, territory_low, drop_low, expected):
    statistics = {"C6-P0/all_age_blend/selectivity": interval(0.1, selectivity_low, 0.2),
                  "C6-P0/all_age_blend/territory": interval(0.0, territory_low, 0.05),
                  "C6/all_age_blend/abnormal/full_drop": interval(0.5, drop_low, 0.6)}
    assert na2.decision(statistics)["C6_replaces_P0"] is expected


def test_export_validation():
    grid = [[0.5, 0.4, 0.3] for _ in range(12)]
    scores = {"original": 0.9, "full": 0.1, "lead": [0.8] * 12, "component": [0.7, 0.6, 0.5], "grid": grid,
              "self_swap_grid": grid, "self_swap_lead": [0.9] * 12, "self_swap_full": 0.88, "hgb_grid": grid,
              "xecg_grid": grid, "hgb_original": 0.9, "xecg_original": 0.9}
    signal = np.zeros((12, 2500)).tolist()
    example = {"id": "A00001", "group": "inferior", "statements": ["Inferior MI, old"], "age": 55, "sex": "M",
               "signal": signal, "counterpart": signal,
               "components": {"P": [[10, 40]], "QRS": [[50, 70]], "ST-T": [[71, 150]]},
               "scores": {"all_age_blend": scores, "ptb_baseline_blend": scores}, "energy_grid": grid}
    statistics = {f"{arm}/{model}/territory": interval(0.3, 0.2, 0.4)
                  for arm in nt.ARMS for model in ("all_age_blend", "ptb_baseline_blend")}
    decisions = {name: {} for name in ("A_selectivity", "B_territory_noninferior", "C_operator_check")}
    data = {"schema_version": 1, "fs": 250, "arm": "C6", "fit_set": {"ptbxl": 3038},
            "leads": list(ne.ns.LEAD_NAMES), "components": ["P", "QRS", "ST-T"],
            "models": {"all_age_blend": {"threshold": 0.65}, "ptb_baseline_blend": {"threshold": 0.73}},
            "examples": [example], "summary": {"statistics": statistics, "n": {}, "decisions":
                                               {**decisions, "C6_replaces_P0": False}}}
    ex.validate(json.loads(json.dumps(data)))
    data["summary"]["statistics"].pop("P6/all_age_blend/territory")
    with pytest.raises(ValueError, match="arm statistics"):
        ex.validate(data)


def test_arm_bootstrap_runs_and_pairs_contrasts():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.array(["inferior", "anterior", "lbbb", "rbbb", "stt", "normal"]), 6)
    count = len(groups)
    base = {"scores": rng.uniform(size=(count, 105, 2, 3)), "energy": rng.uniform(size=(count, 105))}
    new = {"scores": rng.uniform(size=(count, 3, 52, 2, 3)), "energy": rng.uniform(size=(count, 3, 52))}
    thresholds = {f"{model}_{kind}": 0.5 for model in ("all_age", "ptb_baseline")
                  for kind in ("hgb", "xecg", "blend")}
    diagnostics = {f"template_{pair}": rng.uniform(size=len(groups)) for pair in na2.TEMPLATE_PAIRS}
    diagnostics |= {f"{arm}_clipped": rng.integers(0, 3, len(groups)) for arm in ("P6", "C6")}
    tables = na2.arm_tables(base, new, thresholds)
    result = na2.bootstrap(tables, groups, np.arange(len(groups)).astype(str), diagnostics, draws=5, seed=0)
    contrast = result["C6-P0/all_age_blend/selectivity"]["point"]
    assert contrast == pytest.approx(result["C6/all_age_blend/selectivity"]["point"]
                                     - result["P0/all_age_blend/selectivity"]["point"])
    assert result["C6-P0/all_age_blend/swap_territory"]["point"] == 0.0
    assert "template/C6_C0/lbbb" in result
