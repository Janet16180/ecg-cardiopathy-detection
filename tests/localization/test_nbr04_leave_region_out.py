import json

import numpy as np
import pytest

from ecg_experiment import normative_map062 as nm
from ecg_experiment.localization import nbr01_counterpart as nc
from ecg_experiment.localization import nbr01_export as ne
from ecg_experiment.localization import nbr04_export as ex
from ecg_experiment.localization import nbr04_study as study
from ecg_experiment.localization import nbr04_templates as nt


def synthetic_model(rank: int = nm.SUBSPACE) -> nm.PopulationModel:
    rng = np.random.default_rng(7)
    covariates = nm.design(rng.uniform(50, 100, 80), rng.uniform(18, 80, 80), rng.integers(0, 2, 80))
    medians = rng.normal(scale=0.2, size=(80, 12 * nm.BINS)) + covariates[:, 1:2] * 0.05
    return nm.fit_population(medians, covariates, rank)


def test_masked_coefficients_reduce_to_062():
    model = synthetic_model()
    standard = np.random.default_rng(8).normal(size=12 * nm.BINS)
    keep = np.ones(12 * nm.BINS, bool)
    assert np.array_equal(
        nt.masked_subspace_coefficients(model, standard, keep), nm.subspace_coefficients(model, standard)
    )
    assert not nt.masked_subspace_coefficients(model, standard, ~keep).any()


def test_region_keep_masks():
    items = {item.name: item for item in nc.variants()}
    grid = nt.region_keep(items["grid_1_1"]).reshape(12, nm.BINS)
    assert not grid[1, 150:200].any()
    assert grid[1, :150].all()
    assert grid[1, 200:].all()
    assert grid[[0, *range(2, 12)]].all()
    lead = nt.region_keep(items["lead_3"]).reshape(12, nm.BINS)
    assert not lead[3, :330].any()
    assert lead[3, 330:].all()
    component = nt.region_keep(items["component_2"]).reshape(12, nm.BINS)
    assert not component[:, 200:330].any()
    assert component[:, :200].all()
    assert not nt.region_keep(items["full"]).any()


def test_region_signals_match_nbr01_with_one_difference():
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(12, 5000))
    components = np.zeros((3, 5000), bool)
    components[0, 1000:1050] = components[1, 1100:1150] = components[2, 1200:1400] = True
    difference = rng.normal(size=(12, 5000)).astype(np.float32)
    items = nc.variants()[1:53]
    expected, energy = nc.variant_signals(
        raw, {"difference": difference, "swap": difference, "components": components}, items
    )
    found, found_energy = nt.region_signals(
        raw, components, np.repeat(difference[None], len(items), axis=0), items
    )
    assert np.array_equal(found, expected)
    assert np.array_equal(found_energy, energy)


def test_remove_directions():
    model = synthetic_model()
    rng = np.random.default_rng(10)
    normals = rng.normal(size=(80, 12 * nm.BINS))
    directions, _ = np.linalg.qr(rng.normal(size=(12 * nm.BINS, 2)))
    fitted, info = nt.remove_directions(model, directions, normals)
    assert np.allclose(fitted.basis.T @ directions, 0.0, atol=1e-10)
    assert np.allclose(fitted.basis.T @ fitted.basis, np.eye(fitted.basis.shape[1]), atol=1e-10)
    assert np.all(np.diff(fitted.eigenvalues) <= 0)
    assert np.array_equal(fitted.coefficients, model.coefficients)
    assert info["rank"] == fitted.basis.shape[1]


def interval(point: float, low: float) -> dict:
    return {"point": point, "low": low, "high": point + 0.1, "valid_draws": 2000}


@pytest.mark.parametrize(
    ("l6", "l6d", "tie", "chosen"),
    [
        (True, True, -0.01, "L6"),
        (True, True, 0.01, "L6D"),
        (False, True, 0.01, "L6D"),
        (False, False, 0.0, None),
    ],
)
def test_decide_and_tie_rule(l6, l6d, tie, chosen):
    sph, ptb = {}, {}
    for arm, passing in (("L6", l6), ("L6D", l6d)):
        low = 0.01 if passing else -0.2
        sph[f"{arm}-P0/all_age_blend/selectivity"] = interval(0.1, low)
        sph[f"{arm}-P0/all_age_blend/territory"] = interval(0.0, -0.04)
        sph[f"{arm}-P0/all_age_blend/abnormal/component_flip"] = interval(0.0, -0.04)
        ptb[f"{arm}-P0/all_age_blend/selectivity"] = interval(0.05, -0.01)
        ptb[f"{arm}-P0/all_age_blend/territory"] = interval(0.0, -0.04)
    sph["L6D-L6/all_age_blend/selectivity"] = interval(0.05, tie)
    decision = study.decide(sph, ptb)
    assert decision["chosen"] == chosen
    assert decision["exported_arm"] == (chosen or "L6")


def test_export_validation():
    grid = [[0.5, 0.4, 0.3] for _ in range(12)]
    scores = {
        "original": 0.9,
        "full": 0.1,
        "lead": [0.8] * 12,
        "component": [0.7, 0.6, 0.5],
        "grid": grid,
        "self_swap_grid": grid,
        "self_swap_lead": [0.9] * 12,
        "self_swap_full": 0.88,
        "hgb_grid": grid,
        "xecg_grid": grid,
        "hgb_original": 0.9,
        "xecg_original": 0.9,
    }
    signal = np.zeros((12, 2500)).tolist()
    example = {
        "id": "A00001",
        "group": "rbbb",
        "statements": ["Right bundle-branch block"],
        "age": 40,
        "sex": "F",
        "signal": signal,
        "counterpart": signal,
        "components": {"P": [[10, 40]], "QRS": [[50, 70]], "ST-T": [[71, 150]]},
        "scores": {"all_age_blend": scores, "ptb_baseline_blend": scores},
        "energy_grid": grid,
    }
    statistic = {"point": 0.3, "low": 0.2, "high": 0.4}
    statistics = {
        f"{arm}/{model}/territory": statistic
        for arm in ("P0", "L6", "L6D")
        for model in ("all_age_blend", "ptb_baseline_blend")
    }
    data = {
        "schema_version": 1,
        "fs": 250,
        "arm": "L6",
        "counterpart_note": ex.COUNTERPART_NOTE,
        "leads": list(ne.ns.LEAD_NAMES),
        "components": ["P", "QRS", "ST-T"],
        "models": {"all_age_blend": {"threshold": 0.65}, "ptb_baseline_blend": {"threshold": 0.73}},
        "examples": [example],
        "summary": {
            "statistics": statistics,
            "decisions": {"L6": {}, "L6D": {}, "chosen": None, "exported_arm": "L6"},
        },
    }
    ex.validate(json.loads(json.dumps(data)))
    data["arm"] = "C6"
    with pytest.raises(ValueError, match=r"\w"):
        ex.validate(data)
