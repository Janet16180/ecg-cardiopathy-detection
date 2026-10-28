import json

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.eda.ptbxl import diagnostic_classes, load_statements
from ecg_experiment.localization import nbr01_export as ne
from ecg_experiment.localization import nbr03_export as ex
from ecg_experiment.localization import nbr03_ptbxl as ptb
from ecg_experiment.localization import nbr03_study as study


@pytest.fixture(scope="module")
def classes() -> dict:
    return diagnostic_classes(load_statements())


@pytest.mark.parametrize(
    ("codes", "target", "expected"),
    [
        ({"IMI", "SR"}, 1, "inferior"),
        ({"ISCIN", "CRBBB"}, 1, "inferior"),
        ({"ASMI"}, 1, "anterior"),
        ({"INJAS", "ISCAS"}, 1, "anterior"),
        ({"LMI"}, 1, "lateral"),
        ({"IMI", "ASMI"}, 1, None),
        ({"ILMI"}, 1, None),
        ({"IPMI"}, 1, None),
        ({"IMI", "PMI"}, 1, None),
        ({"CLBBB"}, 1, "lbbb"),
        ({"CLBBB", "CRBBB"}, 1, "lbbb"),
        ({"CRBBB", "NDT"}, 1, "rbbb"),
        ({"NDT", "SR"}, 1, "stt"),
        ({"NST_", "LVH"}, 1, None),
        ({"ISC_", "DIG"}, 1, None),
        ({"NORM", "SR"}, 0, "normal"),
        ({"NORM", "NDT"}, 1, "stt"),
    ],
)
def test_group_of(classes, codes, target, expected):
    assert ptb.group_of(codes, target, classes) == expected


def test_covariates_and_fold_guard():
    assert np.isnan(ptb.covariates(pd.Series({"age": "300.0", "sex": "1"}))[0])
    assert ptb.covariates(pd.Series({"age": "54.0", "sex": "0"})) == (54.0, 0.0)
    with pytest.raises(ValueError, match="fold 9"):
        ptb.prepare_record(None, pd.Series({"strat_fold": "10"}), None)


def test_primary_groups_and_parts():
    groups = np.array(["inferior", "anterior", "lateral", "normal"])
    renamed = study.primary_groups(groups, ["inferior", "anterior"])
    assert renamed.tolist() == ["inferior", "anterior", "lateral_secondary", "normal"]
    shares = np.zeros((4, 12))
    shares[0, [1, 2, 5]] = 1 / 3
    shares[1, [6, 7, 8, 9]] = 1 / 4
    shares[2, [0, 4, 10, 11]] = 1 / 4
    shares[3] = 1 / 12
    parts = study.territory_parts(shares, groups)
    assert set(parts) == {"inferior", "anterior", "lateral"}
    assert parts["lateral"] == pytest.approx(1.0)
    assert set(study.territory_parts(shares, renamed)) == {"inferior", "anterior"}


def test_export_validates_dataset():
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
        "id": "21",
        "group": "lateral",
        "statements": ["LMI: lateral myocardial infarction"],
        "age": 90,
        "sex": "F",
        "signal": signal,
        "counterpart": signal,
        "components": {"P": [[10, 40]], "QRS": [[50, 70]], "ST-T": [[71, 150]]},
        "scores": {"all_age_blend": scores, "ptb_baseline_blend": scores},
        "energy_grid": grid,
    }
    statistic = {"point": 0.3, "low": 0.2, "high": 0.4}
    data = {
        "schema_version": 1,
        "dataset": "ptbxl_fold9",
        "fs": 250,
        "leads": list(ne.ns.LEAD_NAMES),
        "components": ["P", "QRS", "ST-T"],
        "models": {"all_age_blend": {"threshold": 0.9}, "ptb_baseline_blend": {"threshold": 0.68}},
        "examples": [example],
        "summary": {
            "statistics": {"all_age_blend/territory": statistic, "ptb_baseline_blend/territory": statistic},
            "n": {},
            "decisions": {},
            "primary_endpoint": "territory",
        },
    }
    ex.validate(json.loads(json.dumps(data)))
    data["dataset"] = "sph"
    with pytest.raises(ValueError, match="dataset"):
        ex.validate(data)
    assert ex.export_age("300.0") == 90
