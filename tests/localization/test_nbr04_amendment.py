import json

import numpy as np
import pytest

from ecg_experiment.localization import nbr01_export as ne
from ecg_experiment.localization import nbr04_amendment as amendment
from ecg_experiment.localization import nbr04_amendment_export as ex


def test_factorial_effects_and_interaction():
    values = {"C6": 1.0, "L6": 1.5, "D6": 2.0, "L6D": 4.0}
    result = {f"{arm}/all_age_blend/selectivity": value for arm, value in values.items()}
    found = amendment.factorial(result)
    assert found["factorial/leave_region_out_without_removal/all_age_blend/selectivity"] == pytest.approx(0.5)
    assert found["factorial/leave_region_out_with_removal/all_age_blend/selectivity"] == pytest.approx(2.0)
    assert found["factorial/removal_without_leave_out/all_age_blend/selectivity"] == pytest.approx(1.0)
    assert found["factorial/removal_with_leave_out/all_age_blend/selectivity"] == pytest.approx(2.5)
    assert found["factorial/interaction/all_age_blend/selectivity"] == pytest.approx(1.5)


def interval(point: float, low: float) -> dict:
    return {"point": point, "low": low, "high": point + 0.1}


@pytest.mark.parametrize(("passing", "chosen"), [((), None), (("D6",), "D6"), (("L6", "D6"), "D6")])
def test_three_arm_decision(passing, chosen):
    sph, ptb = {}, {}
    for arm, selectivity in (("L6", 0.80), ("D6", 0.90), ("L6D", 0.85)):
        low = 0.01 if arm in passing else -0.1
        sph[f"{arm}/all_age_blend/selectivity"] = interval(selectivity, 0.5)
        sph[f"{arm}-P0/all_age_blend/selectivity"] = interval(0.05, low)
        sph[f"{arm}-P0/all_age_blend/territory"] = interval(0.0, -0.04)
        sph[f"{arm}-P0/all_age_blend/abnormal/component_flip"] = interval(0.0, -0.04)
        ptb[f"{arm}-P0/all_age_blend/selectivity"] = interval(0.02, -0.05)
        ptb[f"{arm}-P0/all_age_blend/territory"] = interval(0.0, -0.04)
    sph["D6-L6/all_age_blend/selectivity"] = interval(0.1, 0.0)
    sph["L6-D6/all_age_blend/selectivity"] = interval(-0.1, -0.2)
    decision = amendment.decide(sph, ptb)
    assert decision["chosen"] == chosen
    assert decision["exported_arm"] == (chosen or "L6")
    assert (len(decision["passing_differences"]) == 2) is (len(passing) == 2)


def test_amended_export_validation():
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
        for arm in ex.ARMS
        for model in ("all_age_blend", "ptb_baseline_blend")
    }
    decisions = {"L6": {}, "D6": {}, "L6D": {}, "passing": [], "chosen": None, "exported_arm": "D6"}
    data = {
        "schema_version": 1,
        "fs": 250,
        "arm": "D6",
        "counterpart_note": ex.WHOLE_RECORD_NOTE,
        "leads": list(ne.ns.LEAD_NAMES),
        "components": ["P", "QRS", "ST-T"],
        "models": {"all_age_blend": {"threshold": 0.65}, "ptb_baseline_blend": {"threshold": 0.73}},
        "examples": [example],
        "summary": {"statistics": statistics, "decisions": decisions},
    }
    ex.validate(json.loads(json.dumps(data)))
    data["summary"]["decisions"].pop("passing")
    with pytest.raises(ValueError, match="decisions"):
        ex.validate(data)
