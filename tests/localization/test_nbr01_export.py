import json

import numpy as np
import pytest

from ecg_experiment.localization import nbr01_export as ne


def example(group: str = "inferior") -> dict:
    grid = [[0.5, 0.4, 0.3] for _ in range(12)]
    scores = {"original": 0.9, "full": 0.1, "lead": [0.8] * 12, "component": [0.7, 0.6, 0.5], "grid": grid,
              "self_swap_grid": grid, "self_swap_lead": [0.9] * 12, "self_swap_full": 0.88, "hgb_grid": grid,
              "xecg_grid": grid, "hgb_original": 0.9, "xecg_original": 0.9}
    signal = np.round(np.random.default_rng(0).normal(size=(12, 2500)), 3).tolist()
    return {"id": "A00001", "group": group, "statements": ["Inferior MI, old"], "age": 55, "sex": "M",
            "signal": signal, "counterpart": signal,
            "components": {"P": [[10, 40]], "QRS": [[50, 70]], "ST-T": [[71, 150]]},
            "scores": {"all_age_blend": scores, "ptb_baseline_blend": scores}, "energy_grid": grid}


def payload() -> dict:
    return {"schema_version": 1, "fs": 250, "leads": list(ne.ns.LEAD_NAMES),
            "components": ["P", "QRS", "ST-T"],
            "models": {"all_age_blend": {"threshold": 0.65}, "ptb_baseline_blend": {"threshold": 0.73}},
            "examples": [example(), example("normal")], "summary": summary()}


def summary() -> dict:
    statistics = {"energy/territory": {"point": 0.1, "low": 0.05, "high": 0.15},
                  "all_age_blend/territory": {"point": 0.3, "low": 0.2, "high": 0.4},
                  "ptb_baseline_blend/territory": {"point": 0.4, "low": 0.3, "high": 0.5}}
    return {"statistics": statistics, "n": {"inferior": {"records": 49}}, "decisions": {"primary": "shown"},
            "primary_endpoint": "territory"}


def test_examples_schema_validates_after_json_round_trip():
    ne.validate(json.loads(json.dumps(payload())))


@pytest.mark.parametrize("breaking", ["signal", "grid", "group", "models", "swap_lead", "summary"])
def test_examples_schema_rejects_bad_shapes(breaking):
    data = payload()
    if breaking == "signal":
        data["examples"][0]["signal"] = data["examples"][0]["signal"][:11]
    elif breaking == "grid":
        data["examples"][0]["scores"]["all_age_blend"]["grid"] = [[0.1, 0.2]] * 12
    elif breaking == "group":
        data["examples"][0]["group"] = "posterior"
    elif breaking == "swap_lead":
        data["examples"][0]["scores"]["all_age_blend"] = {**data["examples"][0]["scores"]["all_age_blend"],
                                                          "self_swap_lead": [0.9] * 11}
    elif breaking == "summary":
        data["summary"]["statistics"].pop("ptb_baseline_blend/territory")
    else:
        data["models"].pop("ptb_baseline_blend")
    with pytest.raises(ValueError, match=r"\w"):
        ne.validate(data)


def test_runs_and_draw():
    mask = np.zeros(5000, bool)
    mask[100:120] = mask[4990:] = True
    assert ne.runs(mask) == [[50, 60], [2495, 2500]]
    groups = np.array(["inferior"] * 10 + ["anterior"] * 10 + ["rbbb"] * 5 + ["normal"] * 5)
    ids = np.array([f"A{i:03d}" for i in range(len(groups))])
    chosen = ne.draw_examples(groups, ids)
    assert chosen == ne.draw_examples(groups, ids)
    names = ("inferior", "anterior", "rbbb", "normal")
    counts = {group: int(np.sum(groups[chosen] == group)) for group in names}
    assert counts == {"inferior": 4, "anterior": 4, "rbbb": 2, "normal": 2}
