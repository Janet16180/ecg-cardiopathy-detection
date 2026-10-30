"""Full aggregate parity must reject changes beyond confusion counts and AUROC."""

import copy

import pytest

from ecg_experiment.pr74_strict_parity_verifier import MODEL_IDS, compare_full_payloads


def artifact():
    evaluation = {"threshold": 0.5, "calibration": {"flagged_normals": 1},
                  "metrics": {"tp": 3, "tn": 4, "fp": 1, "fn": 2, "auroc": 0.8, "precision": 0.75},
                  "sweep": {"threshold": [0.1, 0.5], "tp": [4, 3]}}
    return {"complete": True, "models": [
        {"id": name, "evaluations": [dict(copy.deepcopy(evaluation), pool=site) for site in ("ptb", "sph")]}
        for name in sorted(MODEL_IDS)],
        "primary": {"development_advantage": False,
                    "sensitivity_difference": {"ci_low": -0.01, "ci_high": 0.02, "draws": 2000}}}


def test_equal_full_payloads_pass_even_when_model_order_differs():
    expected = artifact()
    actual = copy.deepcopy(expected)
    actual["models"].reverse()
    result = compare_full_payloads(actual, expected)
    assert result["model_site_points"] == 30
    assert result["all_complete_evaluation_payloads_exact"] is True
    assert result["complete_primary_exact"] is True


@pytest.mark.parametrize("mutation", ["threshold", "sweep", "ci", "precision", "calibration", "extra"])
def test_any_complete_scientific_payload_mutation_fails(mutation):
    expected = artifact()
    actual = copy.deepcopy(expected)
    evaluation = actual["models"][0]["evaluations"][0]
    if mutation == "threshold":
        evaluation["threshold"] += 0.123
    elif mutation == "sweep":
        evaluation["sweep"]["tp"][0] += 1
    elif mutation == "ci":
        actual["primary"]["sensitivity_difference"]["ci_low"] = -999
    elif mutation == "precision":
        evaluation["metrics"]["precision"] = 0.1
    elif mutation == "calibration":
        evaluation["calibration"]["flagged_normals"] += 1
    else:
        evaluation["unregistered_field"] = True
    with pytest.raises(ValueError, match="payload differs|primary comparison differs"):
        compare_full_payloads(actual, expected)


def test_duplicate_point_cannot_replace_a_missing_hospital():
    expected = artifact()
    actual = copy.deepcopy(expected)
    actual["models"][0]["evaluations"][1] = copy.deepcopy(actual["models"][0]["evaluations"][0])
    with pytest.raises(ValueError, match="Repeated"):
        compare_full_payloads(actual, expected)


def test_incomplete_aggregate_cannot_pass():
    actual = artifact()
    actual["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        compare_full_payloads(actual, artifact())
