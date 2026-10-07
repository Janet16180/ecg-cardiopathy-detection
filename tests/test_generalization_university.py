"""Check hypothetical workload bookkeeping and selection of the historical operating point."""

import pytest

from ecg_experiment.eda.generalization_university import budget_comparison, workload_scenarios


@pytest.fixture
def pilot_result():
    return {
        "summary": [
            {
                "score": "pooled",
                "encoder": "xecg",
                "m": 200,
                "budget": 0.05,
                "sensitivity_mean": 0.8,
                "sensitivity_ci": [0.75, 0.85],
                "rate_mean": 0.05,
            }
        ]
    }


def test_hypothetical_workload_accounts_for_every_ecg(pilot_result):
    result = workload_scenarios(pilot_result)
    assert result.groupby("Assumed positive share")["ECGs"].sum().tolist() == [1000, 1000, 1000]
    scenario = result[result["Assumed positive share"].eq("1%")].set_index("Outcome")["ECGs"]
    assert scenario["Positive annotations caught"] == 8
    assert scenario["Positive annotations missed"] == pytest.approx(2)
    assert scenario["Normal annotations referred"] == pytest.approx(49.5)


def test_workload_refuses_ambiguous_operating_points(pilot_result):
    pilot_result["summary"] *= 2
    with pytest.raises(ValueError, match="Expected one"):
        workload_scenarios(pilot_result)


def test_workload_refuses_invalid_positive_share(pilot_result):
    with pytest.raises(ValueError, match="between zero and one"):
        workload_scenarios(pilot_result, (-0.1,))


def test_budget_comparison_excludes_other_pilot_sizes_and_models(pilot_result):
    reference = pilot_result["summary"][0]
    pilot_result["summary"].extend([dict(reference, m=1000), dict(reference, encoder="cpc")])
    result = budget_comparison(pilot_result)
    assert len(result) == 1
    assert result["Caught (%)"].tolist() == [80]
    assert result["Normal referral rate (%)"].tolist() == [5]
