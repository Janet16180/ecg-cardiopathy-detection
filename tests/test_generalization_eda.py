"""Check evidence closure and source-specific selection for the explanatory notebook."""

import hashlib
import json

import pytest

from ecg_experiment.eda import generalization


def test_evidence_rejects_changed_result(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text('{}\n')
    monkeypatch.setattr(generalization, "EVIDENCE", {"transfer": str(path)})
    monkeypatch.setattr(generalization, "EXPECTED_HASHES", {"transfer": "wrong"})
    with pytest.raises(ValueError, match="Evidence hash changed"):
        generalization.load_evidence()


@pytest.mark.parametrize("flag", ["challenge_test_read", "ptbxl_test_read"])
def test_evidence_rejects_closed_data(tmp_path, monkeypatch, flag):
    path = tmp_path / "result.json"
    result = {"status": "complete", "challenge_test_read": False, "ptbxl_test_read": False}
    result[flag] = True
    path.write_text(json.dumps(result))
    monkeypatch.setattr(generalization, "EVIDENCE", {"transfer": str(path)})
    monkeypatch.setattr(
        generalization, "EXPECTED_HASHES", {"transfer": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
    with pytest.raises(ValueError, match="Closed data used"):
        generalization.load_evidence()


def test_transfer_matrix_uses_each_targets_own_holdout():
    evaluations = {
        f"family:{family}": {
            "metrics": {
                "ptbxl": {"auroc": 0.6},
                "pooled": {"auroc": 0.9},
                f"loso_{family}": {"auroc": value},
            }
        }
        for family, value in zip(generalization.FAMILIES, [0.7, 0.75, 0.8], strict=True)
    }
    result = generalization.transfer_matrix({"ranking": {"cpc": {"evaluations": evaluations}}}, "cpc")
    for display, expected in zip(generalization.FAMILIES.values(), [0.7, 0.75, 0.8], strict=True):
        assert result.loc[display, "Other families only"] == expected


def test_local_pilot_plot_excludes_source_only_threshold():
    shared = {
        "score": "pooled",
        "encoder": "xecg",
        "budget": 0.05,
        "rate_mean": 0.05,
        "rate_p5": 0.03,
        "rate_p95": 0.08,
        "sensitivity_mean": 0.77,
        "share_rate_within_1pp": 0.5,
    }
    result = generalization.pilot_curve({"summary": [dict(shared, m=0), dict(shared, m=200)]})
    assert result["Local normals"].tolist() == [200]
    assert result["Low"].tolist() == [3.0]
