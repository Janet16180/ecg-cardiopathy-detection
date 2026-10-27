"""Forward admission adapter tests without model data, new inference or metrics."""

import copy
import json

import pytest

from ecg_experiment import ssl_synthetic_counterfactual_v2 as audit
from ecg_experiment.paths import to_stored


def test_forward_binding_requires_exact_original_audit_and_retained_zero_metric_failure():
    closure = {"input_sha256": {to_stored(audit.ORIGINAL_AUDIT): audit.ORIGINAL_AUDIT_SHA256}}
    failure = {"audit_completed": False, "progress": {"verified_positive_pairs": 0, "completed_profiles": []}}
    audit.require_forward_binding(closure, failure)
    with pytest.raises(ValueError, match="exact passed original"):
        audit.require_forward_binding({"input_sha256": {}}, failure)
    changed = copy.deepcopy(failure)
    changed["progress"]["verified_positive_pairs"] = 1
    with pytest.raises(ValueError, match="already derived"):
        audit.require_forward_binding(closure, changed)


def test_forward_runtime_forbids_visible_cuda_and_multiple_numerical_threads(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA hidden"):
        audit._runtime_guard()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    assert set(audit._runtime_guard().values()) == {"1"}
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="one numerical thread"):
        audit._runtime_guard()


def test_forward_failure_is_new_and_does_not_change_original_module_globals(tmp_path, monkeypatch):
    original_destination = audit.original_helpers.DESTINATION
    monkeypatch.setattr(audit, "DESTINATION", tmp_path / "new_forward")
    monkeypatch.setattr(audit, "_runtime_guard", lambda: {})

    def reject(path):
        raise ValueError("Closure binding missing")

    monkeypatch.setattr(audit, "_admission", reject)
    with pytest.raises(ValueError, match="Closure binding missing"):
        audit.audit_counterfactual(tmp_path / "closure.json")
    result = json.loads((audit.DESTINATION / "failure.json").read_text())
    assert result["progress"]["verified_positive_pairs"] == 0
    assert not result["audit_completed"]
    assert original_destination == audit.original_helpers.DESTINATION
    with pytest.raises(FileExistsError):
        audit.audit_counterfactual(tmp_path / "closure.json")
